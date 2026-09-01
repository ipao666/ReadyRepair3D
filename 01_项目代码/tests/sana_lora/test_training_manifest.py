from __future__ import annotations

import pytest

from ops.sana_lora.build_training_manifest import join_training_manifest


def candidate(index: int, split: str = "train") -> dict:
    return {
        "sample_id": f"group_000_c{index}",
        "prompt_group_id": "group_000",
        "split": split,
        "prompt_zh": "单个蓝色金属台灯，灯罩、灯杆和底座完整，居中，纯色背景",
        "caption_en": "a complete blue metal desk lamp, isolated single object",
        "seed": 2026082800 + index,
        "image_path": f"images/group_000_c{index}.png",
        "base_model": "SANA1.5_1.6B_1024px_diffusers",
    }


def test_builds_weighted_training_rows_and_prefers_hunyuan():
    candidates = [candidate(index) for index in range(4)]
    labels = [
        {
            "sample_id": row["sample_id"],
            "hunyuan_quality": 0.8 if index == 0 else None,
            "ready3d_calibrated_quality": 0.6,
        }
        for index, row in enumerate(candidates)
    ]
    adherence = [
        {"sample_id": row["sample_id"], "prompt_adherence": 0.5}
        for row in candidates
    ]

    rows = join_training_manifest(candidates, labels, adherence, expected=4)

    assert len(rows) == 4
    assert rows[0]["label_source"] == "hunyuan_frozen_q"
    assert all(0.25 <= row["sample_weight"] <= 1.5 for row in rows)


def test_excludes_validation_and_test_candidates():
    candidates = [candidate(0), candidate(1, "validation"), candidate(2, "dev_test")]
    labels = [{"sample_id": "group_000_c0", "hunyuan_quality": 0.7}]
    adherence = [{"sample_id": "group_000_c0", "adherence_missing": True}]

    rows = join_training_manifest(candidates, labels, adherence, expected=1)

    assert [row["sample_id"] for row in rows] == ["group_000_c0"]
    assert rows[0]["prompt_adherence"] is None


def test_rejects_missing_labels_and_duplicate_adherence():
    with pytest.raises(ValueError, match="missing 3D label"):
        join_training_manifest([candidate(0)], [], [], expected=1)
    duplicate = {"sample_id": "group_000_c0", "prompt_adherence": 0.5}
    with pytest.raises(ValueError, match="duplicate"):
        join_training_manifest(
            [candidate(0)],
            [{"sample_id": "group_000_c0", "hunyuan_quality": 0.7}],
            [duplicate, duplicate],
            expected=1,
        )
