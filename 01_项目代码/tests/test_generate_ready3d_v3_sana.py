from pathlib import Path

import pytest

from ops.generate_ready3d_v3_sana import (
    ROOT,
    build_generation_records,
    default_model_path,
    pending_records,
)


def _candidates():
    return [
        {
            "group_id": "v3_000",
            "sample_id": f"v3_000_c{i}_s{100 + i}",
            "candidate_index": i,
            "seed": 100 + i,
            "filename": f"c{i}.png",
            "prompt_zh": "一个蓝色茶壶",
        }
        for i in range(4)
    ]


def test_join_preserves_candidate_identity_and_uses_optimized_sana_prompt(tmp_path):
    optimized = [
        {
            "prompt_id": "v3_000",
            "sana_prompt": "A single blue teapot, complete object, studio background",
            "pipeline_ready": True,
        }
    ]

    rows = build_generation_records(_candidates(), optimized, tmp_path / "images")

    assert [row["seed"] for row in rows] == [100, 101, 102, 103]
    assert rows[0]["sample_id"] == "v3_000_c0_s100"
    assert rows[0]["prompt"].startswith("A single blue teapot")
    assert rows[0]["source_path"] == str((tmp_path / "images" / "c0.png").resolve())


def test_join_rejects_missing_or_unvalidated_optimizer_output(tmp_path):
    with pytest.raises(ValueError, match="missing optimized prompt"):
        build_generation_records(_candidates(), [], tmp_path)
    with pytest.raises(ValueError, match="pipeline_ready"):
        build_generation_records(
            _candidates(),
            [{"prompt_id": "v3_000", "sana_prompt": "x", "pipeline_ready": False}],
            tmp_path,
        )


def test_pending_records_are_resume_safe(tmp_path):
    rows = build_generation_records(
        _candidates(),
        [{"prompt_id": "v3_000", "sana_prompt": "optimized", "pipeline_ready": True}],
        tmp_path,
    )
    existing = Path(rows[0]["source_path"])
    existing.write_bytes(b"not validated in this unit test")

    pending = pending_records(rows, validator=lambda path: path != existing)

    assert [row["sample_id"] for row in pending] == [rows[0]["sample_id"]]


def test_default_model_path_is_project_relative(monkeypatch):
    monkeypatch.delenv("R3D_SANA_MODEL", raising=False)

    assert default_model_path() == ROOT / "models" / "SANA1.5_1.6B_1024px_diffusers"
