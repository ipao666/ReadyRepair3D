import json
from pathlib import Path

import pytest

from r3dloop.ready3d_v3.data_protocol import (
    build_optimizer_inputs,
    expand_candidates,
    read_jsonl,
    subject_fingerprint,
    validate_prompt_groups,
)


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "examples" / "ready3d_v3" / "prompts80.jsonl"


def make_groups(count=80):
    splits = ["train"] * 56 + ["validation"] * 12 + ["test"] * 12
    risks = [
        "low_risk",
        "thin_parts",
        "multi_limb",
        "open_structure",
        "asymmetry",
        "reflective",
        "complex_support",
        "repeated_texture",
    ]
    return [
        {
            "schema_version": "r3dguard.ready3d-v3-prompt-group.v1",
            "group_id": f"v3_{index:03d}",
            "split": splits[index],
            "category": f"category_{index % 10}",
            "difficulty": ("easy", "medium", "hard")[index % 3],
            "risk_tags": [risks[index % len(risks)]],
            "subject_zh": f"测试物体{index}",
            "prompt_zh": f"单个测试物体{index}，主体结构完整并清晰可见，居中且不触边，三分之四产品视角，纯色背景，无文字无其他物体",
        }
        for index in range(count)
    ]


def test_subject_fingerprint_ignores_spacing_and_punctuation():
    assert subject_fingerprint("蓝色， 城市自行车！") == subject_fingerprint("蓝色城市自行车")


def test_protocol_requires_80_groups_and_fixed_split_counts():
    summary = validate_prompt_groups(make_groups())
    assert summary["groups"] == 80
    assert summary["split_groups"] == {"train": 56, "validation": 12, "test": 12}
    assert summary["risk_tags"] == [
        "asymmetry",
        "complex_support",
        "low_risk",
        "multi_limb",
        "open_structure",
        "reflective",
        "repeated_texture",
        "thin_parts",
    ]


def test_protocol_rejects_subject_leakage_across_splits():
    groups = make_groups()
    groups[-1]["subject_zh"] = groups[0]["subject_zh"]
    with pytest.raises(ValueError, match="subject leakage"):
        validate_prompt_groups(groups)


def test_candidate_expansion_is_deterministic_and_has_four_seeds_per_group():
    groups = make_groups()
    first = expand_candidates(groups, base_seed=2026082700)
    second = expand_candidates(groups, base_seed=2026082700)
    assert first == second
    assert len(first) == 320
    assert [row["seed"] for row in first[:4]] == [2026082700, 2026082701, 2026082702, 2026082703]
    assert [row["candidate_index"] for row in first[:4]] == [0, 1, 2, 3]
    assert len({row["sample_id"] for row in first}) == 320


def test_real_prompt_catalog_is_complete_and_valid():
    rows = read_jsonl(PROMPTS)
    summary = validate_prompt_groups(rows)
    candidates = expand_candidates(rows)
    assert summary["groups"] == 80
    assert len(candidates) == 320
    assert len({row["category"] for row in rows}) >= 10
    assert {row["difficulty"] for row in rows} == {"easy", "medium", "hard"}
    assert all(len(row["prompt_zh"]) >= 30 for row in rows)


def test_jsonl_reader_rejects_duplicate_keys_or_non_objects(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps([1, 2, 3]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        read_jsonl(path)


def test_optimizer_input_uses_group_id_as_stable_prompt_id():
    rows = make_groups()
    optimized_input = build_optimizer_inputs(rows)

    assert optimized_input[0]["prompt_id"] == rows[0]["group_id"]
    assert optimized_input[0]["prompt_zh"] == rows[0]["prompt_zh"]
