from __future__ import annotations

import json
from pathlib import Path

from r3dloop.sana_lora.data_protocol import read_jsonl, validate_prompt_catalog


ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT = ROOT / "examples" / "sana_lora" / "prompts240.jsonl"
FINAL = ROOT / "examples" / "sana_lora" / "final_test_prompts64.jsonl"
READY3D = ROOT / "examples" / "ready3d_v3" / "prompts80.jsonl"


def test_prompt_catalog_counts_and_splits_are_frozen():
    development = read_jsonl(DEVELOPMENT)
    final = read_jsonl(FINAL)

    dev_summary = validate_prompt_catalog(
        development,
        expected_split_counts={"train": 180, "validation": 30, "dev_test": 30},
    )
    final_summary = validate_prompt_catalog(final, expected_split_counts={"final_test": 64})

    assert dev_summary["groups"] == 240
    assert final_summary["groups"] == 64
    assert len(dev_summary["categories"]) >= 15


def test_development_final_and_ready3d_subjects_do_not_overlap():
    development = read_jsonl(DEVELOPMENT)
    final = read_jsonl(FINAL)
    ready3d = [json.loads(line) for line in READY3D.read_text(encoding="utf-8").splitlines() if line]

    dev_subjects = {row["subject_key"] for row in development}
    final_subjects = {row["subject_key"] for row in final}
    ready3d_subjects = {row["subject_zh"] for row in ready3d}

    assert dev_subjects.isdisjoint(final_subjects)
    assert (dev_subjects | final_subjects).isdisjoint(ready3d_subjects)
    assert len({row["prompt_group_id"] for row in development + final}) == 304
