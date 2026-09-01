from __future__ import annotations

import json

import pytest

from ops.sana_lora.summarize_stage04 import REQUIRED_STEPS, summarize


def test_stage04_requires_all_six_checkpoints(tmp_path):
    for step in REQUIRED_STEPS:
        directory = tmp_path / f"checkpoint-{step}"
        directory.mkdir()
        (directory / "pytorch_lora_weights.safetensors").write_bytes(b"weights")
    (tmp_path / "training_summary.json").write_text(
        json.dumps({"steps": 1500, "peak_memory_mib": 12000}), encoding="utf-8"
    )
    report = summarize(tmp_path)
    assert [row["step"] for row in report["checkpoints"]] == list(REQUIRED_STEPS)
    assert report["final_test_used"] is False
    (tmp_path / "checkpoint-750" / "pytorch_lora_weights.safetensors").unlink()
    with pytest.raises(ValueError, match="step 750"):
        summarize(tmp_path)
