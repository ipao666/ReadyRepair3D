from __future__ import annotations

import pytest

from ops.sana_lora.evaluate_lora_candidates import rank_checkpoints


def metric(checkpoint: str, quality: float, adherence: float, valid: bool = True) -> dict:
    return {
        "checkpoint": checkpoint,
        "sample_id": f"{checkpoint}_{quality}",
        "split": "validation",
        "ready3d_quality": quality,
        "prompt_adherence": adherence,
        "technical_valid": valid,
    }


def test_selects_two_checkpoints_without_test_data():
    rows = [
        metric("checkpoint-250", 0.5, 0.8),
        metric("checkpoint-500", 0.8, 0.7),
        metric("checkpoint-750", 0.7, 0.9),
    ]
    ranked = rank_checkpoints(rows)
    assert sum(row["selected_for_hunyuan_validation"] for row in ranked) == 2
    assert ranked[0]["checkpoint"] == "checkpoint-500"
    assert all(row["test_used_for_selection"] is False for row in ranked)


def test_rejects_test_rows_during_checkpoint_selection():
    row = metric("checkpoint-250", 0.5, 0.8)
    row["split"] = "final_test"
    with pytest.raises(ValueError, match="validation"):
        rank_checkpoints([row])
