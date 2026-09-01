from __future__ import annotations

from ops.sana_lora.calibrate_ready3d_labels import build_calibrated_labels


def make_rows():
    predictions = []
    labels = []
    for index in range(840):
        split = "train" if index < 720 else "validation"
        sample_id = f"sample_{index:04d}"
        predictions.append(
            {
                "sample_id": sample_id,
                "split": split,
                "predicted_quality": (index % 100) / 99,
            }
        )
        if (split == "train" and index < 180) or split == "validation":
            labels.append(
                {
                    "sample_id": sample_id,
                    "quality_score_v2": 0.2 + 0.6 * ((index % 100) / 99),
                }
            )
    return predictions, labels


def test_calibration_prefers_180_real_and_fills_540_pseudo_labels():
    predictions, labels = make_rows()
    rows, summary = build_calibrated_labels(
        predictions, labels, allow_pseudo_labels=True
    )
    assert len(rows) == 840
    assert summary["train_real_hunyuan"] == 180
    assert summary["train_ready3d_calibrated"] == 540
    assert summary["validation_real_hunyuan"] == 120
    assert summary["final_test_labels"] == 0
    assert all(0 <= row["quality"] <= 1 for row in rows)


def test_failed_ready3d_gate_keeps_only_real_training_labels():
    predictions, labels = make_rows()
    rows, summary = build_calibrated_labels(
        predictions, labels, allow_pseudo_labels=False
    )
    assert len(rows) == 300
    assert summary["train_ready3d_calibrated"] == 0
