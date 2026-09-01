#!/usr/bin/env python3
"""Calibrate Ready3D V3 on real validation labels and build training labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from r3dloop.sana_lora.data_protocol import read_jsonl, write_jsonl_atomic  # noqa: E402


def _quality(row: dict) -> float:
    for field in ("quality", "quality_score_v2", "hunyuan_quality"):
        if row.get(field) is not None:
            value = float(row[field])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} outside [0, 1] for {row.get('sample_id')}")
            return value
    raise ValueError(f"missing Hunyuan quality for {row.get('sample_id')}")


def build_calibrated_labels(
    predictions: list[dict],
    hunyuan_labels: list[dict],
    *,
    allow_pseudo_labels: bool,
) -> tuple[list[dict], dict]:
    prediction_by_id = {str(row["sample_id"]): row for row in predictions}
    label_by_id = {str(row["sample_id"]): row for row in hunyuan_labels}
    if len(prediction_by_id) != len(predictions) or len(label_by_id) != len(hunyuan_labels):
        raise ValueError("duplicate sample IDs in predictions or Hunyuan labels")
    validation = [
        row
        for row in predictions
        if row.get("split") == "validation" and str(row["sample_id"]) in label_by_id
    ]
    if len(validation) != 120:
        raise ValueError(f"expected 120 validation calibration pairs, found {len(validation)}")
    x = np.asarray([float(row["predicted_quality"]) for row in validation])
    y = np.asarray([_quality(label_by_id[str(row["sample_id"])]) for row in validation])
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(x, y)
    output = []
    train_predictions = [row for row in predictions if row.get("split") == "train"]
    if len(train_predictions) != 720:
        raise ValueError(f"expected 720 train predictions, found {len(train_predictions)}")
    for prediction in train_predictions:
        sample_id = str(prediction["sample_id"])
        if sample_id in label_by_id:
            output.append(
                {
                    "sample_id": sample_id,
                    "split": "train",
                    "quality": _quality(label_by_id[sample_id]),
                    "hunyuan_quality": _quality(label_by_id[sample_id]),
                    "label_source": "hunyuan_frozen_q",
                }
            )
        elif allow_pseudo_labels:
            calibrated = float(calibrator.predict([float(prediction["predicted_quality"])])[0])
            output.append(
                {
                    "sample_id": sample_id,
                    "split": "train",
                    "quality": calibrated,
                    "ready3d_calibrated_quality": calibrated,
                    "ready3d_raw_prediction": float(prediction["predicted_quality"]),
                    "label_source": "ready3d_v3_calibrated",
                }
            )
    for prediction in validation:
        sample_id = str(prediction["sample_id"])
        output.append(
            {
                "sample_id": sample_id,
                "split": "validation",
                "quality": _quality(label_by_id[sample_id]),
                "hunyuan_quality": _quality(label_by_id[sample_id]),
                "label_source": "hunyuan_frozen_q",
            }
        )
    output.sort(key=lambda row: (str(row["split"]), str(row["sample_id"])))
    train_real = sum(
        row["split"] == "train" and row["label_source"] == "hunyuan_frozen_q"
        for row in output
    )
    train_pseudo = sum(row["label_source"] == "ready3d_v3_calibrated" for row in output)
    if train_real != 180:
        raise ValueError(f"expected 180 real train labels, found {train_real}")
    expected_pseudo = 540 if allow_pseudo_labels else 0
    if train_pseudo != expected_pseudo:
        raise ValueError(f"expected {expected_pseudo} pseudo labels, found {train_pseudo}")
    summary = {
        "calibration_split": "validation",
        "calibration_pairs": 120,
        "calibration_method": "isotonic_regression",
        "train_real_hunyuan": train_real,
        "train_ready3d_calibrated": train_pseudo,
        "validation_real_hunyuan": 120,
        "pseudo_labels_allowed": allow_pseudo_labels,
        "final_test_labels": 0,
        "calibration_mae": float(np.mean(np.abs(calibrator.predict(x) - y))),
        "x_thresholds": [float(value) for value in calibrator.X_thresholds_],
        "y_thresholds": [float(value) for value in calibrator.y_thresholds_],
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--hunyuan-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--allow-pseudo-labels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build_calibrated_labels(
        read_jsonl(args.predictions),
        read_jsonl(args.hunyuan_labels),
        allow_pseudo_labels=args.allow_pseudo_labels,
    )
    write_jsonl_atomic(args.output, rows)
    args.calibration.parent.mkdir(parents=True, exist_ok=True)
    args.calibration.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
