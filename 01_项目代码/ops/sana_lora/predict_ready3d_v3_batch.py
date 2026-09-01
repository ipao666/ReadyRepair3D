#!/usr/bin/env python3
"""Apply a frozen Ready3D V3 checkpoint to precomputed candidate features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from ops.predict_ready3d import assemble_matrices  # noqa: E402
from ops.train_ready3d_v3_top1 import structural_pass_from_scalar  # noqa: E402
from r3dloop.ready3d_v3.ranking import (  # noqa: E402
    add_group_context,
    combine_group_scores,
    pairwise_group_scores,
    select_top1,
    validate_complete_groups,
)
from r3dloop.sana_lora.data_protocol import read_jsonl, write_jsonl_atomic  # noqa: E402


def predict_candidates(
    candidates: list[dict],
    features: list[dict],
    embeddings: np.ndarray,
    checkpoint: dict,
    *,
    splits: set[str],
) -> tuple[list[dict], list[dict]]:
    if checkpoint.get("schema_version") != "r3dguard.ready3d-v3-top1-checkpoint.v1":
        raise ValueError("not a frozen Ready3D V3 Top-1 checkpoint")
    rows = [row for row in candidates if str(row.get("split")) in splits]
    validate_complete_groups(rows)
    feature_by_id = {str(row["sample_id"]): row for row in features}
    if len(feature_by_id) != len(features):
        raise ValueError("duplicate feature sample_id")
    aligned_features: list[dict] = []
    aligned_embeddings: list[np.ndarray] = []
    for row in rows:
        feature = feature_by_id.get(str(row["sample_id"]))
        if feature is None:
            raise ValueError(f"missing features for {row['sample_id']}")
        index = int(feature["feature_index"])
        if not 0 <= index < len(embeddings):
            raise ValueError(f"invalid feature_index for {row['sample_id']}")
        aligned_features.append(feature)
        aligned_embeddings.append(embeddings[index])
    scalar_names = list(checkpoint["scalar_feature_names"])
    raw = assemble_matrices(aligned_features, np.asarray(aligned_embeddings), scalar_names)
    groups = [str(row["group_id"]) for row in rows]
    matrices = {name: add_group_context(values, groups) for name, values in raw.items()}
    quality = checkpoint["quality_model"].predict(matrices[checkpoint["quality_feature_set"]])
    pairwise = pairwise_group_scores(
        checkpoint["ranking_model"], matrices[checkpoint["ranking_feature_set"]], rows
    )
    scalar = raw["engineered"]
    structural_pass = structural_pass_from_scalar(scalar, scalar_names)
    ensemble = combine_group_scores(
        rows, pairwise, quality, structural_pass, checkpoint["ensemble_weights"]
    )
    selections = select_top1(rows, ensemble, structural_pass)
    selected_ids = {row["selected_sample_id"] for row in selections}
    predictions = []
    for index, row in enumerate(rows):
        predictions.append(
            {
                **row,
                "predicted_quality": float(np.clip(quality[index], 0.0, 1.0)),
                "raw_predicted_quality": float(quality[index]),
                "pairwise_score": float(pairwise[index]),
                "ensemble_score": float(ensemble[index]),
                "structural_input_pass": bool(structural_pass[index]),
                "selected_for_3d": str(row["sample_id"]) in selected_ids,
                "label_source": "ready3d_v3_calibrated",
            }
        )
    return predictions, selections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--splits", default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    if not splits or not splits <= {"train", "validation", "dev_test"}:
        raise ValueError(f"invalid requested splits: {sorted(splits)}")
    predictions, selections = predict_candidates(
        read_jsonl(args.candidates),
        read_jsonl(args.features),
        np.load(args.embeddings),
        joblib.load(args.checkpoint),
        splits=splits,
    )
    write_jsonl_atomic(args.output, predictions)
    write_jsonl_atomic(args.selections, selections)
    print(
        json.dumps(
            {
                "candidates": len(predictions),
                "groups": len(selections),
                "splits": sorted(splits),
                "final_test_used": False,
            }
        )
    )


if __name__ == "__main__":
    main()
