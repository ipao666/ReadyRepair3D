#!/usr/bin/env python3
"""Batch frozen-Ready3D scoring and severe input QA for external64."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

OPS = Path(__file__).resolve().parent / "ops"
if not OPS.is_dir():
    OPS = Path("/root/r3dguard/ops")
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

from predict_ready3d import (
    apply_quality_first_selection,
    assemble_matrices,
    predict_arrays,
    severe_input_flags,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_indices(rows: list[dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row["group_id"]), []).append(index)
    for group_id, indices in groups.items():
        indices.sort(key=lambda index: int(rows[index]["candidate_index"]))
        candidates = [int(rows[index]["candidate_index"]) for index in indices]
        if candidates != [0, 1, 2, 3]:
            raise ValueError(f"{group_id}: expected four candidates 0..3, got {candidates}")
    return dict(sorted(groups.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("/root/r3dguard/data/ready3d_external64_new/manifest.jsonl"),
    )
    parser.add_argument(
        "--features", type=Path,
        default=Path("/root/r3dguard/data/ready3d_external64_new/features/features.jsonl"),
    )
    parser.add_argument(
        "--embeddings", type=Path,
        default=Path("/root/r3dguard/data/ready3d_external64_new/features/dino_embeddings.npy"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("/root/r3dguard/checkpoints/ready3d_v2/ready3d_v2.joblib"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/root/r3dguard/evaluation/ready3d_external64_new/input_scoring"),
    )
    parser.add_argument("--top-k", type=int, choices=[1, 2], default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_jsonl(args.manifest)
    features = read_jsonl(args.features)
    if [row["sample_id"] for row in manifest] != [row["sample_id"] for row in features]:
        raise ValueError("manifest and feature sample order differ")
    checkpoint = joblib.load(args.checkpoint)
    embeddings = np.load(args.embeddings)
    matrices = assemble_matrices(features, embeddings, checkpoint["scalar_feature_names"])
    groups = group_indices(manifest)
    output_rows = []
    for group_id, indices in groups.items():
        group_matrices = {name: matrix[indices] for name, matrix in matrices.items()}
        predictions = apply_quality_first_selection(
            predict_arrays(checkpoint, group_matrices),
            [features[index] for index in indices],
            top_k=args.top_k,
        )
        for local_index, prediction in enumerate(predictions):
            global_index = indices[local_index]
            output_rows.append(
                {
                    "sample_id": manifest[global_index]["sample_id"],
                    "group_id": group_id,
                    "candidate_index": int(manifest[global_index]["candidate_index"]),
                    "source_path": manifest[global_index]["source_path"],
                    **prediction,
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    flag_counts = Counter(flag for row in output_rows for flag in row["severe_input_flags"])
    clean_groups = {
        group_id for group_id in groups
        if any(row["structural_input_pass"] for row in output_rows if row["group_id"] == group_id)
    }
    selected_rows = [row for row in output_rows if row["selected"]]
    summary = {
        "schema_version": "r3dguard.external-input-scoring.v1",
        "checkpoint": str(args.checkpoint),
        "scoring_version": checkpoint.get("scoring_version", "v1_legacy"),
        "groups": len(groups),
        "candidates": len(output_rows),
        "top_k": args.top_k,
        "structural_pass_candidates": sum(row["structural_input_pass"] for row in output_rows),
        "groups_with_clean_candidate": len(clean_groups),
        "all_groups_have_clean_candidate": len(clean_groups) == len(groups),
        "flag_counts": dict(flag_counts),
        "selected_sample_ids": [row["sample_id"] for row in selected_rows],
        "mean_predicted_quality": float(np.mean([row["predicted_quality"] for row in output_rows])),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
