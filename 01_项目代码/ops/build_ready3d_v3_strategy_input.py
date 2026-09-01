#!/usr/bin/env python3
"""Build the frozen candidate-level input for six-strategy evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ops.train_ready3d import load_dataset  # noqa: E402
from r3dloop.ready3d_v3.ranking import (  # noqa: E402
    add_group_context,
    combine_group_scores,
    pairwise_group_scores,
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _structure(feature: dict) -> tuple[bool, float]:
    area = float(feature["mask_area_ratio"])
    largest = float(feature["mask_largest_component_ratio"])
    border = float(feature["occlusion_border_contact_ratio"])
    passed = 0.02 <= area <= 0.92 and largest >= 0.75 and border <= 0.10
    area_score = max(0.0, 1.0 - abs(area - 0.45) / 0.47)
    score = 0.35 * area_score + 0.40 * np.clip(largest, 0.0, 1.0) + 0.25 * np.clip(
        1.0 - border / 0.10, 0.0, 1.0
    )
    return passed, float(score)


def build_strategy_rows(
    targets: list[dict],
    features: list[dict],
    v2_scores: dict[str, float],
    v3_scores: dict[str, float],
    split: str = "test",
) -> list[dict]:
    feature_by_id = {str(row["sample_id"]): row for row in features}
    output = []
    for target in targets:
        if target.get("split") != split:
            continue
        sample_id = str(target["sample_id"])
        if sample_id not in feature_by_id:
            raise ValueError(f"missing features for {sample_id}")
        if sample_id not in v2_scores or sample_id not in v3_scores:
            raise ValueError(f"missing model score for {sample_id}")
        passed, structure_score = _structure(feature_by_id[sample_id])
        output.append(
            {
                "schema_version": "r3dguard.ready3d-v3-strategy-candidate.v1",
                "group_id": str(target["group_id"]),
                "sample_id": sample_id,
                "candidate_index": int(target["candidate_index"]),
                "split": split,
                "quality_score": float(target["quality_score_v2"]),
                "qualified": bool(target["qualified"]),
                "technically_valid": bool(target["technically_valid"]),
                "structural_input_pass": passed,
                "structure_score": structure_score,
                "ready_v2_score": float(v2_scores[sample_id]),
                "ready_v3_score": float(v3_scores[sample_id]),
            }
        )
    return sorted(output, key=lambda row: (row["group_id"], row["candidate_index"]))


def _raw_matrices(data: dict) -> dict[str, np.ndarray]:
    return {
        "engineered": np.asarray(data["scalar"], dtype=np.float64),
        "dino": np.asarray(data["dino"], dtype=np.float64),
        "full": np.concatenate([data["dino"], data["scalar"]], axis=1).astype(
            np.float64
        ),
    }


def compute_model_scores(data: dict, v2_checkpoint: dict, v3_checkpoint: dict) -> tuple[dict, dict]:
    rows = data["rows"]
    ids = [str(row["sample_id"]) for row in rows]
    groups = [str(row["group_id"]) for row in rows]
    raw = _raw_matrices(data)
    v2_values = v2_checkpoint["quality_model"].predict(
        raw[v2_checkpoint["quality_feature_set"]]
    )
    contextual = {
        name: add_group_context(matrix, groups) for name, matrix in raw.items()
    }
    utility = v3_checkpoint["quality_model"].predict(
        contextual[v3_checkpoint["quality_feature_set"]]
    )
    pairwise = pairwise_group_scores(
        v3_checkpoint["ranking_model"],
        contextual[v3_checkpoint["ranking_feature_set"]],
        rows,
    )
    feature_by_id = {
        str(row["sample_id"]): row for row in read_jsonl(Path(data["features_path"]))
    } if "features_path" in data else None
    if feature_by_id is None:
        scalar_names = list(data["scalar_names"])
        columns = {name: scalar_names.index(name) for name in (
            "mask_area_ratio", "mask_largest_component_ratio", "occlusion_border_contact_ratio"
        )}
        passes = np.asarray([
            0.02 <= data["scalar"][index, columns["mask_area_ratio"]] <= 0.92
            and data["scalar"][index, columns["mask_largest_component_ratio"]] >= 0.75
            and data["scalar"][index, columns["occlusion_border_contact_ratio"]] <= 0.10
            for index in range(len(rows))
        ], dtype=bool)
    else:
        passes = np.asarray([_structure(feature_by_id[sample_id])[0] for sample_id in ids])
    v3_values = combine_group_scores(
        rows, pairwise, utility, passes, v3_checkpoint["ensemble_weights"]
    )
    return (
        {sample_id: float(value) for sample_id, value in zip(ids, v2_values, strict=True)},
        {sample_id: float(value) for sample_id, value in zip(ids, v3_values, strict=True)},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--ready-v2-checkpoint", type=Path, required=True)
    parser.add_argument("--ready-v3-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(args.targets, args.features, args.embeddings)
    v2_scores, v3_scores = compute_model_scores(
        data, joblib.load(args.ready_v2_checkpoint), joblib.load(args.ready_v3_checkpoint)
    )
    rows = build_strategy_rows(
        data["rows"], read_jsonl(args.features), v2_scores, v3_scores, args.split
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps({"split": args.split, "candidates": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
