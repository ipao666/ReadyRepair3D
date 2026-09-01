#!/usr/bin/env python3
"""Offline paired evaluation of Ready3D Top-1 versus Top-2 plus 3D scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate(
    predictions: list[dict],
    labels: list[dict],
    bootstrap_draws: int = 10_000,
    seed: int = 20260717,
) -> dict:
    labels_by_id = {str(row["sample_id"]): row for row in labels}
    if len(labels_by_id) != len(labels):
        raise ValueError("quality labels contain duplicate sample IDs")
    groups: dict[str, list[dict]] = {}
    for row in predictions:
        groups.setdefault(str(row["group_id"]), []).append(row)
    group_results = []
    for group_id in sorted(groups):
        selected = [row for row in groups[group_id] if row.get("selected") is True]
        ranks = sorted(row.get("selection_rank") for row in selected)
        if ranks != [1, 2]:
            raise ValueError(
                f"{group_id} selected candidates must have ranks [1, 2], got {ranks}"
            )
        selected.sort(key=lambda row: row["selection_rank"])
        selected_labels = []
        for row in selected:
            sample_id = str(row["sample_id"])
            if sample_id not in labels_by_id:
                raise ValueError(f"missing quality label for {sample_id}")
            selected_labels.append(labels_by_id[sample_id])
        top1 = selected_labels[0]
        dual = max(
            selected_labels,
            key=lambda row: (
                bool(row["technically_valid"]),
                float(row["quality_score_v2"]),
            ),
        )
        top1_quality = float(top1["quality_score_v2"])
        dual_quality = float(dual["quality_score_v2"])
        group_results.append(
            {
                "group_id": group_id,
                "top1_sample_id": top1["sample_id"],
                "top2_dual_sample_id": dual["sample_id"],
                "top1_quality": top1_quality,
                "top2_dual_quality": dual_quality,
                "gain": dual_quality - top1_quality,
                "top1_technically_valid": bool(top1["technically_valid"]),
                "top2_dual_technically_valid": bool(
                    dual["technically_valid"]
                ),
            }
        )
    if not group_results:
        raise ValueError("no prediction groups found")
    gains = np.asarray([row["gain"] for row in group_results], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        gains, size=(bootstrap_draws, len(gains)), replace=True
    ).mean(axis=1)
    mean_top1 = float(
        np.mean([row["top1_quality"] for row in group_results])
    )
    mean_dual = float(
        np.mean([row["top2_dual_quality"] for row in group_results])
    )
    valid_top1 = float(
        np.mean([row["top1_technically_valid"] for row in group_results])
    )
    valid_dual = float(
        np.mean([row["top2_dual_technically_valid"] for row in group_results])
    )
    return {
        "schema_version": "r3dguard.top2-dual3d-offline-eval.v1",
        "groups": len(group_results),
        "mean_top1_quality": mean_top1,
        "mean_top2_dual_quality": mean_dual,
        "mean_gain": mean_dual - mean_top1,
        "gain_bootstrap_95_ci": [
            float(np.percentile(sampled, 2.5)),
            float(np.percentile(sampled, 97.5)),
        ],
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": seed,
        "positive_groups": int(np.sum(gains > 1e-12)),
        "zero_groups": int(np.sum(np.abs(gains) <= 1e-12)),
        "negative_groups": int(np.sum(gains < -1e-12)),
        "top1_technical_valid_rate": valid_top1,
        "top2_dual_technical_valid_rate": valid_dual,
        "stage_gate_passed": bool(
            mean_dual >= mean_top1 and valid_dual >= valid_top1
        ),
        "group_results": group_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        read_jsonl(args.predictions),
        read_jsonl(args.labels),
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "groups",
                    "mean_top1_quality",
                    "mean_top2_dual_quality",
                    "mean_gain",
                    "gain_bootstrap_95_ci",
                    "top1_technical_valid_rate",
                    "top2_dual_technical_valid_rate",
                    "stage_gate_passed",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["stage_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
