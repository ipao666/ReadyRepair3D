#!/usr/bin/env python3
"""Evaluate six frozen Ready3D selection strategies on test groups only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


STRATEGIES = (
    "direct",
    "random_top1",
    "structure_top1",
    "ready_v2_top1",
    "ready_v3_top1",
    "oracle_all4",
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _deterministic_random_index(group_id: str, seed: int, count: int) -> int:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def _highest(rows: list[dict], field: str, eligible_only: bool = True) -> dict:
    eligible = [row for row in rows if bool(row["structural_input_pass"])]
    pool = eligible if eligible_only and eligible else rows
    return min(
        pool,
        key=lambda row: (-float(row[field]), int(row["candidate_index"])),
    )


def _select(rows: list[dict], strategy: str, seed: int) -> dict:
    ordered = sorted(rows, key=lambda row: int(row["candidate_index"]))
    if strategy == "direct":
        return ordered[0]
    if strategy == "random_top1":
        return ordered[_deterministic_random_index(str(rows[0]["group_id"]), seed, 4)]
    if strategy == "structure_top1":
        return _highest(rows, "structure_score")
    if strategy == "ready_v2_top1":
        return _highest(rows, "ready_v2_score")
    if strategy == "ready_v3_top1":
        return _highest(rows, "ready_v3_score")
    if strategy == "oracle_all4":
        return _highest(rows, "quality_score", eligible_only=False)
    raise ValueError(f"unknown strategy: {strategy}")


def _bootstrap_mean_ci(values: np.ndarray, draws: int, rng: np.random.Generator) -> list[float]:
    if draws <= 0:
        raise ValueError("bootstrap_draws must be positive")
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(sampled, 2.5)), float(np.percentile(sampled, 97.5))]


def evaluate_strategies(
    rows: list[dict], bootstrap_draws: int = 10_000, seed: int = 20260827
) -> dict:
    if not rows:
        raise ValueError("at least one test group is required")
    if any(row.get("split") != "test" for row in rows):
        raise ValueError("strategy evaluation accepts test split only")
    required = {
        "group_id",
        "sample_id",
        "candidate_index",
        "quality_score",
        "qualified",
        "technically_valid",
        "structural_input_pass",
        "structure_score",
        "ready_v2_score",
        "ready_v3_score",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"missing fields: {missing}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["group_id"])].append(row)
    for group_id, candidates in groups.items():
        indices = sorted(int(row["candidate_index"]) for row in candidates)
        if indices != [0, 1, 2, 3]:
            raise ValueError(f"{group_id}: expected candidate indices [0, 1, 2, 3]")

    group_results = []
    for group_id, candidates in sorted(groups.items()):
        optimum = _highest(candidates, "quality_score", eligible_only=False)
        optimal_quality = float(optimum["quality_score"])
        for strategy in STRATEGIES:
            selected = _select(candidates, strategy, seed)
            selected_quality = float(selected["quality_score"])
            group_results.append(
                {
                    "group_id": group_id,
                    "strategy": strategy,
                    "selected_sample_id": str(selected["sample_id"]),
                    "selected_quality": selected_quality,
                    "qualified": bool(selected["qualified"]),
                    "technically_valid": bool(selected["technically_valid"]),
                    "optimal_sample_id": str(optimum["sample_id"]),
                    "optimal_quality": optimal_quality,
                    "regret": optimal_quality - selected_quality,
                    "top1_correct": abs(selected_quality - optimal_quality) <= 1e-8,
                }
            )

    metrics = {}
    records_by_strategy = {
        strategy: [row for row in group_results if row["strategy"] == strategy]
        for strategy in STRATEGIES
    }
    for strategy, records in records_by_strategy.items():
        metrics[strategy] = {
            "groups": len(records),
            "mean_quality": float(np.mean([row["selected_quality"] for row in records])),
            "qualified_rate": float(np.mean([row["qualified"] for row in records])),
            "technical_valid_rate": float(
                np.mean([row["technically_valid"] for row in records])
            ),
            "top1_accuracy": float(np.mean([row["top1_correct"] for row in records])),
            "mean_regret": float(np.mean([row["regret"] for row in records])),
            "quality_capture": float(
                np.mean(
                    [
                        row["selected_quality"] / max(row["optimal_quality"], 1e-8)
                        for row in records
                    ]
                )
            ),
            "two_d_candidates_per_group": 4,
            "three_d_calls_per_group": 4 if strategy == "oracle_all4" else 1,
        }

    direct = {
        row["group_id"]: row for row in records_by_strategy["direct"]
    }
    comparisons = {}
    for strategy in STRATEGIES:
        if strategy == "direct":
            continue
        deltas = np.asarray(
            [
                row["selected_quality"] - direct[row["group_id"]]["selected_quality"]
                for row in records_by_strategy[strategy]
            ],
            dtype=np.float64,
        )
        rng = np.random.default_rng(seed)
        comparisons[strategy] = {
            "mean_quality_delta": float(deltas.mean()),
            "mean_quality_delta_95_ci": _bootstrap_mean_ci(
                deltas, bootstrap_draws, rng
            ),
            "qualified_rate_delta": metrics[strategy]["qualified_rate"]
            - metrics["direct"]["qualified_rate"],
        }
    return {
        "schema_version": "r3dguard.ready3d-v3-six-strategy-eval.v1",
        "evaluation_split": "test",
        "test_used_for_model_or_threshold_selection": False,
        "groups": len(groups),
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": seed,
        "strategies": metrics,
        "comparisons_vs_direct": comparisons,
        "group_results": group_results,
    }


def metrics_csv(report: dict) -> str:
    fields = [
        "strategy",
        "groups",
        "mean_quality",
        "qualified_rate",
        "technical_valid_rate",
        "top1_accuracy",
        "mean_regret",
        "quality_capture",
        "two_d_candidates_per_group",
        "three_d_calls_per_group",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for strategy in STRATEGIES:
        writer.writerow({"strategy": strategy, **report["strategies"][strategy]})
    return buffer.getvalue()


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_strategies(
        read_jsonl(args.input), args.bootstrap_draws, args.seed
    )
    write_atomic(
        args.output_dir / "strategy_metrics.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    write_atomic(args.output_dir / "strategy_metrics.csv", metrics_csv(report))
    with io.StringIO() as buffer:
        for row in report["group_results"]:
            buffer.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_atomic(args.output_dir / "strategy_group_results.jsonl", buffer.getvalue())
    print(json.dumps(report["strategies"], ensure_ascii=False))


if __name__ == "__main__":
    main()
