#!/usr/bin/env python3
"""Select 180 stratified train candidates and all 120 validation candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from r3dloop.sana_lora.data_protocol import read_jsonl, write_jsonl_atomic


def _stable_key(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def _risk_type(row: dict, feature: dict) -> str:
    if not bool(row.get("structural_input_pass", True)):
        return "structural_failure"
    if float(feature.get("mask_convex_fill_ratio", 1.0)) < 0.75:
        return "open_or_thin"
    if float(feature.get("depth_discontinuity_ratio", 0.0)) > 0.03:
        return "high_detail"
    return "low_risk"


def _quality_bins(rows: list[dict], bins: int = 5) -> dict[str, int]:
    ordered = sorted(
        rows,
        key=lambda row: (float(row["predicted_quality"]), str(row["sample_id"])),
    )
    return {
        str(row["sample_id"]): min(bins - 1, index * bins // len(ordered))
        for index, row in enumerate(ordered)
    }


def select_train_subset(
    predictions: list[dict], features: list[dict], *, count: int = 180, seed: int = 20260831
) -> list[dict]:
    train = [row for row in predictions if row.get("split") == "train"]
    if len(train) != 720:
        raise ValueError(f"expected 720 train predictions, found {len(train)}")
    feature_by_id = {str(row["sample_id"]): row for row in features}
    bins = _quality_bins(train)
    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    enriched = []
    for row in train:
        sample_id = str(row["sample_id"])
        feature = feature_by_id.get(sample_id)
        if feature is None:
            raise ValueError(f"missing feature row for {sample_id}")
        item = {
            **row,
            "quality_quantile": bins[sample_id],
            "risk_type": _risk_type(row, feature),
        }
        enriched.append(item)
        cells[(str(row["category"]), bins[sample_id])].append(item)
    allocation = {cell: count * len(rows) // len(train) for cell, rows in cells.items()}
    remaining = count - sum(allocation.values())
    remainders = sorted(
        cells,
        key=lambda cell: (
            -(count * len(cells[cell]) % len(train)),
            cell,
        ),
    )
    for cell in remainders[:remaining]:
        allocation[cell] += 1
    selected = []
    for cell in sorted(cells):
        by_risk: dict[str, list[dict]] = defaultdict(list)
        for row in cells[cell]:
            by_risk[str(row["risk_type"])].append(row)
        for risk_rows in by_risk.values():
            risk_rows.sort(key=lambda row: _stable_key(str(row["sample_id"]), seed))
        ordered = []
        while any(by_risk.values()):
            for risk in sorted(by_risk):
                if by_risk[risk]:
                    ordered.append(by_risk[risk].pop(0))
        selected.extend(ordered[: allocation[cell]])
    if len(selected) != count or len({row["sample_id"] for row in selected}) != count:
        raise RuntimeError("stratified selection did not produce the requested unique count")
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def build_queue(
    candidates: list[dict],
    predictions: list[dict],
    features: list[dict],
    *,
    train_count: int = 180,
    seed: int = 20260831,
) -> tuple[list[dict], dict]:
    selected_train = select_train_subset(
        predictions, features, count=train_count, seed=seed
    )
    validation = [row for row in candidates if row.get("split") == "validation"]
    if len(validation) != 120:
        raise ValueError(f"expected 120 validation candidates, found {len(validation)}")
    train_ids = {str(row["sample_id"]): row for row in selected_train}
    queue = []
    for row in candidates:
        sample_id = str(row["sample_id"])
        if sample_id in train_ids:
            selection = train_ids[sample_id]
            queue.append(
                {
                    **row,
                    "label_source": "hunyuan_frozen_q_pending",
                    "sampling_reason": "stratified_train_subset",
                    "quality_quantile": selection["quality_quantile"],
                    "risk_type": selection["risk_type"],
                    "ready3d_predicted_quality": selection["predicted_quality"],
                }
            )
        elif row.get("split") == "validation":
            queue.append(
                {
                    **row,
                    "label_source": "hunyuan_frozen_q_pending",
                    "sampling_reason": "all_validation",
                    "quality_quantile": None,
                    "risk_type": None,
                    "ready3d_predicted_quality": None,
                }
            )
    queue.sort(key=lambda row: (str(row["split"]), str(row["sample_id"])))
    coverage = {
        "train_inputs": 720,
        "train_hunyuan": sum(row["split"] == "train" for row in queue),
        "validation_hunyuan": sum(row["split"] == "validation" for row in queue),
        "final_test_labels": 0,
        "queue_samples": len(queue),
        "quality_quantiles": dict(
            Counter(str(row["quality_quantile"]) for row in queue if row["split"] == "train")
        ),
        "categories": dict(Counter(str(row["category"]) for row in queue)),
        "risk_types": dict(
            Counter(str(row["risk_type"]) for row in queue if row["split"] == "train")
        ),
        "selection_seed": seed,
    }
    if coverage["train_hunyuan"] != 180 or coverage["validation_hunyuan"] != 120:
        raise RuntimeError(f"invalid queue coverage: {coverage}")
    return queue, coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue, coverage = build_queue(
        read_jsonl(args.candidates),
        read_jsonl(args.predictions),
        read_jsonl(args.features),
        seed=args.seed,
    )
    write_jsonl_atomic(args.output, queue)
    args.coverage.parent.mkdir(parents=True, exist_ok=True)
    args.coverage.write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(coverage, ensure_ascii=False))


if __name__ == "__main__":
    main()
