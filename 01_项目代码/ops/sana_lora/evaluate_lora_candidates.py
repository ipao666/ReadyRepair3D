#!/usr/bin/env python3
"""Rank LoRA checkpoints on frozen development validation metrics."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from r3dloop.sana_lora.data_protocol import (  # noqa: E402
    read_jsonl,
    validate_unit_interval,
    write_jsonl_atomic,
)


def rank_checkpoints(rows: list[dict], *, top_k: int = 2) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "validation":
            raise ValueError("checkpoint selection may only consume validation rows")
        checkpoint = str(row.get("checkpoint", ""))
        if not checkpoint:
            raise ValueError("metric row is missing checkpoint")
        grouped[checkpoint].append(row)
    summaries = []
    for checkpoint, records in grouped.items():
        ready3d = [validate_unit_interval(row["ready3d_quality"], "ready3d_quality") for row in records]
        adherence = [validate_unit_interval(row["prompt_adherence"], "prompt_adherence") for row in records]
        technical = [1.0 if row.get("technical_valid") is True else 0.0 for row in records]
        proxy = [0.70 * quality + 0.30 * prompt for quality, prompt in zip(ready3d, adherence)]
        summaries.append(
            {
                "checkpoint": checkpoint,
                "samples": len(records),
                "mean_ready3d_quality": statistics.fmean(ready3d),
                "mean_prompt_adherence": statistics.fmean(adherence),
                "technical_valid_rate": statistics.fmean(technical),
                "proxy_score": statistics.fmean(proxy),
                "test_used_for_selection": False,
            }
        )
    ordered = sorted(
        summaries,
        key=lambda row: (
            row["technical_valid_rate"],
            row["proxy_score"],
            row["mean_prompt_adherence"],
            row["checkpoint"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(ordered, 1):
        row["rank"] = rank
        row["selected_for_hunyuan_validation"] = rank <= top_k
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranked = rank_checkpoints(read_jsonl(args.metrics), top_k=args.top_k)
    write_jsonl_atomic(args.output, ranked)
    print(json.dumps({"checkpoints": len(ranked), "selected": min(args.top_k, len(ranked))}))


if __name__ == "__main__":
    main()
