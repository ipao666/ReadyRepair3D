#!/usr/bin/env python3
"""Build a ranked candidate manifest selected by Ready3D."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_rows(
    manifest: list[dict],
    predictions: list[dict],
    expected_per_group: int = 1,
) -> list[dict]:
    if expected_per_group < 1:
        raise ValueError("expected_per_group must be at least one")
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    if len(manifest_by_id) != len(manifest):
        raise ValueError("manifest contains duplicate sample IDs")
    groups = sorted({str(row["group_id"]) for row in manifest})
    output = []
    for group_id in groups:
        selected = [
            row for row in predictions
            if str(row.get("group_id")) == group_id and row.get("selected") is True
        ]
        if len(selected) != expected_per_group:
            expected_text = "one" if expected_per_group == 1 else str(expected_per_group)
            raise ValueError(
                f"{group_id} must have exactly {expected_text} selected candidate"
                f"{'s' if expected_per_group != 1 else ''}"
            )
        ranked = []
        for prediction in selected:
            rank = prediction.get("selection_rank")
            if rank is None and expected_per_group == 1:
                rank = 1
            ranked.append((rank, prediction))
        actual_ranks = sorted(rank for rank, _ in ranked if rank is not None)
        expected_ranks = list(range(1, expected_per_group + 1))
        if actual_ranks != expected_ranks or len(actual_ranks) != len(ranked):
            raise ValueError(
                f"{group_id} selection ranks must be exactly {expected_ranks}, "
                f"got {actual_ranks}"
            )
        for rank, prediction in sorted(ranked, key=lambda item: item[0]):
            sample_id = prediction["sample_id"]
            if sample_id not in manifest_by_id:
                raise ValueError(
                    f"selected sample is missing from manifest: {sample_id}"
                )
            source = manifest_by_id[sample_id]
            ready3d = {
                key: prediction.get(key)
                for key in (
                    "predicted_quality", "uncertainty_tree_std", "interval_low",
                    "interval_high", "structural_input_pass", "severe_input_flags",
                    "selection_rule", "selection_fallback", "selection_rank",
                )
            }
            ready3d["selection_rank"] = rank
            output.append({**source, "ready3d": ready3d})
    return output


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-per-group", type=int, default=1)
    args = parser.parse_args()
    rows = select_rows(
        read_jsonl(args.manifest),
        read_jsonl(args.predictions),
        expected_per_group=args.expected_per_group,
    )
    write_jsonl_atomic(args.output, rows)
    print(json.dumps({"selected": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
