#!/usr/bin/env python3
"""Split remaining SANA groups into Ready3D train and held-out test manifests."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_group_splits(
    rows: list[dict], excluded: set[str], validation: set[str], seed: int
) -> dict:
    metadata = {}
    counts = {}
    for row in rows:
        group = str(row["group_id"])
        metadata.setdefault(group, row)
        counts[group] = counts.get(group, 0) + 1
    if any(count != 4 for count in counts.values()):
        raise ValueError("Every group must contain exactly four candidates")
    remaining = sorted(set(metadata) - excluded - validation)
    if len(remaining) != 30:
        raise ValueError(f"Expected 30 remaining groups, found {len(remaining)}")
    rng = random.Random(seed)
    test = None
    for _ in range(100_000):
        candidate = sorted(rng.sample(remaining, 6))
        categories = {metadata[group]["category"] for group in candidate}
        difficulties = {metadata[group]["difficulty"] for group in candidate}
        if len(categories) >= 6 and difficulties == {"easy", "medium", "hard"}:
            test = candidate
            break
    if test is None:
        raise ValueError("Unable to construct a balanced held-out test split")
    train = sorted(set(remaining) - set(test))
    return {
        "train": train,
        "test": test,
        "test_categories": sorted({metadata[group]["category"] for group in test}),
        "test_difficulties": sorted({metadata[group]["difficulty"] for group in test}),
    }


def select_rows(rows: list[dict], groups: list[str]) -> list[dict]:
    selected = [row for row in rows if row["group_id"] in set(groups)]
    selected.sort(key=lambda row: (row["group_id"], int(row["candidate_index"])))
    if len(selected) != len(groups) * 4:
        raise ValueError("Split manifest is missing candidates")
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--exclude", default="sana_000,sana_005,sana_022,sana_047")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    validation_summary = json.loads(args.validation_summary.read_text(encoding="utf-8"))
    excluded = {item.strip() for item in args.exclude.split(",") if item.strip()}
    validation = set(validation_summary["groups"])
    splits = build_group_splits(rows, excluded, validation, args.seed)
    write_jsonl(args.output_dir / "train96.jsonl", select_rows(rows, splits["train"]))
    write_jsonl(args.output_dir / "test24.jsonl", select_rows(rows, splits["test"]))
    summary = {
        "seed": args.seed,
        "excluded_development_groups": sorted(excluded),
        "validation_groups": sorted(validation),
        **splits,
        "train_samples": 96,
        "test_samples": 24,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
