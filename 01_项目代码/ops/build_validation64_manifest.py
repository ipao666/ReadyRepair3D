#!/usr/bin/env python3
"""Build a deterministic, group-isolated 64-image validation manifest."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def select_validation_groups(
    rows: list[dict], excluded_groups: set[str], seed: int, group_count: int = 16
) -> list[str]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["group_id"]), []).append(row)
    for group_id, candidates in grouped.items():
        indexes = {int(row["candidate_index"]) for row in candidates}
        if indexes != {0, 1, 2, 3} or len(candidates) != 4:
            raise ValueError(f"{group_id} must contain candidates 0,1,2,3 exactly once")
    available = sorted(set(grouped) - set(excluded_groups))
    if len(available) < group_count:
        raise ValueError("Not enough non-excluded groups")
    metadata = {group_id: grouped[group_id][0] for group_id in available}
    all_categories = {row["category"] for row in metadata.values()}
    rng = random.Random(seed)
    for _ in range(100_000):
        selected = sorted(rng.sample(available, group_count))
        categories = Counter(metadata[group]["category"] for group in selected)
        difficulties = Counter(metadata[group]["difficulty"] for group in selected)
        if set(categories) != all_categories:
            continue
        if max(categories.values()) > 2:
            continue
        if any(difficulties[name] < 4 for name in ("easy", "medium", "hard")):
            continue
        return selected
    raise ValueError("Unable to find a balanced validation selection")


def build_manifest(rows: list[dict], selected_groups: list[str]) -> list[dict]:
    selected = set(selected_groups)
    output = [row for row in rows if row["group_id"] in selected]
    output.sort(key=lambda row: (row["group_id"], int(row["candidate_index"])))
    if len(output) != len(selected_groups) * 4:
        raise ValueError("Selected manifest does not contain four candidates per group")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--exclude", default="sana_000,sana_005,sana_022,sana_047")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    excluded = {value.strip() for value in args.exclude.split(",") if value.strip()}
    groups = select_validation_groups(rows, excluded, args.seed)
    manifest = build_manifest(rows, groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    group_meta = {row["group_id"]: row for row in manifest}
    summary = {
        "seed": args.seed,
        "samples": len(manifest),
        "groups": groups,
        "excluded_groups": sorted(excluded),
        "category_counts": dict(sorted(Counter(row["category"] for row in group_meta.values()).items())),
        "difficulty_counts": dict(sorted(Counter(row["difficulty"] for row in group_meta.values()).items())),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
