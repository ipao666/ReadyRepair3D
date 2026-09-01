#!/usr/bin/env python3
"""Freeze aligned 56/12/12 Ready3D V3 manifests and quality labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


SPLITS = ("train", "validation", "test")
EXPECTED_GROUP_COUNTS = {"train": 56, "validation": 12, "test": 12}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prepare_splits(
    manifest: list[dict],
    labels: list[dict],
    expected_group_counts: dict[str, int] = EXPECTED_GROUP_COUNTS,
) -> dict[str, dict[str, list[dict]]]:
    manifest_by_id = {str(row["sample_id"]): row for row in manifest}
    labels_by_id = {str(row["sample_id"]): row for row in labels}
    if len(manifest_by_id) != len(manifest) or len(labels_by_id) != len(labels):
        raise ValueError("duplicate sample IDs")
    if set(manifest_by_id) != set(labels_by_id):
        raise ValueError("manifest/label mismatch")
    result = {}
    seen_groups: dict[str, str] = {}
    for split in SPLITS:
        split_manifest = sorted(
            [row for row in manifest if row.get("split") == split],
            key=lambda row: (str(row["group_id"]), int(row["candidate_index"])),
        )
        groups = {str(row["group_id"]) for row in split_manifest}
        if len(groups) != expected_group_counts[split]:
            raise ValueError(
                f"{split}: expected {expected_group_counts[split]} groups, found {len(groups)}"
            )
        for group in groups:
            previous = seen_groups.setdefault(group, split)
            if previous != split:
                raise ValueError(f"group leakage: {group}")
        for group in groups:
            candidates = [row for row in split_manifest if row["group_id"] == group]
            if sorted(int(row["candidate_index"]) for row in candidates) != [0, 1, 2, 3]:
                raise ValueError(f"{group}: incomplete Best-of-4 group")
        split_labels = [labels_by_id[str(row["sample_id"])] for row in split_manifest]
        result[split] = {"manifest": split_manifest, "labels": split_labels}
    if sum(len(row["manifest"]) for row in result.values()) != len(manifest):
        raise ValueError("manifest contains unsupported split values")
    return result


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_splits(read_jsonl(args.manifest), read_jsonl(args.labels))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, payload in result.items():
        write_jsonl_atomic(args.output_dir / f"{split}_manifest.jsonl", payload["manifest"])
        write_jsonl_atomic(args.output_dir / f"{split}_labels.jsonl", payload["labels"])
    summary = {
        split: {
            "groups": len({row["group_id"] for row in payload["manifest"]}),
            "candidates": len(payload["manifest"]),
        }
        for split, payload in result.items()
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
