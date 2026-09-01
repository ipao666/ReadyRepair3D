#!/usr/bin/env python3
"""Validate all Stage 2 artifacts before immutable freeze."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from r3dloop.sana_lora.data_protocol import read_jsonl  # noqa: E402


def summarize(manifest: list[dict], adherence: list[dict]) -> dict:
    if len(manifest) != 960:
        raise ValueError(f"expected 960 candidate rows, found {len(manifest)}")
    sample_ids = [str(row["sample_id"]) for row in manifest]
    if len(set(sample_ids)) != 960:
        raise ValueError("candidate sample IDs are not unique")
    if any(row.get("split") == "final_test" for row in manifest):
        raise ValueError("final_test entered Stage 2")
    invalid_images = []
    resolutions = Counter()
    for row in manifest:
        path = Path(str(row.get("source_path") or row.get("image_path") or row.get("path")))
        try:
            with Image.open(path) as image:
                image.load()
                resolutions[f"{image.width}x{image.height}:{image.mode}"] += 1
                if image.size != (1024, 1024) or image.mode != "RGB":
                    invalid_images.append(str(path))
        except Exception:
            invalid_images.append(str(path))
    adherence_by_id = {str(row["sample_id"]): row for row in adherence}
    if len(adherence_by_id) != len(adherence):
        raise ValueError("duplicate adherence sample IDs")
    missing_rows = sorted(set(sample_ids) - set(adherence_by_id))
    extra_rows = sorted(set(adherence_by_id) - set(sample_ids))
    if missing_rows or extra_rows:
        raise ValueError(
            f"adherence rows do not align: missing={missing_rows[:5]} extra={extra_rows[:5]}"
        )
    failed_scores = sum(bool(adherence_by_id[sample_id]["adherence_missing"]) for sample_id in sample_ids)
    split_counts = Counter(str(row["split"]) for row in manifest)
    summary = {
        "schema_version": "r3dguard.stage02-summary.v1",
        "status": "ready",
        "candidates_expected": 960,
        "candidates_success": 960 - len(invalid_images),
        "candidates_failed": len(invalid_images),
        "candidate_split_counts": dict(split_counts),
        "unique_seeds": len({int(row["seed"]) for row in manifest}),
        "seed_coverage_rate": len({int(row["seed"]) for row in manifest}) / 960,
        "image_resolutions": dict(resolutions),
        "adherence_rows": len(adherence),
        "adherence_scored": 960 - failed_scores,
        "adherence_missing": failed_scores,
        "adherence_coverage_rate": (960 - failed_scores) / 960,
        "final_test_used": False,
    }
    if invalid_images:
        raise ValueError(f"invalid Stage 2 images: {invalid_images[:5]}")
    if dict(split_counts) != {"train": 720, "validation": 120, "dev_test": 120}:
        raise ValueError(f"invalid candidate split counts: {dict(split_counts)}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adherence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(read_jsonl(args.manifest), read_jsonl(args.adherence))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
