#!/usr/bin/env python3
"""Select and deliver the better of two V2-scored Hunyuan assets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


FROZEN_QUALITY_THRESHOLD = 0.806912747446761


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_best(
    labels: list[dict],
    manifest: list[dict],
    threshold: float = FROZEN_QUALITY_THRESHOLD,
) -> dict:
    manifest_by_id = {
        str(row["sample_id"]): row for row in manifest if row.get("sample_id")
    }
    labels_by_id = {str(row["sample_id"]): row for row in labels}
    if len(labels_by_id) != len(labels):
        raise ValueError("manifest and labels must contain unique sample IDs")
    if len(labels) not in (1, 2) or len(manifest) != len(labels):
        raise ValueError("one or two aligned manifest rows and labels are required")
    if len(manifest_by_id) == len(manifest) and set(labels_by_id) == set(manifest_by_id):
        aligned = [
            (source, labels_by_id[sample_id])
            for sample_id, source in manifest_by_id.items()
        ]
    else:
        def path_key(row: dict) -> str:
            value = row.get("source_path") or row.get("path")
            return Path(str(value)).name if value else ""

        manifest_by_source = {
            path_key(row): row for row in manifest
        }
        labels_by_source = {
            path_key(row): row for row in labels
        }
        if (
            "" in manifest_by_source
            or "" in labels_by_source
            or len(manifest_by_source) != len(manifest)
            or len(labels_by_source) != len(labels)
            or set(manifest_by_source) != set(labels_by_source)
        ):
            raise ValueError(
                "one or two aligned manifest rows and labels are required"
            )
        aligned = [
            (source, labels_by_source[source_path])
            for source_path, source in manifest_by_source.items()
        ]

    candidates = []
    for source, label in aligned:
        candidates.append(
            {
                **label,
                "sample_id": str(label["sample_id"]),
                "candidate_index": int(source["candidate_index"]),
            }
        )
    candidates.sort(key=lambda row: row["candidate_index"])
    valid = [row for row in candidates if bool(row["technically_valid"])]

    def quality_key(row: dict) -> tuple[float, float, float, int]:
        return (
            float(row["quality_score_v2"]),
            float(row.get("largest_component_area_ratio", 0.0)),
            -float(row.get("effective_component_count", float("inf"))),
            -int(row["candidate_index"]),
        )

    pool = valid or candidates
    winner = max(pool, key=quality_key)
    if len(candidates) == 1 and valid:
        selection_reason = "only_candidate_after_filtering"
    elif not valid:
        selection_reason = "diagnostic_best_no_valid_assets"
    elif len(valid) == 1:
        selection_reason = "only_technically_valid"
    elif len({float(row["quality_score_v2"]) for row in valid}) == 1:
        selection_reason = "frozen_tie_break"
    else:
        selection_reason = "highest_quality_score_v2"
    winner_valid = bool(winner["technically_valid"])
    winner_score = float(winner["quality_score_v2"])
    return {
        "schema_version": "r3dguard.top2-3d-selection.v1",
        "winner_sample_id": winner["sample_id"],
        "winner_candidate_index": winner["candidate_index"],
        "winner_glb_path": str(winner["glb_path"]),
        "winner_quality_score_v2": winner_score,
        "round_failed": not bool(valid),
        "selection_reason": selection_reason,
        "quality_threshold": float(threshold),
        "meets_quality_threshold": bool(
            winner_valid and winner_score >= float(threshold)
        ),
        "candidates": [
            {
                "sample_id": row["sample_id"],
                "candidate_index": row["candidate_index"],
                "quality_score_v2": float(row["quality_score_v2"]),
                "technically_valid": bool(row["technically_valid"]),
                "largest_component_area_ratio": float(
                    row.get("largest_component_area_ratio", 0.0)
                ),
                "effective_component_count": int(
                    row.get("effective_component_count", 0)
                ),
                "glb_path": str(row["glb_path"]),
            }
            for row in candidates
        ],
    }


def deliver(report: dict, output_dir: Path) -> Path:
    source = Path(report["winner_glb_path"])
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / "final.glb"
    temporary = output_dir / "final.glb.tmp"
    shutil.copy2(source, temporary)
    os.replace(temporary, final)
    (output_dir / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--threshold", type=float, default=FROZEN_QUALITY_THRESHOLD
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = select_best(
        read_jsonl(args.labels),
        read_jsonl(args.manifest),
        threshold=args.threshold,
    )
    final = deliver(report, args.output_dir)
    print(
        json.dumps(
            {
                "winner_sample_id": report["winner_sample_id"],
                "quality_score_v2": report["winner_quality_score_v2"],
                "technically_valid": not report["round_failed"],
                "meets_quality_threshold": report["meets_quality_threshold"],
                "final_glb": str(final),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
