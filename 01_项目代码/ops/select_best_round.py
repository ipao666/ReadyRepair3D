#!/usr/bin/env python3
"""Select and deliver the better result from one or two completed rounds."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


FROZEN_QUALITY_THRESHOLD = 0.806912747446761


def select_rounds(
    reports: list[dict],
    threshold: float = FROZEN_QUALITY_THRESHOLD,
) -> dict:
    if not 1 <= len(reports) <= 2:
        raise ValueError("exactly one or two round reports are required")
    candidates = [
        {
            **report,
            "round_index": index,
        }
        for index, report in enumerate(reports, 1)
    ]
    winner = max(
        candidates,
        key=lambda row: (
            bool(row["final_technically_valid"]),
            float(row["final_quality_score_v2"]),
            -int(row["round_index"]),
        ),
    )
    final_valid = bool(winner["final_technically_valid"])
    final_score = float(winner["final_quality_score_v2"])
    return {
        "schema_version": "r3dguard.best-round.v1",
        "round_count": len(candidates),
        "second_round_executed": len(candidates) == 2,
        "selected_round": int(winner["round_index"]),
        "selected_glb_source": str(winner["final_glb"]),
        "final_quality_score_v2": final_score,
        "final_technically_valid": final_valid,
        "quality_threshold": float(threshold),
        "meets_quality_threshold": bool(
            final_valid and final_score >= float(threshold)
        ),
        "low_confidence": not final_valid,
        "rounds": [
            {
                "round_index": row["round_index"],
                "quality_score_v2": float(row["final_quality_score_v2"]),
                "technically_valid": bool(row["final_technically_valid"]),
                "meets_quality_threshold": bool(
                    row.get("meets_quality_threshold", False)
                ),
                "repair3d_accepted": bool(row.get("repair3d_accepted", False)),
                "v2_repair_accepted": bool(
                    row.get("v2_repair_accepted", False)
                ),
                "final_glb": str(row["final_glb"]),
            }
            for row in candidates
        ],
    }


def deliver(report: dict, output_dir: Path) -> Path:
    source = Path(report["selected_glb_source"])
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "final.glb.tmp"
    final = output_dir / "final.glb"
    shutil.copy2(source, temporary)
    os.replace(temporary, final)
    report = {**report, "final_glb": str(final.resolve())}
    (output_dir / "final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--round-report", type=Path, nargs="+", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--threshold", type=float, default=FROZEN_QUALITY_THRESHOLD
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.round_report
    ]
    report = select_rounds(reports, threshold=args.threshold)
    final = deliver(report, args.output_dir)
    print(
        json.dumps(
            {
                "selected_round": report["selected_round"],
                "quality_score_v2": report["final_quality_score_v2"],
                "technically_valid": report["final_technically_valid"],
                "meets_quality_threshold": report["meets_quality_threshold"],
                "low_confidence": report["low_confidence"],
                "final": str(final),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
