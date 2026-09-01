#!/usr/bin/env python3
"""Apply post-repair V2 validation and deliver one safe round result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


FROZEN_QUALITY_THRESHOLD = 0.806912747446761
MIN_REPAIR_GAIN = 0.005


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def finalize_round(
    original_labels: list[dict],
    selection_report: dict,
    repair_report: dict,
    repaired_labels: list[dict] | None = None,
    threshold: float = FROZEN_QUALITY_THRESHOLD,
) -> dict:
    winner_id = str(selection_report["winner_sample_id"])
    originals = [
        row for row in original_labels if str(row.get("sample_id")) == winner_id
    ]
    if len(originals) != 1:
        raise ValueError(f"winner must have exactly one original label: {winner_id}")
    original = originals[0]
    delivered = original
    delivered_variant = "original"
    v2_repair_accepted = False
    candidate_ready = bool(
        repair_report.get("candidate_ready_for_rescore")
        or repair_report.get("accepted")
    )
    if not candidate_ready:
        rollback_reason = str(
            repair_report.get("rollback_reason") or "repair3d_rejected"
        )
    else:
        if repaired_labels is None or len(repaired_labels) != 1:
            raise ValueError("accepted Repair3D result requires one repaired label")
        repaired = repaired_labels[0]
        if repair_report.get("schema_version") == "r3dguard.repair3d-auto.v2":
            expected_path = Path(str(repair_report["candidate_path"])).resolve()
            actual_path = Path(str(repaired.get("glb_path", ""))).resolve()
            candidate_sha = str(
                repair_report.get("sha256", {}).get("candidate", "")
            )
            expected_id = f"{winner_id}_postrepair_{candidate_sha[:12]}"
            if (
                not candidate_sha
                or not expected_path.is_file()
                or file_sha256(expected_path) != candidate_sha
                or actual_path != expected_path
                or str(repaired.get("sample_id")) != expected_id
            ):
                raise ValueError("repaired quality label is not bound to candidate")
        repaired_valid = bool(repaired["technically_valid"])
        original_valid = bool(original["technically_valid"])
        repaired_score = float(repaired["quality_score_v2"])
        original_score = float(original["quality_score_v2"])
        if not repaired_valid:
            rollback_reason = "post_repair_technically_invalid"
        elif repaired_score >= original_score + MIN_REPAIR_GAIN:
            delivered = repaired
            delivered_variant = "repaired"
            v2_repair_accepted = True
            rollback_reason = None
        else:
            rollback_reason = "post_repair_gain_below_0.005"
    final_valid = bool(delivered["technically_valid"])
    final_score = float(delivered["quality_score_v2"])
    return {
        "schema_version": "r3dguard.repaired-round.v1",
        "winner_sample_id": winner_id,
        "repair3d_action": repair_report.get("recommended_action"),
        "repair3d_status": (
            "repair_accepted"
            if v2_repair_accepted
            else str(repair_report.get("status") or "repair_rollback")
        ),
        "repair3d_accepted": v2_repair_accepted,
        "repair3d_rollback_reason": repair_report.get("rollback_reason"),
        "v2_repair_accepted": v2_repair_accepted,
        "v2_rollback_reason": rollback_reason,
        "delivered_variant": delivered_variant,
        "delivered_glb_source": str(delivered["glb_path"]),
        "original_quality_score_v2": float(original["quality_score_v2"]),
        "post_repair_quality_score_v2": (
            None
            if repaired_labels is None
            else float(repaired_labels[0]["quality_score_v2"])
        ),
        "final_quality_score_v2": final_score,
        "final_technically_valid": final_valid,
        "quality_threshold": float(threshold),
        "meets_quality_threshold": bool(
            final_valid and final_score >= float(threshold)
        ),
        "low_confidence": not final_valid,
    }


def finalize_repair_artifacts(round_report: dict, repair_report_path: Path) -> None:
    repair_report = read_json(repair_report_path)
    repair_dir = repair_report_path.parent
    original = repair_dir / "original.glb"
    candidate = repair_dir / "candidate.glb"
    final = repair_dir / "final.glb"
    repaired = repair_dir / "repaired.glb"
    accepted = bool(round_report["v2_repair_accepted"])
    source = candidate if accepted else original
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, final)
    if accepted:
        shutil.copy2(candidate, repaired)
    else:
        repaired.unlink(missing_ok=True)
    original_sha = file_sha256(original)
    final_sha = file_sha256(final)
    if not accepted and original_sha != final_sha:
        raise RuntimeError("rollback SHA256 mismatch")
    before = float(round_report["original_quality_score_v2"])
    after = round_report["post_repair_quality_score_v2"]
    original_status = str(repair_report.get("status") or "repair_rollback")
    final_status = (
        "repair_accepted"
        if accepted
        else (
            original_status
            if original_status in {"accept_original", "regenerate_required"}
            else "repair_rollback"
        )
    )
    repair_report.update(
        {
            "status": final_status,
            "accepted": accepted,
            "rollback_reason": (
                None if accepted else round_report["v2_rollback_reason"]
            ),
            "frozen_quality_before": before,
            "frozen_quality_after": after,
            "quality_gain": (
                None if after is None else float(after) - before
            ),
            "final_path": str(final.resolve()),
            "repaired_path": str(repaired.resolve()) if accepted else None,
            "delivered_source": str(source.resolve()),
            "sha256": {
                "original": original_sha,
                "candidate": file_sha256(candidate) if candidate.is_file() else None,
                "final": final_sha,
            },
        }
    )
    repair_report_path.write_text(
        json.dumps(repair_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pipeline_summary_path = repair_dir / "pipeline_summary.json"
    if pipeline_summary_path.is_file():
        summary = read_json(pipeline_summary_path)
        summary.update(
            {
                "repair_status": final_status,
                "accepted": accepted,
                "rollback_reason": repair_report["rollback_reason"],
                "quality_gain": repair_report["quality_gain"],
                "final_path": str(final.resolve()),
            }
        )
        pipeline_summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def deliver(report: dict, output_dir: Path) -> Path:
    source = Path(report["delivered_glb_source"])
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "final.glb.tmp"
    final = output_dir / "final.glb"
    shutil.copy2(source, temporary)
    os.replace(temporary, final)
    report = {**report, "final_glb": str(final.resolve())}
    (output_dir / "round_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-labels", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--repair-report", type=Path, required=True)
    parser.add_argument("--repaired-labels", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--threshold", type=float, default=FROZEN_QUALITY_THRESHOLD
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repaired_labels = (
        None if args.repaired_labels is None else read_jsonl(args.repaired_labels)
    )
    report = finalize_round(
        read_jsonl(args.original_labels),
        read_json(args.selection_report),
        read_json(args.repair_report),
        repaired_labels,
        threshold=args.threshold,
    )
    finalize_repair_artifacts(report, args.repair_report)
    final = deliver(report, args.output_dir)
    print(
        json.dumps(
            {
                "final": str(final),
                "quality_score_v2": report["final_quality_score_v2"],
                "technically_valid": report["final_technically_valid"],
                "meets_quality_threshold": report["meets_quality_threshold"],
                "delivered_variant": report["delivered_variant"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
