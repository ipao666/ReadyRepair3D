import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = []
    for row in manifest:
        sample_dir = args.runs / row["prompt_id"]
        summary_path = sample_dir / "pipeline_summary.json"
        item = {
            "prompt_id": row["prompt_id"],
            "category": row["category"],
            "base_seed": row["base_seed"],
            "status": "pending",
        }
        if (sample_dir / "BATCH_SAMPLE_FAILED").exists():
            item["status"] = "failed"
        if summary_path.exists() and (sample_dir / "PIPELINE_COMPLETE").exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            item.update(
                {
                    "status": "complete",
                    "round_count": summary["round_count"],
                    "second_round_executed": summary["second_round_executed"],
                    "selected_round": summary["selected_round"],
                    "final_quality_score_v2": summary["final_quality_score_v2"],
                    "final_technically_valid": summary["final_technically_valid"],
                    "meets_quality_threshold": summary["meets_quality_threshold"],
                    "low_confidence": summary["low_confidence"],
                    "elapsed_seconds": summary["elapsed_seconds"],
                    "repair3d_accepted_any": any(
                        round_row["v2_repair_accepted"] for round_row in summary["rounds"]
                    ),
                }
            )
        samples.append(item)

    complete = [row for row in samples if row["status"] == "complete"]
    valid = [row for row in complete if row["final_technically_valid"]]
    passed = [row for row in complete if row["meets_quality_threshold"]]
    report = {
        "schema_version": "r3dguard.e2e16-summary.v1",
        "planned": len(samples),
        "completed": len(complete),
        "failed": sum(row["status"] == "failed" for row in samples),
        "pending": sum(row["status"] == "pending" for row in samples),
        "technically_valid": len(valid),
        "meets_quality_threshold": len(passed),
        "second_round_count": sum(
            bool(row.get("second_round_executed")) for row in complete
        ),
        "repair3d_accepted_count": sum(
            bool(row.get("repair3d_accepted_any")) for row in complete
        ),
        "mean_final_quality_score_v2": (
            sum(row["final_quality_score_v2"] for row in complete) / len(complete)
            if complete
            else None
        ),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}))


if __name__ == "__main__":
    main()
