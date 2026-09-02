#!/usr/bin/env python3
"""Join frozen validation predictions and adherence scores for LoRA selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from r3dloop.sana_lora.data_protocol import read_jsonl, write_jsonl_atomic


def build_metrics(
    manifest: list[dict], predictions: list[dict], adherence: list[dict], checkpoint: str
) -> list[dict]:
    validation = [row for row in manifest if row.get("split") == "validation"]
    if len(validation) != 120:
        raise ValueError(f"expected 120 validation candidates, found {len(validation)}")
    predicted = {str(row["sample_id"]): row for row in predictions}
    scored = {str(row["sample_id"]): row for row in adherence}
    if len(predicted) != 120 or len(scored) != 120:
        raise ValueError("predictions and adherence must each contain 120 validation rows")
    rows = []
    for row in validation:
        sample_id = str(row["sample_id"])
        prediction, score = predicted.get(sample_id), scored.get(sample_id)
        if prediction is None or score is None:
            raise ValueError(f"missing proxy component for {sample_id}")
        if bool(score.get("adherence_missing")):
            raise ValueError(f"adherence is missing for {sample_id}")
        rows.append(
            {
                "checkpoint": checkpoint,
                "sample_id": sample_id,
                "group_id": row["group_id"],
                "split": "validation",
                "ready3d_quality": prediction["predicted_quality"],
                "prompt_adherence": score["prompt_adherence"],
                "technical_valid": bool(prediction["structural_input_pass"]),
                "final_test_used": False,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--adherence", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_metrics(
        read_jsonl(args.manifest), read_jsonl(args.predictions), read_jsonl(args.adherence), args.checkpoint
    )
    write_jsonl_atomic(args.output, rows)
    print({"checkpoint": args.checkpoint, "validation_rows": len(rows), "final_test_used": False})


if __name__ == "__main__":
    main()
