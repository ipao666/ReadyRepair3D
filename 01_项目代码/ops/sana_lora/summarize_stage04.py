#!/usr/bin/env python3
"""Validate the six required LoRA checkpoints before proxy selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_STEPS = (250, 500, 750, 1000, 1250, 1500)


def summarize(output_dir: Path) -> dict:
    checkpoints = []
    for step in REQUIRED_STEPS:
        directory = output_dir / f"checkpoint-{step}"
        weights = directory / "pytorch_lora_weights.safetensors"
        if not weights.is_file() or weights.stat().st_size == 0:
            raise ValueError(f"missing LoRA checkpoint weights for step {step}")
        checkpoints.append(
            {"step": step, "path": str(directory), "weights_bytes": weights.stat().st_size}
        )
    training_summary = json.loads(
        (output_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    if int(training_summary.get("steps", 0)) != 1500:
        raise ValueError("training did not finish exactly 1500 steps")
    return {
        "schema_version": "r3dguard.stage04-training-summary.v1",
        "status": "ready_for_proxy_selection",
        "checkpoints": checkpoints,
        "training": training_summary,
        "final_test_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(args.training_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
