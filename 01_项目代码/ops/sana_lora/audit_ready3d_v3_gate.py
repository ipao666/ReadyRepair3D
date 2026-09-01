#!/usr/bin/env python3
"""Audit the frozen Ready3D V3 gate before allowing LoRA pseudo-labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


def audit_gate(model_metrics: dict, strategy_metrics: dict, checkpoint: dict) -> dict:
    model_selection = model_metrics.get("model_selection", {})
    validation = model_metrics.get("validation", {})
    strategies = strategy_metrics.get("strategies", {})
    v3 = strategies.get("ready_v3_top1", {})
    random = strategies.get("random_top1", {})
    checks = {
        "checkpoint_schema_valid": checkpoint.get("schema_version")
        == "r3dguard.ready3d-v3-top1-checkpoint.v1",
        "validation_only_model_selection": model_selection.get("split") == "validation",
        "test_not_used_for_model_selection": model_selection.get("test_used_for_selection")
        is False,
        "test_not_used_for_threshold_selection": strategy_metrics.get(
            "test_used_for_model_or_threshold_selection"
        )
        is False,
        "v3_test_mean_quality_above_random": float(v3.get("mean_quality", -1.0))
        > float(random.get("mean_quality", -1.0)),
        "validation_top1_above_random_chance": float(
            validation.get("top1_accuracy", -1.0)
        )
        > 0.25,
        "validation_quality_capture_positive": float(
            validation.get("quality_capture", -1.0)
        )
        > 0.0,
    }
    return {
        "schema_version": "r3dguard.ready3d-v3-pseudolabel-gate.v1",
        "checks": checks,
        "allow_ready3d_v3_pseudo_labels": all(checks.values()),
        "validation": validation,
        "test_ready_v3_top1": v3,
        "test_random_top1": random,
        "test_used_for_selection": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-metrics", type=Path, required=True)
    parser.add_argument("--strategy-metrics", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_gate(
        json.loads(args.model_metrics.read_text(encoding="utf-8")),
        json.loads(args.strategy_metrics.read_text(encoding="utf-8")),
        joblib.load(args.checkpoint),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
