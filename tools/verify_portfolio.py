#!/usr/bin/env python3
"""Check published records and example bytes without models or a GPU.

This is an audit of committed evidence, not a rerun of generation or a proof
that the automatic quality metric agrees with human preferences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = {"direct", "random_top1", "structure_top1", "ready_v2_top1",
              "ready_v3_top1", "oracle_all4"}


def close(actual: float, expected: float, label: str) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError(f"{label}: non-finite value")
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label}: {actual} != {expected}")


def validate_strategies(data: dict) -> dict:
    if set(data["strategies"]) != STRATEGIES:
        raise ValueError("expected all six strategies, including structural baseline")
    rows = data["group_results"]
    if {row["strategy"] for row in rows} != STRATEGIES:
        raise ValueError("unexpected or missing strategy records")
    group_sets = []
    means = {}
    asset_scores = {}
    for name, reported in data["strategies"].items():
        selected = [row for row in rows if row["strategy"] == name]
        groups = {row["group_id"] for row in selected}
        if len(selected) != len(groups) or len(groups) != data["groups"]:
            raise ValueError(f"{name}: duplicate or missing groups")
        if reported["groups"] != len(groups):
            raise ValueError(f"{name}: inconsistent group count")
        group_sets.append(groups)
        for row in selected:
            q, oracle = float(row["selected_quality"]), float(row["optimal_quality"])
            if not (0 <= q <= 1 and 0 <= oracle <= 1 and q <= oracle + 1e-9):
                raise ValueError(f"{name}: invalid quality range")
            close(float(row["regret"]), oracle - q, f"{name} regret")
            for id_key, value in (("selected_sample_id", q), ("optimal_sample_id", oracle)):
                key = (row["group_id"], row[id_key])
                if key in asset_scores:
                    close(value, asset_scores[key], "same asset quality")
                asset_scores[key] = value
        for metric, column in (("mean_quality", "selected_quality"),
                               ("mean_regret", "regret"),
                               ("top1_accuracy", "top1_correct"),
                               ("qualified_rate", "qualified"),
                               ("technical_valid_rate", "technically_valid")):
            average = sum(float(row[column]) for row in selected) / len(selected)
            close(average, float(reported[metric]), f"{name} {metric}")
        means[name] = sum(float(row["selected_quality"]) for row in selected) / len(selected)
    if any(groups != group_sets[0] for groups in group_sets):
        raise ValueError("strategies use different prompt groups")
    return means


def validate_final(data: dict) -> None:
    base, lora = data["base_summary"], data["lora_summary"]
    if base["groups"] != 64 or lora["groups"] != 64:
        raise ValueError("expected the frozen 64-group final comparison")
    for summary in (base, lora):
        for metric in ("mean_q", "pass_rate", "adherence_rate", "technical_valid_rate"):
            if not 0 <= float(summary[metric]) <= 1:
                raise ValueError(f"invalid final {metric}")
    bootstrap = data["paired_bootstrap"]
    close(lora["mean_q"] - base["mean_q"],
          bootstrap["mean_q_difference_lora_minus_base"], "final mean difference")
    low, high = bootstrap["ci95_low"], bootstrap["ci95_high"]
    if not (math.isfinite(low) and math.isfinite(high) and low <= high):
        raise ValueError("invalid final confidence interval")
    # Preserve the archived decision; raw samples for a fresh decision are absent.
    if data["decision"] != "keep_base_sana":
        raise ValueError("frozen final decision changed")


def verify_asset(root: Path, relative: str, expected: str) -> None:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("asset path escapes repository")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"asset checksum mismatch: {relative}")


def verify(root: Path) -> dict:
    stage1 = json.loads((root / "frozen_results/stage01_ready3d_v3_r2/strategy_metrics.json").read_text(encoding="utf-8"))
    means = validate_strategies(stage1)
    final = json.loads((root / "frozen_results/stage06_final_test/final_comparison.json").read_text(encoding="utf-8"))
    validate_final(final)
    code = root / "01_项目代码"
    showcase = json.loads((code / "examples/showcase_manifest.json").read_text(encoding="utf-8"))
    for asset in showcase["assets"]:
        for kind in ("glb", "preview"):
            verify_asset(code, asset[kind], asset[f"{kind}_sha256"])
    return {
        "status": "passed",
        "scope": "committed record consistency and showcase checksums; no model inference",
        "stage1_groups": stage1["groups"],
        "stage1_recomputed_mean_quality": means,
        "final_decision": final["decision"],
        "final_bootstrap_recomputed": False,
        "final_raw_group_records_in_repository": False,
        "showcase_assets_verified": len(showcase["assets"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = verify(args.root)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"Evidence check failed: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
