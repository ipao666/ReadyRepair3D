from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.aggregate_final_results import build_markdown


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation_summary" / "independent32"


def test_submission_keeps_exactly_three_ppt_strategies() -> None:
    with (RESULTS / "strategy_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["strategy"]: row for row in csv.DictReader(handle)}

    assert set(rows) == {"direct", "ready_top2", "all"}
    assert float(rows["all"]["mean_quality_score_v2"]) == 0.7495
    assert float(rows["all"]["qualified_rate"]) == 0.472
    assert float(rows["all"]["mean_3d_calls"]) == 4.0
    assert float(rows["all"]["mean_gpu_generation_seconds"]) == 457.2


def test_same_asset_has_same_frozen_quality_score_across_strategies() -> None:
    rows = [
        json.loads(line)
        for line in (RESULTS / "strategy_group_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == 64
    assert {row["strategy"] for row in rows} == {"direct", "ready_top2"}
    scores: dict[tuple[str, str], set[float]] = {}
    for row in rows:
        key = (row["group_id"], row["selected_sample_id"])
        scores.setdefault(key, set()).add(float(row["quality_score_v2"]))
    assert all(len(values) == 1 for values in scores.values())


def test_report_uses_only_three_ppt_strategies() -> None:
    metrics = json.loads(
        (ROOT / "evaluation_summary" / "final_acceptance" / "final_metrics.json")
        .read_text(encoding="utf-8")
    )
    report = build_markdown(metrics)

    assert "All" in report
    assert "0.7495" in report
    assert "47.2%" in report
    assert "Ready Top-1" not in report
    assert "覆盖All" not in report


def test_all_has_no_unsubmitted_group_level_diagnostics() -> None:
    diagnostics = json.loads(
        (RESULTS / "selection_diagnostics.json").read_text(encoding="utf-8")
    )

    forbidden = {
        "ready_top2_all_coverage_rate",
        "all_mean_regret",
        "all_quality_capture",
    }
    assert forbidden.isdisjoint(diagnostics)
