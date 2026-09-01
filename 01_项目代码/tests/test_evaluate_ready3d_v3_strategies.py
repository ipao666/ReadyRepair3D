import csv
import io

import pytest

from ops.evaluate_ready3d_v3_strategies import STRATEGIES, evaluate_strategies, metrics_csv


def _rows(split="test"):
    rows = []
    qualities = [[0.1, 0.9, 0.3, 0.2], [0.8, 0.2, 0.4, 0.1]]
    v2 = [[0.1, 0.8, 0.2, 0.3], [0.7, 0.2, 0.4, 0.1]]
    v3 = [[0.2, 0.7, 0.95, 0.1], [0.9, 0.1, 0.2, 0.3]]
    for group in range(2):
        for candidate in range(4):
            rows.append(
                {
                    "group_id": f"g{group}",
                    "sample_id": f"g{group}_c{candidate}",
                    "candidate_index": candidate,
                    "split": split,
                    "quality_score": qualities[group][candidate],
                    "qualified": qualities[group][candidate] >= 0.75,
                    "technically_valid": True,
                    "structural_input_pass": not (group == 0 and candidate == 1),
                    "structure_score": 1.0 - 0.1 * candidate,
                    "ready_v2_score": v2[group][candidate],
                    "ready_v3_score": v3[group][candidate],
                }
            )
    return rows


def test_six_strategies_report_quality_cost_and_oracle_bound():
    report = evaluate_strategies(_rows(), bootstrap_draws=200, seed=17)

    assert list(report["strategies"]) == list(STRATEGIES)
    assert report["strategies"]["direct"]["mean_quality"] == pytest.approx(0.45)
    assert report["strategies"]["ready_v3_top1"]["mean_quality"] == pytest.approx(0.55)
    assert report["strategies"]["oracle_all4"]["mean_quality"] == pytest.approx(0.85)
    assert report["strategies"]["ready_v3_top1"]["mean_regret"] == pytest.approx(0.3)
    assert report["strategies"]["ready_v3_top1"]["quality_capture"] == pytest.approx(
        (0.3 / 0.9 + 0.8 / 0.8) / 2
    )
    assert report["strategies"]["ready_v3_top1"]["three_d_calls_per_group"] == 1
    assert report["strategies"]["oracle_all4"]["three_d_calls_per_group"] == 4
    assert len(report["comparisons_vs_direct"]["ready_v3_top1"]["mean_quality_delta_95_ci"]) == 2


def test_random_top1_is_reproducible_for_fixed_seed():
    first = evaluate_strategies(_rows(), bootstrap_draws=50, seed=99)
    second = evaluate_strategies(_rows(), bootstrap_draws=50, seed=99)

    assert first["group_results"] == second["group_results"]
    assert first["comparisons_vs_direct"] == second["comparisons_vs_direct"]


def test_evaluator_rejects_non_test_rows_to_prevent_selection_leakage():
    with pytest.raises(ValueError, match="test split only"):
        evaluate_strategies(_rows(split="validation"), bootstrap_draws=10)


def test_metrics_csv_contains_one_row_per_strategy():
    report = evaluate_strategies(_rows(), bootstrap_draws=20, seed=3)
    parsed = list(csv.DictReader(io.StringIO(metrics_csv(report))))

    assert len(parsed) == 6
    assert {row["strategy"] for row in parsed} == set(STRATEGIES)
