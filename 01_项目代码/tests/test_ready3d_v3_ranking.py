import numpy as np
import pytest

from r3dloop.ready3d_v3.ranking import (
    add_group_context,
    combine_group_scores,
    groupwise_top1_metrics,
    pairwise_group_scores,
    select_ensemble_weights,
    select_top1,
    validate_complete_groups,
)


def _rows(groups=2):
    return [
        {
            "group_id": f"g{group}",
            "sample_id": f"g{group}_c{candidate}",
            "candidate_index": candidate,
        }
        for group in range(groups)
        for candidate in range(4)
    ]


def test_validate_complete_groups_rejects_missing_candidate():
    with pytest.raises(ValueError, match="expected 4"):
        validate_complete_groups(_rows()[:-1])


def test_add_group_context_centres_features_inside_each_group():
    rows = _rows()
    x = np.arange(16, dtype=np.float64).reshape(8, 2)

    augmented = add_group_context(x, [row["group_id"] for row in rows])

    assert augmented.shape == (8, 6)
    for group in ("g0", "g1"):
        mask = np.asarray([row["group_id"] == group for row in rows])
        np.testing.assert_allclose(augmented[mask, 2:4].mean(axis=0), 0.0)
        np.testing.assert_allclose(augmented[mask, 4:6].mean(axis=0), 0.0, atol=1e-7)


def test_structure_gate_precedes_model_score_and_ties_are_deterministic():
    rows = _rows(groups=1)
    combined = combine_group_scores(
        rows,
        pairwise=np.asarray([1.0, 0.7, 0.7, 0.1]),
        utility=np.asarray([1.0, 0.7, 0.7, 0.1]),
        structural_pass=np.asarray([False, True, True, True]),
        weights={"pairwise": 0.5, "utility": 0.5},
    )

    selected = select_top1(rows, combined, np.asarray([False, True, True, True]))

    assert selected[0]["selected_sample_id"] == "g0_c1"
    assert selected[0]["selection_fallback"] is None


def test_all_structure_failures_still_choose_exactly_one():
    rows = _rows(groups=1)
    passes = np.zeros(4, dtype=bool)
    selected = select_top1(rows, np.asarray([0.1, 0.8, 0.3, 0.2]), passes)

    assert selected == [
        {
            "group_id": "g0",
            "selected_sample_id": "g0_c1",
            "selected_candidate_index": 1,
            "selection_score": 0.8,
            "structural_input_pass": False,
            "selection_fallback": "all_candidates_failed_structure_gate",
        }
    ]


def test_validation_weight_search_prefers_predictive_utility_signal():
    rows = _rows(groups=2)
    quality = np.asarray([0.1, 0.2, 0.9, 0.3, 0.8, 0.2, 0.1, 0.4])
    utility = quality.copy()
    pairwise = 1.0 - quality
    structural = np.ones(8, dtype=bool)

    weights, metrics = select_ensemble_weights(
        rows, quality, pairwise, utility, structural, step=0.25
    )

    assert weights["utility"] > weights["pairwise"]
    assert metrics["mean_regret"] == pytest.approx(0.0)
    assert metrics["top1_accuracy"] == pytest.approx(1.0)


def test_groupwise_top1_metrics_reports_capture_and_regret():
    rows = _rows(groups=1)
    quality = np.asarray([0.2, 0.8, 0.6, 0.4])
    selected = select_top1(rows, np.asarray([0.1, 0.2, 0.9, 0.0]), np.ones(4, dtype=bool))

    metrics, records = groupwise_top1_metrics(rows, quality, selected)

    assert metrics["groups"] == 1
    assert metrics["mean_selected_quality"] == pytest.approx(0.6)
    assert metrics["mean_regret"] == pytest.approx(0.2)
    assert metrics["quality_capture"] == pytest.approx(0.75)
    assert records[0]["optimal_sample_id"] == "g0_c1"


def test_pairwise_scores_are_computed_separately_per_group():
    class DifferenceModel:
        def predict_proba(self, x):
            probability = 1.0 / (1.0 + np.exp(-x[:, 0]))
            return np.column_stack([1.0 - probability, probability])

    rows = _rows(groups=2)
    x = np.asarray([[0.0], [1.0], [2.0], [3.0], [30.0], [10.0], [20.0], [0.0]])

    scores = pairwise_group_scores(DifferenceModel(), x, rows)

    assert int(np.argmax(scores[:4])) == 3
    assert int(np.argmax(scores[4:])) == 0
