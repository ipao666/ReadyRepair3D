import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "ops" / "evaluate_top2_dual3d.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_top2_dual3d", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prediction(group, candidate, rank):
    return {
        "sample_id": f"{group}_c{candidate}",
        "group_id": group,
        "candidate_index": candidate,
        "selected": rank is not None,
        "selection_rank": rank,
    }


def quality(group, candidate, score, valid=True):
    return {
        "sample_id": f"{group}_c{candidate}",
        "quality_score_v2": score,
        "technically_valid": valid,
    }


def test_evaluate_compares_ready_top1_with_best_of_ready_top2():
    module = load_module()
    predictions = [
        prediction("g1", 0, 1),
        prediction("g1", 1, 2),
        prediction("g1", 2, None),
        prediction("g1", 3, None),
        prediction("g2", 0, 1),
        prediction("g2", 1, 2),
        prediction("g2", 2, None),
        prediction("g2", 3, None),
    ]
    labels = [
        quality("g1", 0, 0.60),
        quality("g1", 1, 0.80),
        quality("g1", 2, 0.99),
        quality("g1", 3, 0.10),
        quality("g2", 0, 0.90),
        quality("g2", 1, 0.70, valid=False),
        quality("g2", 2, 0.95),
        quality("g2", 3, 0.20),
    ]

    report = module.evaluate(
        predictions, labels, bootstrap_draws=100, seed=20260717
    )

    assert report["groups"] == 2
    assert report["mean_top1_quality"] == pytest.approx(0.75)
    assert report["mean_top2_dual_quality"] == pytest.approx(0.85)
    assert report["mean_gain"] == pytest.approx(0.10)
    assert report["positive_groups"] == 1
    assert report["zero_groups"] == 1
    assert report["negative_groups"] == 0
    assert report["top1_technical_valid_rate"] == pytest.approx(1.0)
    assert report["top2_dual_technical_valid_rate"] == pytest.approx(1.0)
    assert report["stage_gate_passed"] is True


def test_evaluate_rejects_missing_ready_rank_two():
    module = load_module()
    predictions = [
        prediction("g1", 0, 1),
        prediction("g1", 1, None),
        prediction("g1", 2, None),
        prediction("g1", 3, None),
    ]
    labels = [quality("g1", index, 0.5) for index in range(4)]

    with pytest.raises(ValueError, match="ranks"):
        module.evaluate(predictions, labels, bootstrap_draws=10)
