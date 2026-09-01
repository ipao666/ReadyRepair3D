import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "ops" / "train_ready3d.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_ready3d", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_group_leakage_is_rejected():
    module = load_module()
    rows = [
        {"group_id": "g", "split": "train"}, {"group_id": "g", "split": "test"},
        {"group_id": "v", "split": "validation"},
    ]
    try:
        module.validate_splits(rows)
    except ValueError as exc:
        assert "leakage" in str(exc).lower()
    else:
        raise AssertionError("Group leakage must be rejected")


def test_pairwise_training_data_has_balanced_orientations():
    module = load_module()
    x = np.arange(16, dtype=np.float32).reshape(4, 4)
    quality = np.array([0.1, 0.2, 0.3, 0.4])
    pair_x, pair_y, pair_weight = module.build_pairwise_data(
        x, quality, ["g"] * 4, np.ones(4, dtype=bool)
    )
    assert len(pair_x) == 12
    assert pair_y.sum() == 6
    assert np.allclose(pair_x[0], -pair_x[1])
    assert np.allclose(pair_weight[0::2], pair_weight[1::2])
    assert np.isclose(pair_weight.mean(), 1.0)
    assert pair_weight.min() >= 0.25
    assert pair_weight.max() <= 4.0


def test_pairwise_training_weights_prioritize_large_quality_gaps():
    module = load_module()
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    quality = np.array([0.10, 0.11, 0.50, 0.90])
    _, _, pair_weight = module.build_pairwise_data(
        x, quality, ["g"] * 4, np.ones(4, dtype=bool)
    )
    assert pair_weight[0] == pair_weight[1]
    assert pair_weight[-1] == pair_weight[-2]
    assert pair_weight[-1] > pair_weight[0]


def test_conformal_quantile_uses_finite_sample_correction():
    module = load_module()
    residuals = np.arange(1, 11, dtype=np.float32)
    assert module.conformal_quantile(residuals, 0.8) == 9.0


def test_quality_model_selection_prioritizes_validation_spearman_before_mae():
    module = load_module()
    ablations = {
        "engineered": {"validation": {"spearman": 0.5, "mae": 0.2}, "validation_ranking": {"mean_regret": 0.03, "quality_capture": 0.98}},
        "full": {"validation": {"spearman": 0.4, "mae": 0.01}, "validation_ranking": {"mean_regret": 0.01, "quality_capture": 0.99}},
    }
    assert module.quality_selection_key("engineered", ablations) < module.quality_selection_key("full", ablations)


def test_ranking_model_selection_prioritizes_validation_regret():
    module = load_module()
    candidates = {
        "dino/weighted_logistic": {
            "validation_ranking": {
                "mean_regret": 0.03, "quality_capture": 0.99,
                "top1_tie_aware": 0.80,
            },
        },
        "full/extra_trees": {
            "validation_ranking": {
                "mean_regret": 0.01, "quality_capture": 0.96,
                "top1_tie_aware": 0.60,
            },
        },
    }
    assert module.ranking_selection_key("full/extra_trees", candidates) < module.ranking_selection_key(
        "dino/weighted_logistic", candidates
    )


@pytest.mark.parametrize(
    "name", ["weighted_logistic", "extra_trees", "hist_gradient_boosting"]
)
def test_registered_rankers_accept_weights_and_predict_probabilities(name):
    module = load_module()
    x = np.asarray([
        [-2.0, -1.0], [-1.0, -0.5], [1.0, 0.5], [2.0, 1.0],
        [-1.5, -0.7], [1.5, 0.7],
    ])
    y = np.asarray([0, 0, 1, 1, 0, 1])
    weight = np.asarray([2.0, 1.0, 1.0, 2.0, 0.5, 0.5])
    model = module.make_ranker(name, seed=7)
    module.fit_ranker(model, name, x, y, weight)
    probability = model.predict_proba(x)[:, 1]
    assert probability.shape == (6,)
    assert np.all((probability >= 0.0) & (probability <= 1.0))


def test_unregistered_ranker_is_rejected():
    module = load_module()
    with pytest.raises(ValueError, match="unknown ranker"):
        module.make_ranker("neural_ranker", seed=7)


def test_failure_threshold_is_fit_from_validation_probabilities():
    module = load_module()
    truth = np.asarray([0, 0, 1, 1]); probability = np.asarray([0.1, 0.4, 0.45, 0.9])
    threshold, score = module.select_failure_threshold(truth, probability)
    assert threshold == 0.45
    assert score == 1.0


def test_v2_cannot_overwrite_v1_checkpoint_or_metrics():
    module = load_module()
    with pytest.raises(ValueError, match="must not overwrite"):
        module.validate_v2_output_paths(
            Path("/root/r3dguard/checkpoints/ready3d/ready3d.joblib"),
            Path("/root/r3dguard/evaluation/ready3d_v2"), "ready3d-v2",
        )


def test_failure_support_rejects_constant_labels_in_any_split():
    module = load_module()
    failures = np.asarray([
        [0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [1, 1],
    ], dtype=np.int8)
    masks = {
        "train": np.asarray([1, 1, 0, 0, 0, 0], dtype=bool),
        "validation": np.asarray([0, 0, 1, 1, 0, 0], dtype=bool),
        "test": np.asarray([0, 0, 0, 0, 1, 1], dtype=bool),
    }
    try:
        module.validate_failure_support(failures, masks, ["constant", "balanced"])
    except ValueError as exc:
        assert "constant/train" in str(exc)
    else:
        raise AssertionError("Constant failure labels must be rejected")


def test_failure_probabilities_returns_zero_for_one_class_estimators():
    module = load_module()

    class Estimator:
        classes_ = np.asarray([0])

        def predict_proba(self, x):
            return np.ones((len(x), 1))

    class Model:
        estimators_ = [Estimator()]

    result = module.failure_probabilities(Model(), np.zeros((3, 2)))

    assert result.shape == (3, 1)
    assert np.array_equal(result, np.zeros((3, 1)))


def test_best_of_four_metrics_reject_incomplete_groups():
    module = load_module()
    class Ranker:
        def predict_proba(self, x):
            return np.tile([[0.5, 0.5]], (len(x), 1))
    rows = [{"group_id": "g", "sample_id": str(i)} for i in range(3)]
    try:
        module.best_of_four_metrics(rows, np.ones(3), np.ones((3, 2)), np.ones(3, dtype=bool), Ranker())
    except ValueError as exc:
        assert "four candidates" in str(exc).lower()
    else:
        raise AssertionError("Incomplete Best-of-4 groups must be rejected")


def test_training_main_writes_checkpoint_metrics_and_figures(tmp_path, monkeypatch):
    module = load_module()
    labels, features, embeddings = [], [], []
    failure_names = module.FAILURE_TYPES
    for group in range(12):
        split = "train" if group < 4 else "validation" if group < 8 else "test"
        for candidate in range(4):
            index = len(labels)
            sample_id = f"g{group}_c{candidate}"
            quality = 0.55 + 0.08 * candidate + 0.002 * group
            labels.append({
                "sample_id": sample_id, "group_id": f"g{group}", "candidate_index": candidate,
                "domain": "ai_sana", "split": split,
                "ready_target": {
                    "quality": quality,
                    "failure_types": {name: bool((group + candidate + offset) % 2) for offset, name in enumerate(failure_names)},
                },
            })
            features.append({
                "sample_id": sample_id, "feature_index": index,
                "image_signal": float(candidate), "depth_signal": float(group) / 12,
            })
            embeddings.append(np.asarray([candidate, group / 12, quality, 1, 0, 0, 0, 0], dtype=np.float32))

    labels_path, features_path = tmp_path / "labels.jsonl", tmp_path / "features.jsonl"
    labels_path.write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
    features_path.write_text("".join(json.dumps(row) + "\n" for row in features), encoding="utf-8")
    embeddings_path = tmp_path / "embeddings.npy"; np.save(embeddings_path, np.stack(embeddings))
    checkpoint, output = tmp_path / "ready3d.joblib", tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", [
        "train_ready3d.py", "--labels", str(labels_path), "--features", str(features_path),
        "--embeddings", str(embeddings_path), "--checkpoint", str(checkpoint), "--output-dir", str(output),
    ])

    module.main()

    metrics = json.loads((output / "metrics.json").read_text())
    assert checkpoint.is_file()
    assert metrics["ranking"]["groups"] == 4
    assert metrics["ranking"]["training_pairs"] > 0
    assert metrics["ranking"]["training_positive_pairs"] == metrics["ranking"]["training_negative_pairs"]
    assert set(metrics["failure"]["support"]) == set(failure_names)
    assert all(
        set(splits) == {"train", "validation", "test"}
        for splits in metrics["failure"]["support"].values()
    )
    assert metrics["by_domain"]["ai_sana"]["test_samples"] == 16
    assert metrics["model_selection"]["ranking_priority"] == [
        "mean_regret_min", "quality_capture_max", "top1_max"
    ]
    assert metrics["selected_ranking_model"] in {
        "weighted_logistic", "extra_trees", "hist_gradient_boosting"
    }
    assert metrics["selected_ranking_features"] in {"engineered", "dino", "full"}
    assert (output / "figures/quality_scatter.png").is_file()
    assert (output / "best_of_four.jsonl").is_file()
