import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "ops" / "predict_ready3d.py"


def load_module():
    spec = importlib.util.spec_from_file_location("predict_ready3d", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_assembly_uses_checkpoint_scalar_order():
    module = load_module()
    matrices = module.assemble_matrices(
        [{"b": 2, "a": 1}, {"b": 4, "a": 3}], np.array([[10, 11], [12, 13]]), ["a", "b"]
    )
    assert matrices["engineered"].tolist() == [[1, 2], [3, 4]]
    assert matrices["full"].tolist()[0] == [10, 11, 1, 2]


def test_matrix_assembly_rejects_missing_features():
    module = load_module()
    try:
        module.assemble_matrices([{"a": 1}], np.ones((1, 2)), ["a", "b"])
    except ValueError as exc:
        assert "missing" in str(exc).lower()
    else:
        raise AssertionError("Missing checkpoint features must be rejected")


def test_predict_source_exposes_v2_scoring_version_and_validation_thresholds():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '"scoring_version"' in source
    assert '"failure_thresholds"' in source
    assert "checkpoints/ready3d_v2/ready3d_v2.joblib" in source


def test_v3_prediction_uses_independent_ranking_feature_set():
    module = load_module()

    class Tree:
        def predict(self, x):
            return x[:, 0]

    class QualityModel:
        estimators_ = [Tree(), Tree()]

        def predict(self, x):
            return x[:, 0]

    class BinaryEstimator:
        classes_ = np.asarray([0, 1])

        def predict_proba(self, x):
            return np.tile([[0.8, 0.2]], (len(x), 1))

    class FailureModel:
        estimators_ = [BinaryEstimator()]

    class Ranker:
        def predict_proba(self, x):
            positive = np.where(x[:, 0] > 0, 0.9, 0.1)
            return np.stack([1.0 - positive, positive], axis=1)

    checkpoint = {
        "quality_feature_set": "engineered",
        "ranking_feature_set": "dino",
        "quality_model": QualityModel(),
        "ranking_model": Ranker(),
        "failure_model": FailureModel(),
        "failure_types": ["depth_ambiguity"],
        "failure_thresholds": {"depth_ambiguity": 0.5},
        "failure_experimental": {},
        "conformal_half_width": 0.1,
        "high_quality_threshold": 0.8,
        "scoring_version": "ready3d-v2",
    }
    matrices = {
        "engineered": np.asarray([[0.9], [0.1]], dtype=np.float32),
        "dino": np.asarray([[0.1], [0.9]], dtype=np.float32),
        "full": np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
    }
    predictions = module.predict_arrays(checkpoint, matrices)
    assert predictions[0]["predicted_quality"] == pytest.approx(0.9)
    assert predictions[1]["selected"] is True
