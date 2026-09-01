from pathlib import Path
import sys

import pytest

OPS = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS))

from ready3d_quality_v2 import (
    apply_high_quality,
    fit_high_quality_threshold,
    fit_normalization,
    score_record,
)


def raw(value: float, split: str) -> dict:
    return {
        "split": split,
        "dino_similarity": value,
        "silhouette_iou": value,
        "contour_similarity": value,
        "largest_component_area_ratio": value,
        "debris_area_ratio": 1.0 - value,
        "boundary_edge_length_ratio": 1.0 - value,
        "non_manifold_edge_length_ratio": 1.0 - value,
        "glb_loadable": True,
        "finite": True,
        "faces": 100,
        "render_count": 8,
        "textured_surface_ratio": 1.0,
    }


def test_canonical_v2_score_and_threshold_contract():
    calibration = fit_normalization(
        [raw(0.2, "train"), raw(0.8, "train")]
    )
    low = score_record(raw(0.3, "validation"), calibration)
    high = score_record(raw(0.7, "validation"), calibration)
    threshold = fit_high_quality_threshold(
        [
            {**low, "split": "validation"},
            {**high, "split": "validation"},
        ]
    )

    finalized = apply_high_quality(high, threshold["threshold"])

    assert 0.0 <= finalized["quality_score_v2"] <= 1.0
    assert finalized["technically_valid"] is True
    assert finalized["high_quality"] is True
    assert "technical_failure_reasons" in finalized


def test_technical_gate_rejects_incomplete_render_set():
    calibration = fit_normalization(
        [raw(0.2, "train"), raw(0.8, "train")]
    )
    broken = raw(0.7, "test")
    broken["render_count"] = 7

    result = score_record(broken, calibration)

    assert result["technically_valid"] is False
    assert "incomplete_eight_view_render" in result["technical_failure_reasons"]


def test_normalization_rejects_non_training_rows():
    with pytest.raises(ValueError, match="expected only train"):
        fit_normalization([raw(0.5, "validation")])
