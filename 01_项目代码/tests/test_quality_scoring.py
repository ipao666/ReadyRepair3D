import math
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from quality_scoring import (  # noqa: E402
    fit_calibration,
    fit_quality_threshold,
    normalize_metric,
    score_record,
)


def good_raw(**changes):
    row = {
        "glb_loadable": True,
        "finite": True,
        "faces": 1000,
        "render_count": 8,
        "dino_similarity": 0.8,
        "silhouette_iou": 0.8,
        "contour_similarity": 0.8,
        "largest_component_area_ratio": 0.9,
        "debris_area_ratio": 0.01,
        "non_manifold_edge_length_ratio": 0.01,
        "boundary_edge_length_ratio": 0.01,
        "textured_surface_ratio": 0.9,
    }
    row.update(changes)
    return row


def unit_calibration():
    positive = {
        "dino_similarity",
        "silhouette_iou",
        "contour_similarity",
        "largest_component_area_ratio",
        "textured_surface_ratio",
    }
    metrics = set(good_raw()) - {
        "glb_loadable", "finite", "faces", "render_count"
    }
    return {
        metric: {
            "low": 0.0,
            "high": 1.0,
            "direction": "positive" if metric in positive else "negative",
        }
        for metric in metrics
    }


def test_normalize_metric_clips_and_inverts_negative_metrics():
    assert normalize_metric(2.0, {"low": 0.0, "high": 1.0, "direction": "positive"}) == 1.0
    assert normalize_metric(-1.0, {"low": 0.0, "high": 1.0, "direction": "positive"}) == 0.0
    assert normalize_metric(0.2, {"low": 0.0, "high": 1.0, "direction": "negative"}) == 0.8


def test_constant_clean_negative_metric_receives_full_credit():
    assert normalize_metric(
        0.0, {"low": 0.0, "high": 0.0, "direction": "negative"}
    ) == 1.0


def test_score_record_uses_53_47_weights_without_texture_bonus():
    raw = good_raw(
        dino_similarity=1.0,
        silhouette_iou=1.0,
        contour_similarity=1.0,
        largest_component_area_ratio=0.0,
        debris_area_ratio=1.0,
        non_manifold_edge_length_ratio=1.0,
        boundary_edge_length_ratio=1.0,
        textured_surface_ratio=0.0,
    )
    result = score_record(raw, unit_calibration())
    assert result["input_match"] == 1.0
    assert result["geometry"] == 0.0
    assert result["texture_available"] is False
    assert result["quality"] == 0.53


def test_hard_gate_reports_every_failure_but_keeps_finite_score():
    raw = good_raw(
        glb_loadable=False,
        finite=False,
        faces=0,
        render_count=7,
        largest_component_area_ratio=0.6,
        textured_surface_ratio=0.4,
    )
    result = score_record(raw, unit_calibration())
    assert not result["hard_gate_passed"]
    assert set(result["failure_reasons"]) == {
        "glb_unloadable",
        "non_finite_geometry",
        "no_valid_faces",
        "incomplete_eight_view_render",
        "largest_component_area_below_0.65",
        "textured_surface_below_0.50",
    }
    assert math.isfinite(result["quality"])


def test_calibration_uses_fifth_and_ninety_fifth_percentiles():
    rows = [good_raw(dino_similarity=float(value)) for value in range(101)]
    calibration = fit_calibration(rows)
    assert calibration["dino_similarity"]["low"] == 5.0
    assert calibration["dino_similarity"]["high"] == 95.0
    assert calibration["debris_area_ratio"]["direction"] == "negative"


def test_threshold_uses_only_hard_gate_passes_and_seventieth_percentile():
    scored = [
        {"quality": value / 10, "hard_gate_passed": True} for value in range(10)
    ] + [{"quality": 1.0, "hard_gate_passed": False}]
    assert fit_quality_threshold(scored) == 0.63
