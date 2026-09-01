"""Pure calibration and aggregation logic for automatic 3D quality labels."""

from __future__ import annotations

from typing import Iterable

import numpy as np


METRIC_DIRECTIONS = {
    "dino_similarity": "positive",
    "silhouette_iou": "positive",
    "contour_similarity": "positive",
    "largest_component_area_ratio": "positive",
    "debris_area_ratio": "negative",
    "non_manifold_edge_length_ratio": "negative",
    "boundary_edge_length_ratio": "negative",
}


def fit_calibration(records: Iterable[dict], low_percentile: float = 5, high_percentile: float = 95) -> dict:
    rows = list(records)
    if not rows:
        raise ValueError("Cannot fit calibration without records")
    calibration = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite calibration values for {metric}")
        calibration[metric] = {
            "low": float(np.percentile(values, low_percentile)),
            "high": float(np.percentile(values, high_percentile)),
            "direction": direction,
        }
    return calibration


def normalize_metric(value: float, spec: dict) -> float:
    value, low, high = float(value), float(spec["low"]), float(spec["high"])
    if not np.isfinite(value):
        raise ValueError("Metric value must be finite")
    if high < low:
        raise ValueError("Calibration high bound cannot be below low bound")
    if high == low:
        if spec["direction"] == "positive":
            return 1.0 if value >= high else 0.0
        if spec["direction"] == "negative":
            return 1.0 if value <= low else 0.0
        raise ValueError(f"Unknown metric direction: {spec['direction']}")
    else:
        score = float(np.clip((value - low) / (high - low), 0.0, 1.0))
    if spec["direction"] == "negative":
        score = 1.0 - score
    elif spec["direction"] != "positive":
        raise ValueError(f"Unknown metric direction: {spec['direction']}")
    return score


def _hard_gate_failures(raw: dict) -> list[str]:
    failures = []
    if not raw["glb_loadable"]:
        failures.append("glb_unloadable")
    if not raw["finite"]:
        failures.append("non_finite_geometry")
    if int(raw["faces"]) <= 0:
        failures.append("no_valid_faces")
    if int(raw["render_count"]) != 8:
        failures.append("incomplete_eight_view_render")
    if float(raw["largest_component_area_ratio"]) < 0.65:
        failures.append("largest_component_area_below_0.65")
    if float(raw["textured_surface_ratio"]) < 0.50:
        failures.append("textured_surface_below_0.50")
    return failures


def score_record(raw: dict, calibration: dict) -> dict:
    scores = {
        metric: normalize_metric(raw[metric], calibration[metric])
        for metric in METRIC_DIRECTIONS
    }
    input_match = (
        30 * scores["dino_similarity"]
        + 10 * scores["silhouette_iou"]
        + 5 * scores["contour_similarity"]
    ) / 45
    geometry = (
        45 * scores["largest_component_area_ratio"]
        + 20 * scores["debris_area_ratio"]
        + 15 * scores["non_manifold_edge_length_ratio"]
        + 20 * scores["boundary_edge_length_ratio"]
    ) / 100
    texture_available = float(raw["textured_surface_ratio"]) >= 0.50
    quality = 0.53 * input_match + 0.47 * geometry
    failures = _hard_gate_failures(raw)
    return {
        "quality": float(quality),
        "input_match": float(input_match),
        "geometry": float(geometry),
        "texture_available": texture_available,
        "normalized_metrics": scores,
        "hard_gate_passed": not failures,
        "failure_reasons": failures,
    }


def fit_quality_threshold(scored_records: Iterable[dict], percentile: float = 70) -> float:
    values = [
        float(row["quality"])
        for row in scored_records
        if row["hard_gate_passed"]
    ]
    if not values:
        raise ValueError("No hard-gate-passing records for threshold fitting")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))
