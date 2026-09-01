"""Canonical Ready3D V2 calibration, hard gates, and quality aggregation."""

from __future__ import annotations

from typing import Iterable

import numpy as np


SCORING_VERSION = "ready3d-v2"
TARGET_SCHEMA_VERSION = "r3dguard.ready3d-target.v2"
CALIBRATION_VERSION = "ready3d-v2-train-p5-p95"
METRIC_DIRECTIONS = {
    "dino_similarity": "positive",
    "silhouette_iou": "positive",
    "contour_similarity": "positive",
    "largest_component_area_ratio": "positive",
    "debris_area_ratio": "negative",
    "boundary_edge_length_ratio": "negative",
    "non_manifold_edge_length_ratio": "negative",
}
MATCH_WEIGHTS = {
    "dino_similarity": 30.0 / 45.0,
    "silhouette_iou": 10.0 / 45.0,
    "contour_similarity": 5.0 / 45.0,
}
GEOMETRY_WEIGHTS = {
    "largest_component_area_ratio": 0.45,
    "debris_area_ratio": 0.20,
    "boundary_edge_length_ratio": 0.20,
    "non_manifold_edge_length_ratio": 0.15,
}
QUALITY_WEIGHTS = {"input_match": 0.53, "geometry": 0.47}


def _rows(records: Iterable[dict], required_split: str | None = None) -> list[dict]:
    rows = list(records)
    if not rows:
        raise ValueError("records must not be empty")
    if required_split is not None:
        invalid = sorted(
            {str(row.get("split")) for row in rows if row.get("split") != required_split}
        )
        if invalid:
            raise ValueError(f"expected only {required_split} records; found {invalid}")
    return rows


def fit_normalization(
    records: Iterable[dict],
    required_split: str = "train",
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
) -> dict:
    """Fit P5/P95 normalization using training records only."""
    rows = _rows(records, required_split)
    metrics = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite training values for {metric}")
        metrics[metric] = {
            "low": float(np.percentile(values, low_percentile)),
            "high": float(np.percentile(values, high_percentile)),
            "direction": direction,
        }
    return {
        "calibration_version": CALIBRATION_VERSION,
        "normalization_source_split": required_split,
        "low_percentile": float(low_percentile),
        "high_percentile": float(high_percentile),
        "training_records": len(rows),
        "metrics": metrics,
    }


def normalize_metric(value: float, spec: dict) -> float:
    value, low, high = float(value), float(spec["low"]), float(spec["high"])
    if not np.isfinite(value) or not np.isfinite([low, high]).all():
        raise ValueError("metric and calibration bounds must be finite")
    if high < low:
        raise ValueError("calibration high bound cannot be below low bound")
    direction = spec["direction"]
    if high == low:
        if direction == "positive":
            return float(value >= high)
        if direction == "negative":
            return float(value <= low)
        raise ValueError(f"unknown metric direction: {direction}")
    score = float(np.clip((value - low) / (high - low), 0.0, 1.0))
    if direction == "negative":
        return 1.0 - score
    if direction != "positive":
        raise ValueError(f"unknown metric direction: {direction}")
    return score


def technical_failures(raw: dict) -> list[str]:
    failures = []
    if not bool(raw.get("glb_loadable")):
        failures.append("glb_unloadable")
    if not bool(raw.get("finite")):
        failures.append("non_finite_geometry")
    if int(raw.get("faces", 0)) <= 0:
        failures.append("no_valid_faces")
    if int(raw.get("render_count", 0)) != 8:
        failures.append("incomplete_eight_view_render")
    if float(raw.get("largest_component_area_ratio", 0.0)) < 0.65:
        failures.append("largest_component_area_below_0.65")
    if float(raw.get("textured_surface_ratio", 0.0)) < 0.50:
        failures.append("textured_surface_below_0.50")
    return failures


def score_record(raw: dict, calibration: dict) -> dict:
    specs = calibration.get("metrics", calibration)
    normalized = {
        metric: normalize_metric(raw[metric], specs[metric])
        for metric in METRIC_DIRECTIONS
    }
    input_match = float(
        sum(MATCH_WEIGHTS[key] * normalized[key] for key in MATCH_WEIGHTS)
    )
    geometry = float(
        sum(GEOMETRY_WEIGHTS[key] * normalized[key] for key in GEOMETRY_WEIGHTS)
    )
    quality = float(
        QUALITY_WEIGHTS["input_match"] * input_match
        + QUALITY_WEIGHTS["geometry"] * geometry
    )
    failures = technical_failures(raw)
    return {
        "scoring_version": SCORING_VERSION,
        "quality_score_v2": quality,
        "input_match": input_match,
        "geometry": geometry,
        "texture_available": float(raw["textured_surface_ratio"]) >= 0.50,
        "technically_valid": not failures,
        "technical_failure_reasons": failures,
        "normalized_metrics": normalized,
    }


def fit_high_quality_threshold(
    scored_records: Iterable[dict],
    required_split: str = "validation",
    percentile: float = 70.0,
) -> dict:
    """Freeze the relative high-quality threshold from validation only."""
    rows = _rows(scored_records, required_split)
    values = np.asarray(
        [
            float(row["quality_score_v2"])
            for row in rows
            if row["technically_valid"]
        ],
        dtype=np.float64,
    )
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("validation has no finite technically-valid scores")
    return {
        "threshold_source_split": required_split,
        "threshold_percentile": float(percentile),
        "threshold": float(np.percentile(values, percentile)),
        "validation_records": len(rows),
        "validation_valid_records": len(values),
    }


def apply_high_quality(scored: dict, threshold: float) -> dict:
    row = dict(scored)
    high_quality = bool(
        row["technically_valid"]
        and float(row["quality_score_v2"]) >= float(threshold)
    )
    row["high_quality"] = high_quality
    row["qualified"] = bool(row["technically_valid"] and high_quality)
    return row
