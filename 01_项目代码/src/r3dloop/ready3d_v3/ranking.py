"""Group-aware, deterministic Top-1 ranking primitives for Ready3D V3."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations

import numpy as np


def _group_indices(rows: Sequence[dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["group_id"])].append(index)
    return dict(groups)


def validate_complete_groups(rows: Sequence[dict], expected_candidates: int = 4) -> None:
    if not rows:
        raise ValueError("at least one candidate group is required")
    for group_id, indices in _group_indices(rows).items():
        if len(indices) != expected_candidates:
            raise ValueError(
                f"{group_id}: expected {expected_candidates} candidates, got {len(indices)}"
            )
        candidate_indices = [int(rows[index]["candidate_index"]) for index in indices]
        expected = list(range(expected_candidates))
        if sorted(candidate_indices) != expected:
            raise ValueError(
                f"{group_id}: candidate_index must be exactly {expected}, got {candidate_indices}"
            )


def add_group_context(x: np.ndarray, group_ids: Sequence[str]) -> np.ndarray:
    """Append within-group centred and standardized features.

    Absolute appearance statistics remain available in ``x`` while the extra blocks
    make relative differences between the four candidates explicit to a small model.
    """

    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(group_ids):
        raise ValueError("x must be 2-D and aligned with group_ids")
    centred = np.zeros_like(values)
    standardized = np.zeros_like(values)
    group_array = np.asarray(group_ids, dtype=object)
    for group_id in sorted(set(group_ids)):
        mask = group_array == group_id
        group = values[mask]
        centred[mask] = group - group.mean(axis=0)
        scale = group.std(axis=0)
        standardized[mask] = centred[mask] / np.where(scale > 1e-12, scale, 1.0)
    return np.concatenate([values, centred, standardized], axis=1)


def _normalize_within_groups(rows: Sequence[dict], values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.shape != (len(rows),):
        raise ValueError("scores must contain exactly one value per row")
    output = np.zeros_like(source)
    for indices in _group_indices(rows).values():
        group = source[indices]
        span = float(group.max() - group.min())
        output[indices] = (group - group.min()) / span if span > 1e-12 else 0.5
    return output


def combine_group_scores(
    rows: Sequence[dict],
    pairwise: np.ndarray,
    utility: np.ndarray,
    structural_pass: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    """Combine relative and absolute quality signals; structure remains a hard gate."""

    passes = np.asarray(structural_pass, dtype=bool)
    if passes.shape != (len(rows),):
        raise ValueError("structural_pass must contain exactly one value per row")
    pairwise_weight = float(weights.get("pairwise", 0.0))
    utility_weight = float(weights.get("utility", 0.0))
    total = pairwise_weight + utility_weight
    if total <= 0:
        raise ValueError("pairwise and utility weights must have a positive sum")
    score = (
        pairwise_weight * _normalize_within_groups(rows, pairwise)
        + utility_weight * _normalize_within_groups(rows, utility)
    ) / total
    # The numerical margin is deliberately much larger than the normalized range.
    # It implements the policy: rank only among structural passes when one exists.
    for indices in _group_indices(rows).values():
        if passes[indices].any():
            score[np.asarray(indices)[~passes[indices]]] -= 2.0
    return score


def pairwise_group_scores(model, x: np.ndarray, rows: Sequence[dict]) -> np.ndarray:
    """Return Borda-style pairwise scores without comparing different groups."""

    values = np.asarray(x)
    if values.ndim != 2 or len(values) != len(rows):
        raise ValueError("x must be 2-D and aligned with rows")
    output = np.zeros(len(rows), dtype=np.float64)
    for indices in _group_indices(rows).values():
        for left, right in combinations(indices, 2):
            probability = float(
                model.predict_proba((values[left] - values[right])[None])[0, 1]
            )
            output[left] += probability
            output[right] += 1.0 - probability
    return output


def select_top1(
    rows: Sequence[dict], scores: np.ndarray, structural_pass: np.ndarray
) -> list[dict]:
    values = np.asarray(scores, dtype=np.float64)
    passes = np.asarray(structural_pass, dtype=bool)
    if values.shape != (len(rows),) or passes.shape != (len(rows),):
        raise ValueError("scores and structural_pass must align with rows")
    selections = []
    for group_id, indices in sorted(_group_indices(rows).items()):
        eligible = [index for index in indices if passes[index]]
        pool = eligible or indices
        selected = min(
            pool,
            key=lambda index: (
                -float(values[index]),
                int(rows[index]["candidate_index"]),
                str(rows[index]["sample_id"]),
            ),
        )
        selections.append(
            {
                "group_id": group_id,
                "selected_sample_id": str(rows[selected]["sample_id"]),
                "selected_candidate_index": int(rows[selected]["candidate_index"]),
                "selection_score": float(values[selected]),
                "structural_input_pass": bool(passes[selected]),
                "selection_fallback": None
                if eligible
                else "all_candidates_failed_structure_gate",
            }
        )
    return selections


def groupwise_top1_metrics(
    rows: Sequence[dict], quality: np.ndarray, selections: Sequence[dict]
) -> tuple[dict, list[dict]]:
    truth = np.asarray(quality, dtype=np.float64)
    if truth.shape != (len(rows),):
        raise ValueError("quality must contain exactly one value per row")
    row_by_id = {str(row["sample_id"]): index for index, row in enumerate(rows)}
    selection_by_group = {str(row["group_id"]): row for row in selections}
    records = []
    for group_id, indices in sorted(_group_indices(rows).items()):
        if group_id not in selection_by_group:
            raise ValueError(f"missing selection for {group_id}")
        selected_id = str(selection_by_group[group_id]["selected_sample_id"])
        selected = row_by_id.get(selected_id)
        if selected is None or selected not in indices:
            raise ValueError(f"selection {selected_id} does not belong to {group_id}")
        optimum = min(
            indices,
            key=lambda index: (-float(truth[index]), int(rows[index]["candidate_index"])),
        )
        optimal_quality = float(truth[optimum])
        selected_quality = float(truth[selected])
        records.append(
            {
                "group_id": group_id,
                "selected_sample_id": selected_id,
                "selected_quality": selected_quality,
                "optimal_sample_id": str(rows[optimum]["sample_id"]),
                "optimal_quality": optimal_quality,
                "regret": optimal_quality - selected_quality,
                "top1_correct": abs(optimal_quality - selected_quality) <= 1e-8,
            }
        )
    return {
        "groups": len(records),
        "mean_selected_quality": float(np.mean([row["selected_quality"] for row in records])),
        "top1_accuracy": float(np.mean([row["top1_correct"] for row in records])),
        "mean_regret": float(np.mean([row["regret"] for row in records])),
        "quality_capture": float(
            np.mean(
                [
                    row["selected_quality"] / max(row["optimal_quality"], 1e-8)
                    for row in records
                ]
            )
        ),
    }, records


def select_ensemble_weights(
    rows: Sequence[dict],
    quality: np.ndarray,
    pairwise: np.ndarray,
    utility: np.ndarray,
    structural_pass: np.ndarray,
    step: float = 0.1,
) -> tuple[dict[str, float], dict]:
    """Freeze blend weights using validation groups only.

    The caller supplies validation rows explicitly. Ties prefer more pairwise weight,
    keeping the model ranking-first when both signals perform equally.
    """

    validate_complete_groups(rows)
    if not 0 < step <= 1:
        raise ValueError("step must be in (0, 1]")
    candidates = []
    for pairwise_weight in np.arange(0.0, 1.0 + step / 2.0, step):
        pairwise_weight = min(float(pairwise_weight), 1.0)
        weights = {"pairwise": pairwise_weight, "utility": 1.0 - pairwise_weight}
        scores = combine_group_scores(
            rows, pairwise, utility, structural_pass, weights
        )
        selections = select_top1(rows, scores, structural_pass)
        metrics, _ = groupwise_top1_metrics(rows, quality, selections)
        key = (
            metrics["mean_regret"],
            -metrics["quality_capture"],
            -metrics["top1_accuracy"],
            -pairwise_weight,
        )
        candidates.append((key, weights, metrics))
    _, weights, metrics = min(candidates, key=lambda row: row[0])
    return weights, metrics
