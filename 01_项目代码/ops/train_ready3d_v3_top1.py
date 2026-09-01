#!/usr/bin/env python3
"""Train the independent Ready3D V3 group-aware Top-1 selector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ops.train_ready3d import (  # noqa: E402
    RANKER_NAMES,
    build_pairwise_data,
    fit_ranker,
    load_dataset,
    make_ranker,
)
from r3dloop.ready3d_v3.ranking import (  # noqa: E402
    add_group_context,
    combine_group_scores,
    groupwise_top1_metrics,
    pairwise_group_scores,
    select_ensemble_weights,
    select_top1,
    validate_complete_groups,
)


def split_mask(rows: list[dict], split: str) -> np.ndarray:
    return np.asarray([row["split"] == split for row in rows], dtype=bool)


def structural_pass_from_scalar(
    scalar: np.ndarray, scalar_names: list[str]
) -> np.ndarray:
    required = {
        "mask_area_ratio",
        "mask_largest_component_ratio",
        "occlusion_border_contact_ratio",
    }
    missing = required - set(scalar_names)
    if missing:
        raise ValueError(f"missing structural scalar features: {sorted(missing)}")
    columns = {name: scalar_names.index(name) for name in required}
    area = scalar[:, columns["mask_area_ratio"]]
    largest = scalar[:, columns["mask_largest_component_ratio"]]
    border = scalar[:, columns["occlusion_border_contact_ratio"]]
    return (area >= 0.02) & (area <= 0.92) & (largest >= 0.75) & (border <= 0.10)


def _rows_for_mask(rows: list[dict], mask: np.ndarray) -> list[dict]:
    return [rows[index] for index in np.flatnonzero(mask)]


def _selection_key(metrics: dict, name: str) -> tuple:
    return (
        metrics["mean_regret"],
        -metrics["quality_capture"],
        -metrics["top1_accuracy"],
        name,
    )


def fit_top1_ensemble(
    data: dict, seed: int = 20260827, n_estimators: int = 300
) -> tuple[dict, dict, list[dict]]:
    rows = data["rows"]
    quality = np.asarray(data["quality"], dtype=np.float64)
    groups = [str(row["group_id"]) for row in rows]
    masks = {split: split_mask(rows, split) for split in ("train", "validation", "test")}
    for split, mask in masks.items():
        validate_complete_groups(_rows_for_mask(rows, mask))
        if not mask.any():
            raise ValueError(f"empty split: {split}")

    raw_matrices = {
        "engineered": np.asarray(data["scalar"], dtype=np.float64),
        "dino": np.asarray(data["dino"], dtype=np.float64),
        "full": np.concatenate([data["dino"], data["scalar"]], axis=1).astype(np.float64),
    }
    matrices = {
        name: add_group_context(matrix, groups) for name, matrix in raw_matrices.items()
    }
    structural_pass = structural_pass_from_scalar(
        np.asarray(data["scalar"]), list(data["scalar_names"])
    )

    quality_models = {}
    utility_predictions = {}
    for feature_set, matrix in matrices.items():
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(matrix[masks["train"]], quality[masks["train"]])
        quality_models[feature_set] = model
        utility_predictions[feature_set] = model.predict(matrix)

    ranking_models = {}
    pairwise_predictions = {}
    train_mask = masks["train"]
    for feature_set, matrix in matrices.items():
        pair_x, pair_y, pair_weight = build_pairwise_data(
            matrix, quality, groups, train_mask
        )
        for ranker_name in RANKER_NAMES:
            name = f"{feature_set}/{ranker_name}"
            model = make_ranker(ranker_name, seed)
            if hasattr(model, "set_params") and ranker_name == "extra_trees":
                model.set_params(n_estimators=n_estimators)
            fit_ranker(model, ranker_name, pair_x, pair_y, pair_weight)
            ranking_models[name] = model
            pairwise_predictions[name] = pairwise_group_scores(model, matrix, rows)

    validation = masks["validation"]
    validation_rows = _rows_for_mask(rows, validation)
    candidates = []
    for ranking_name, pairwise in pairwise_predictions.items():
        for quality_name, utility in utility_predictions.items():
            weights, validation_metrics = select_ensemble_weights(
                validation_rows,
                quality[validation],
                pairwise[validation],
                utility[validation],
                structural_pass[validation],
            )
            name = f"{ranking_name}+{quality_name}"
            candidates.append(
                {
                    "name": name,
                    "ranking_name": ranking_name,
                    "ranking_feature_set": ranking_name.split("/", 1)[0],
                    "ranking_model_name": ranking_name.split("/", 1)[1],
                    "quality_feature_set": quality_name,
                    "weights": weights,
                    "validation": validation_metrics,
                }
            )
    selected = min(candidates, key=lambda row: _selection_key(row["validation"], row["name"]))

    test = masks["test"]
    test_rows = _rows_for_mask(rows, test)
    test_scores = combine_group_scores(
        test_rows,
        pairwise_predictions[selected["ranking_name"]][test],
        utility_predictions[selected["quality_feature_set"]][test],
        structural_pass[test],
        selected["weights"],
    )
    test_selections = select_top1(test_rows, test_scores, structural_pass[test])
    test_metrics, test_records = groupwise_top1_metrics(
        test_rows, quality[test], test_selections
    )

    validation_selected_scores = combine_group_scores(
        validation_rows,
        pairwise_predictions[selected["ranking_name"]][validation],
        utility_predictions[selected["quality_feature_set"]][validation],
        structural_pass[validation],
        selected["weights"],
    )
    validation_selections = select_top1(
        validation_rows, validation_selected_scores, structural_pass[validation]
    )
    validation_selected_ids = {
        row["selected_sample_id"] for row in validation_selections
    }
    validation_selected_prediction = np.asarray(
        [
            utility_predictions[selected["quality_feature_set"]][index]
            for index in np.flatnonzero(validation)
            if rows[index]["sample_id"] in validation_selected_ids
        ],
        dtype=np.float64,
    )
    resample_threshold = float(np.quantile(validation_selected_prediction, 0.25))

    model_selection = {
        "split": "validation",
        "priority": ["mean_regret_min", "quality_capture_max", "top1_accuracy_max"],
        "test_used_for_selection": False,
    }
    checkpoint = {
        "schema_version": "r3dguard.ready3d-v3-top1-checkpoint.v1",
        "scoring_version": data.get("scoring_version", "unknown"),
        "quality_model": quality_models[selected["quality_feature_set"]],
        "quality_feature_set": selected["quality_feature_set"],
        "ranking_model": ranking_models[selected["ranking_name"]],
        "ranking_feature_set": selected["ranking_feature_set"],
        "ranking_model_name": selected["ranking_model_name"],
        "ensemble_weights": selected["weights"],
        "scalar_feature_names": list(data["scalar_names"]),
        "expected_candidates": 4,
        "resample_suggestion_threshold": resample_threshold,
        "model_selection": model_selection,
        "feature_version": 3,
    }
    metrics = {
        "schema_version": "r3dguard.ready3d-v3-top1-metrics.v1",
        "model_selection": model_selection,
        "selected": {key: value for key, value in selected.items() if key != "validation"},
        "validation": selected["validation"],
        "test": test_metrics,
        "group_counts": {
            split: len({rows[index]["group_id"] for index in np.flatnonzero(mask)})
            for split, mask in masks.items()
        },
        "resample_suggestion_threshold": resample_threshold,
        "selection_candidates": [
            {
                "name": row["name"],
                "weights": row["weights"],
                "validation": row["validation"],
            }
            for row in candidates
        ],
    }
    return checkpoint, metrics, test_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--n-estimators", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(args.labels, args.features, args.embeddings)
    checkpoint, metrics, records = fit_top1_ensemble(
        data, seed=args.seed, n_estimators=args.n_estimators
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(checkpoint, args.checkpoint)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "test_top1.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
