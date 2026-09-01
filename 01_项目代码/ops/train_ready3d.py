#!/usr/bin/env python3
"""Train Ready3D V2 with validation ranking-first model selection."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FAILURE_TYPES = [
    "silhouette_truncation", "depth_ambiguity", "background_leakage",
    "texture_lighting_conflict", "view_conflict", "insufficient_detail",
]
FEATURE_SETS = ("engineered", "dino", "full")
RANKER_NAMES = ("weighted_logistic", "extra_trees", "hist_gradient_boosting")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_splits(rows: list[dict]) -> None:
    groups: dict[str, str] = {}
    for row in rows:
        previous = groups.setdefault(row["group_id"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"group leakage: {row['group_id']} appears in {previous} and {row['split']}")
    if set(groups.values()) != {"train", "validation", "test"}:
        raise ValueError("train, validation, and test groups are all required")


def load_dataset(labels_path: Path, features_path: Path, embeddings_path: Path) -> dict:
    labels, features = read_jsonl(labels_path), read_jsonl(features_path)
    validate_splits(labels); feature_by_id = {row["sample_id"]: row for row in features}
    embeddings = np.load(embeddings_path)
    scalar_names = sorted(key for key in features[0] if key.startswith(("image_", "mask_", "depth_", "normal_", "occlusion_")))
    scalar, dino, quality, failure, high_quality, rows = [], [], [], [], [], []
    scoring_version = "ready3d-v2" if labels[0].get("schema_version") == "r3dguard.ready3d-target.v2" else "v1_legacy"
    for label in labels:
        feature = feature_by_id.get(label["sample_id"])
        if feature is None: raise ValueError(f"missing features for {label['sample_id']}")
        index = int(feature["feature_index"])
        if index >= len(embeddings): raise ValueError(f"invalid feature_index for {label['sample_id']}")
        scalar.append([float(feature[name]) for name in scalar_names]); dino.append(embeddings[index])
        if scoring_version == "ready3d-v2":
            quality.append(float(label["quality_score_v2"])); labels_failure = label["failure_types"]
            high_quality.append(bool(label["high_quality"]))
        else:
            target = label["ready_target"]; quality.append(float(target["quality"])); labels_failure = target["failure_types"]
            high_quality.append(bool(target.get("qualified", False)))
        failure.append([int(bool(labels_failure[name])) for name in FAILURE_TYPES]); rows.append(label)
    return {
        "rows": rows, "scalar_names": scalar_names, "scalar": np.asarray(scalar, dtype=np.float32),
        "dino": np.asarray(dino, dtype=np.float32), "quality": np.asarray(quality, dtype=np.float32),
        "failure": np.asarray(failure, dtype=np.int8), "high_quality": np.asarray(high_quality, dtype=bool),
        "scoring_version": scoring_version,
    }


def split_mask(rows: list[dict], split: str) -> np.ndarray:
    return np.asarray([row["split"] == split for row in rows])


def validate_failure_support(failures: np.ndarray, split_masks: dict[str, np.ndarray], failure_types: list[str] = FAILURE_TYPES) -> None:
    invalid = {}
    for index, name in enumerate(failure_types):
        for split, mask in split_masks.items():
            samples, positive = int(mask.sum()), int(failures[mask, index].sum())
            if not 0 < positive < samples: invalid[f"{name}/{split}"] = {"positive": positive, "samples": samples}
    if invalid: raise ValueError(f"failure labels require positive and negative support in every split: {invalid}")


def make_regressor(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.7, random_state=seed, n_jobs=-1)


def make_ranker(name: str, seed: int):
    if name == "weighted_logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=3000, random_state=seed),
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.7,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )
    raise ValueError(f"unknown ranker: {name}")


def fit_ranker(model, name: str, x: np.ndarray, y: np.ndarray, weight: np.ndarray):
    if name == "weighted_logistic":
        model.fit(x, y, logisticregression__sample_weight=weight)
    else:
        model.fit(x, y, sample_weight=weight)
    return model


def regressor_uncertainty(model: ExtraTreesRegressor, x: np.ndarray) -> np.ndarray:
    return np.stack([tree.predict(x) for tree in model.estimators_], axis=0).std(axis=0)


def build_pairwise_data(
    x: np.ndarray,
    quality: np.ndarray,
    groups: list[str],
    mask: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_x, pair_y, pair_weight = [], [], []
    for group in sorted(set(groups[index] for index in np.flatnonzero(mask))):
        indices = [index for index in np.flatnonzero(mask) if groups[index] == group]
        for left, right in combinations(indices, 2):
            delta = float(quality[left] - quality[right])
            if abs(delta) <= epsilon: continue
            pair_x.extend([x[left] - x[right], x[right] - x[left]])
            pair_y.extend([int(delta > 0), int(delta < 0)])
            pair_weight.extend([abs(delta), abs(delta)])
    if not pair_x: raise ValueError("no non-tied training pairs")
    weights = np.asarray(pair_weight, dtype=np.float64)
    weights = np.clip(weights / weights.mean(), 0.25, 4.0)
    return np.asarray(pair_x), np.asarray(pair_y), weights


def rank_scores(model, x: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(x), dtype=np.float64)
    for left, right in combinations(range(len(x)), 2):
        probability = float(model.predict_proba((x[left] - x[right])[None])[0, 1])
        scores[left] += probability; scores[right] += 1.0 - probability
    return scores


def best_of_four_metrics(rows: list[dict], quality: np.ndarray, x: np.ndarray, mask: np.ndarray, ranker) -> tuple[dict, list[dict]]:
    groups = sorted({rows[index]["group_id"] for index in np.flatnonzero(mask)}); records = []
    for group in groups:
        indices = [index for index in np.flatnonzero(mask) if rows[index]["group_id"] == group]
        if len(indices) != 4: raise ValueError(f"{group}: expected four candidates, got {len(indices)}")
        selected = indices[int(np.argmax(rank_scores(ranker, x[indices])))]
        optimum, chosen = float(max(quality[index] for index in indices)), float(quality[selected])
        records.append({"group_id": group, "selected_sample_id": rows[selected]["sample_id"], "selected_quality": chosen, "optimal_quality": optimum, "regret": optimum - chosen, "top1_tie_aware": abs(chosen - optimum) <= 1e-8})
    return {
        "groups": len(records), "top1_tie_aware": float(np.mean([row["top1_tie_aware"] for row in records])),
        "mean_regret": float(np.mean([row["regret"] for row in records])),
        "quality_capture": float(np.mean([row["selected_quality"] / max(row["optimal_quality"], 1e-8) for row in records])),
    }, records


def random_selection_metrics(rows: list[dict], quality: np.ndarray, mask: np.ndarray) -> dict:
    records = []
    for group in sorted({rows[index]["group_id"] for index in np.flatnonzero(mask)}):
        values = np.asarray([quality[index] for index in np.flatnonzero(mask) if rows[index]["group_id"] == group])
        optimum = float(values.max()); records.append((float(values.mean()), optimum, float(np.mean(np.isclose(values, optimum)))))
    return {
        "top1_tie_aware": float(np.mean([row[2] for row in records])),
        "mean_regret": float(np.mean([row[1] - row[0] for row in records])),
        "quality_capture": float(np.mean([row[0] / max(row[1], 1e-8) for row in records])),
    }


def evaluate_regression(y: np.ndarray, prediction: np.ndarray) -> dict:
    correlation = spearmanr(y, prediction).statistic if len(y) > 1 else float("nan")
    return {"mae": float(mean_absolute_error(y, prediction)), "rmse": float(mean_squared_error(y, prediction) ** 0.5), "spearman": float(correlation) if np.isfinite(correlation) else None}


def conformal_quantile(residuals: np.ndarray, coverage: float = 0.90) -> float:
    if not 0 < coverage < 1 or len(residuals) == 0: raise ValueError("non-empty residuals and coverage in (0, 1) are required")
    rank = min(len(residuals), int(np.ceil((len(residuals) + 1) * coverage)))
    return float(np.sort(np.abs(residuals))[rank - 1])


def select_failure_threshold(truth: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate(([0.0, 1.0], probability)))
    scored = [(float(f1_score(truth, probability >= value, zero_division=0)), float(value)) for value in candidates]
    best_f1 = max(row[0] for row in scored)
    threshold = min((row[1] for row in scored if row[0] == best_f1), key=lambda value: (abs(value - 0.5), value))
    return threshold, best_f1


def failure_probabilities(model: MultiOutputClassifier, x: np.ndarray) -> np.ndarray:
    values = []
    for estimator in model.estimators_:
        positive = np.flatnonzero(estimator.classes_ == 1)
        if len(positive):
            values.append(estimator.predict_proba(x)[:, int(positive[0])])
        else:
            values.append(np.zeros(len(x), dtype=np.float64))
    return np.stack(values, axis=1)


def quality_selection_key(name: str, ablations: dict) -> tuple:
    values = ablations[name]; spearman = values["validation"]["spearman"]
    return (-(spearman if spearman is not None else -np.inf), values["validation"]["mae"], name)


def ranking_selection_key(name: str, candidates: dict) -> tuple:
    values = candidates[name]["validation_ranking"]
    return (
        values["mean_regret"],
        -values["quality_capture"],
        -values["top1_tie_aware"],
        name,
    )


def validate_v2_output_paths(checkpoint: Path, output: Path, scoring_version: str) -> None:
    if scoring_version != "ready3d-v2": return
    legacy_checkpoint = Path("/root/r3dguard/checkpoints/ready3d/ready3d.joblib").resolve()
    legacy_output = Path("/root/r3dguard/evaluation/ready3d").resolve()
    if checkpoint.resolve() == legacy_checkpoint or output.resolve() == legacy_output:
        raise ValueError("V2 checkpoint and metrics must not overwrite V1 legacy paths")


def save_figures(output: Path, truth: np.ndarray, prediction: np.ndarray, ablations: dict, groups: list[dict]) -> None:
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.2, 4.2)); plt.scatter(truth, prediction, c="#167d8d")
    limits = [min(truth.min(), prediction.min()), max(truth.max(), prediction.max())]
    plt.plot(limits, limits, "--", color="#555555"); plt.xlabel("Automatic quality proxy"); plt.ylabel("Predicted proxy"); plt.tight_layout(); plt.savefig(figures / "quality_scatter.png", dpi=180); plt.close()
    names = list(ablations); values = [ablations[name]["test"]["spearman"] or 0.0 for name in names]
    plt.figure(figsize=(5.8, 4.0)); plt.bar(names, values, color=["#167d8d", "#d26a3a", "#58636e"]); plt.ylabel("Test Spearman"); plt.tight_layout(); plt.savefig(figures / "ablation_spearman.png", dpi=180); plt.close()
    plt.figure(figsize=(6.2, 3.8)); plt.bar([row["group_id"] for row in groups], [row["regret"] for row in groups], color="#d26a3a"); plt.ylabel("Best-of-4 regret"); plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.savefig(figures / "best_of_four_regret.png", dpi=180); plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True); parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--calibration", type=Path)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--allow-unsupported-failures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); data = load_dataset(args.labels, args.features, args.embeddings)
    validate_v2_output_paths(args.checkpoint, args.output_dir, data["scoring_version"])
    rows, y, failures = data["rows"], data["quality"], data["failure"]
    train, validation, test = (split_mask(rows, split) for split in ("train", "validation", "test"))
    split_masks = {"train": train, "validation": validation, "test": test}
    if not args.allow_unsupported_failures:
        validate_failure_support(failures, split_masks)
    matrices = {"engineered": data["scalar"], "dino": data["dino"], "full": np.concatenate([data["dino"], data["scalar"]], axis=1)}
    groups = [row["group_id"] for row in rows]
    ablations, quality_models, ranking_models, ranking_candidates = {}, {}, {}, {}
    for name, matrix in matrices.items():
        quality_model = make_regressor(args.seed)
        quality_model.fit(matrix[train], y[train])
        quality_models[name] = quality_model
        ablations[name] = {
            "validation": evaluate_regression(y[validation], quality_model.predict(matrix[validation])),
        }
        pair_x, pair_y, pair_weight = build_pairwise_data(matrix, y, groups, train)
        for ranker_name in RANKER_NAMES:
            candidate_name = f"{name}/{ranker_name}"
            ranker = make_ranker(ranker_name, args.seed)
            fit_ranker(ranker, ranker_name, pair_x, pair_y, pair_weight)
            validation_ranking, _ = best_of_four_metrics(rows, y, matrix, validation, ranker)
            ranking_models[candidate_name] = ranker
            ranking_candidates[candidate_name] = {
                "feature_set": name,
                "ranker": ranker_name,
                "validation_ranking": validation_ranking,
                "training_pairs": int(len(pair_y)),
                "weight_min": float(pair_weight.min()),
                "weight_max": float(pair_weight.max()),
                "weight_mean": float(pair_weight.mean()),
            }
        feature_candidates = [key for key in ranking_candidates if key.startswith(f"{name}/")]
        best_feature_candidate = min(
            feature_candidates,
            key=lambda key: ranking_selection_key(key, ranking_candidates),
        )
        ablations[name]["validation_ranking"] = ranking_candidates[best_feature_candidate]["validation_ranking"]
        ablations[name]["selected_ranker"] = ranking_candidates[best_feature_candidate]["ranker"]
        ablations[name]["training_pairs"] = ranking_candidates[best_feature_candidate]["training_pairs"]

    selected_quality_name = min(
        FEATURE_SETS, key=lambda name: quality_selection_key(name, ablations)
    )
    selected_ranking_candidate = min(
        ranking_candidates,
        key=lambda name: ranking_selection_key(name, ranking_candidates),
    )
    selected_ranking_name = ranking_candidates[selected_ranking_candidate]["ranker"]
    selected_ranking_features = ranking_candidates[selected_ranking_candidate]["feature_set"]
    for name, matrix in matrices.items():
        ablations[name]["test"] = evaluate_regression(y[test], quality_models[name].predict(matrix[test]))
        feature_candidate = f"{name}/{ablations[name]['selected_ranker']}"
        ablations[name]["test_ranking"], _ = best_of_four_metrics(
            rows, y, matrix, test, ranking_models[feature_candidate]
        )
    quality_x = matrices[selected_quality_name]
    ranking_x = matrices[selected_ranking_features]
    quality_model = quality_models[selected_quality_name]
    ranker = ranking_models[selected_ranking_candidate]
    validation_prediction = quality_model.predict(quality_x[validation])
    test_prediction = quality_model.predict(quality_x[test])
    best_of_four, group_records = best_of_four_metrics(rows, y, ranking_x, test, ranker)
    pair_x, pair_y, pair_weight = build_pairwise_data(ranking_x, y, groups, train)
    best_of_four.update({
        "training_pairs": int(len(pair_y)),
        "training_positive_pairs": int(pair_y.sum()),
        "training_negative_pairs": int(len(pair_y) - pair_y.sum()),
        "weight_min": float(pair_weight.min()),
        "weight_max": float(pair_weight.max()),
        "weight_mean": float(pair_weight.mean()),
    })
    random_baseline = random_selection_metrics(rows, y, test)
    best_of_four["random_baseline"] = random_baseline
    best_of_four["relative_to_random"] = {
        "top1_absolute_gain": best_of_four["top1_tie_aware"] - random_baseline["top1_tie_aware"],
        "regret_reduction": random_baseline["mean_regret"] - best_of_four["mean_regret"],
        "quality_capture_gain": best_of_four["quality_capture"] - random_baseline["quality_capture"],
    }
    conformal_q = conformal_quantile(y[validation] - validation_prediction)
    uncertainty = regressor_uncertainty(quality_model, quality_x[test]); absolute_error = np.abs(y[test] - test_prediction)
    uncertainty_correlation = spearmanr(uncertainty, absolute_error).statistic

    full_x = matrices["full"]
    failure_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1))
    failure_model.fit(full_x[train], failures[train])
    validation_probability = failure_probabilities(failure_model, full_x[validation])
    test_probability = failure_probabilities(failure_model, full_x[test])
    thresholds, per_type = {}, {}
    failure_prediction = np.zeros_like(failures[test])
    for index, name in enumerate(FAILURE_TYPES):
        unsupported = any(
            int(failures[mask, index].sum()) in {0, int(mask.sum())}
            for mask in split_masks.values()
        )
        threshold, validation_f1 = select_failure_threshold(failures[validation, index], validation_probability[:, index])
        prediction = test_probability[:, index] >= threshold; failure_prediction[:, index] = prediction
        test_f1 = float(f1_score(failures[test, index], prediction, zero_division=0)); thresholds[name] = threshold
        per_type[name] = {
            "threshold": threshold, "validation_f1": validation_f1, "test_f1": test_f1,
            "validation_pr_auc": float(average_precision_score(failures[validation, index], validation_probability[:, index])),
            "test_pr_auc": float(average_precision_score(failures[test, index], test_probability[:, index])),
            "experimental": unsupported or test_f1 == 0.0,
            "unsupported_in_at_least_one_split": unsupported,
        }
    failure_support = {name: {split_name: {"positive": int(failures[mask, index].sum()), "samples": int(mask.sum())} for split_name, mask in (("train", train), ("validation", validation), ("test", test))} for index, name in enumerate(FAILURE_TYPES)}
    failure_metrics = {
        "macro_f1": float(f1_score(failures[test], failure_prediction, average="macro", zero_division=0)),
        "exact_match_accuracy": float(accuracy_score(failures[test], failure_prediction)),
        "per_type": per_type, "per_type_f1": {name: row["test_f1"] for name, row in per_type.items()},
        "thresholds": thresholds, "support": failure_support,
    }
    domain_metrics = {}
    for domain in sorted({row["domain"] for row in rows}):
        domain_mask = test & np.asarray([row["domain"] == domain for row in rows])
        if domain_mask.any():
            ranking, _ = best_of_four_metrics(rows, y, ranking_x, domain_mask, ranker)
            domain_metrics[domain] = {
                "quality": evaluate_regression(
                    y[domain_mask], quality_model.predict(quality_x[domain_mask])
                ),
                "ranking": ranking,
                "test_samples": int(domain_mask.sum()),
                "test_groups": len({rows[index]["group_id"] for index in np.flatnonzero(domain_mask)}),
            }
    high_threshold = 0.95
    if args.calibration:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8")); high_threshold = float(calibration["high_quality"]["threshold"])
    elif data["scoring_version"] == "ready3d-v2":
        raise ValueError("V2 training requires the frozen train/validation calibration file")
    sample_counts = {name: int(mask.sum()) for name, mask in (("train", train), ("validation", validation), ("test", test))}
    group_counts = {name: len({rows[index]["group_id"] for index in np.flatnonzero(mask)}) for name, mask in (("train", train), ("validation", validation), ("test", test))}
    test_coverage = float(np.mean(absolute_error <= conformal_q))
    metrics = {
        "schema_version": "r3dguard.ready3d-metrics.v3", "scoring_version": data["scoring_version"],
        "selected_quality_features": selected_quality_name,
        "selected_ranking_features": selected_ranking_features,
        "selected_ranking_model": selected_ranking_name,
        "model_selection": {
            "split": "validation",
            "quality_priority": ["spearman_max", "mae_min"],
            "ranking_priority": ["mean_regret_min", "quality_capture_max", "top1_max"],
            "test_used_for_selection": False,
        },
        "sample_counts": sample_counts, "group_counts": group_counts,
        "quality": {"validation": evaluate_regression(y[validation], validation_prediction), "test": evaluate_regression(y[test], test_prediction)},
        "high_quality_accuracy": float(accuracy_score(data["high_quality"][test], test_prediction >= high_threshold)),
        "failure": failure_metrics, "ranking": best_of_four, "by_domain": domain_metrics,
        "uncertainty": {
            "method": "validation_split_conformal_plus_extra_trees_dispersion", "nominal_coverage": 0.90,
            "calibration_split": "validation", "test_used_for_calibration": False,
            "conformal_half_width": conformal_q, "test_coverage": test_coverage,
            "mean_interval_width": 2.0 * conformal_q, "mean_tree_std": float(uncertainty.mean()),
            "spearman_uncertainty_vs_absolute_error": float(uncertainty_correlation) if np.isfinite(uncertainty_correlation) else None,
        },
        "ablations": ablations,
        "ranking_candidates_validation": ranking_candidates,
        "target_scope": "automatic quality proxy; no semantic, aesthetic, hidden-surface, or human perceptual ground truth",
    }
    checkpoint = {
        "schema_version": "r3dguard.ready3d-checkpoint.v3", "scoring_version": data["scoring_version"],
        "quality_model": quality_model, "quality_feature_set": selected_quality_name,
        "failure_model": failure_model, "failure_thresholds": thresholds, "ranking_model": ranker,
        "ranking_feature_set": selected_ranking_features,
        "ranking_model_name": selected_ranking_name,
        "ranking_weighting": {
            "method": "absolute_quality_gap_normalized_mean_clipped",
            "minimum": 0.25,
            "maximum": 4.0,
        },
        "failure_experimental": {name: row["experimental"] for name, row in per_type.items()},
        "scalar_feature_names": data["scalar_names"], "failure_types": FAILURE_TYPES,
        "conformal_half_width": conformal_q, "high_quality_threshold": high_threshold,
        "feature_version": 1,
        "model_selection_priority": metrics["model_selection"],
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True); joblib.dump(checkpoint, args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "best_of_four.jsonl").open("w", encoding="utf-8") as handle:
        for record in group_records: handle.write(json.dumps(record) + "\n")
    test_indices = np.flatnonzero(test); predictions = []
    for local, index in enumerate(test_indices):
        predictions.append({"sample_id": rows[index]["sample_id"], "group_id": rows[index]["group_id"], "quality_true": float(y[index]), "quality_predicted": float(test_prediction[local]), "uncertainty_tree_std": float(uncertainty[local]), "interval_low": float(test_prediction[local] - conformal_q), "interval_high": float(test_prediction[local] + conformal_q)})
    with (args.output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in predictions: handle.write(json.dumps(record) + "\n")
    save_figures(args.output_dir, y[test], test_prediction, ablations, group_records)
    report = [
        "# Ready3D V3 evaluation", "",
        f"Selected quality features `{selected_quality_name}` by validation Spearman.",
        f"Selected ranking `{selected_ranking_features}/{selected_ranking_name}` by validation regret.",
        f"Test Spearman: {metrics['quality']['test']['spearman']:.4f}.",
        f"Best-of-4 Top-1: {best_of_four['top1_tie_aware']:.2%}; mean regret: {best_of_four['mean_regret']:.4f}; capture: {best_of_four['quality_capture']:.2%}.",
        f"Failure Macro-F1: {failure_metrics['macro_f1']:.4f}.",
        f"90% conformal test coverage: {test_coverage:.2%}; mean interval width: {2 * conformal_q:.4f}.",
        "", "All labels are automatic quality proxies and do not replace semantic, aesthetic, or human perceptual ground truth.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
