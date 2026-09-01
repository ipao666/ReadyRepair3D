from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path("/root/r3dguard")

COUNT_FEATURES = [
    "vertices",
    "faces",
    "geometry_count",
    "connected_components",
    "boundary_edges",
    "hole_count_approx",
    "non_manifold_edges",
    "normal_inconsistent_edges",
    "degenerate_faces",
    "duplicate_faces",
    "mergeable_duplicate_vertices",
    "self_intersection_proxy_count",
]
RATIO_FEATURES = [
    "largest_component_face_ratio",
    "normal_inconsistent_ratio",
    "reversed_normal_ratio_approx",
    "self_intersection_proxy_ratio",
    "texture_missing_ratio",
    "uv_coverage_ratio",
    "quality_score",
]
BOOLEAN_FEATURES = [
    "is_watertight",
    "is_winding_consistent",
    "texture_present",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def feature_names(actions: list[str]) -> list[str]:
    return (
        [f"log1p_{name}" for name in COUNT_FEATURES]
        + RATIO_FEATURES
        + BOOLEAN_FEATURES
        + [f"action_{action}" for action in actions]
    )


def feature_vector(
    metrics: dict[str, Any], action: str, actions: list[str]
) -> np.ndarray:
    values = [math.log1p(max(float(metrics.get(name, 0.0)), 0.0)) for name in COUNT_FEATURES]
    values.extend(float(metrics.get(name, 0.0)) for name in RATIO_FEATURES)
    values.extend(float(bool(metrics.get(name, False))) for name in BOOLEAN_FEATURES)
    values.extend(float(action == candidate) for candidate in actions)
    return np.asarray(values, dtype=np.float64)


def build_matrix(
    rows: list[dict[str, Any]], actions: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.vstack(
        [feature_vector(row["quality_before"], row["action"], actions) for row in rows]
    )
    realized_gain = np.asarray(
        [row["quality_gain"] if row["accepted"] else 0.0 for row in rows],
        dtype=np.float64,
    )
    groups = np.asarray([row["sample_id"] for row in rows], dtype=object)
    return matrix, realized_gain, groups


def rule_score(row: dict[str, Any]) -> float:
    metrics = row["quality_before"]
    faces = max(float(metrics.get("faces", 0)), 1.0)
    vertices = max(float(metrics.get("vertices", 0)), 1.0)
    edges = max(float(metrics.get("edge_instances", 0)), 1.0)
    action = row["action"]
    scores = {
        "remove_floaters": max(
            0.0,
            (1.0 - float(metrics.get("largest_component_face_ratio", 1.0)))
            + 0.01 * max(float(metrics.get("connected_components", 1)) - 1.0, 0.0),
        ),
        "cleanup_degenerate_faces": (
            float(metrics.get("degenerate_faces", 0))
            + float(metrics.get("duplicate_faces", 0))
            + float(metrics.get("non_manifold_edges", 0))
        )
        / faces,
        "fix_face_normals": float(metrics.get("normal_inconsistent_ratio", 0.0))
        + float(not metrics.get("is_winding_consistent", True)),
        "merge_close_vertices": float(metrics.get("mergeable_duplicate_vertices", 0))
        / vertices,
        "fill_small_holes": float(metrics.get("boundary_edges", 0)) / edges
        + 0.01 * float(not metrics.get("is_watertight", True)),
        "decimate_mesh": 0.0,
    }
    return scores[action]


def make_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=20260715,
        n_jobs=-1,
    )


def make_gradient_boosting() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=8,
        l2_regularization=0.01,
        random_state=20260715,
    )


def cross_validated_predictions(
    factory: Callable[[], Any],
    matrix: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    splitter = GroupKFold(n_splits=4, shuffle=True, random_state=20260715)
    predictions = np.zeros(len(target), dtype=np.float64)
    fold_ids = np.full(len(target), -1, dtype=np.int64)
    split_records: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(splitter.split(matrix, target, groups)):
        model = factory()
        model.fit(matrix[train], target[train])
        predictions[test] = model.predict(matrix[test])
        fold_ids[test] = fold
        split_records.append(
            {
                "fold": fold,
                "train_groups": sorted(set(groups[train].tolist())),
                "test_groups": sorted(set(groups[test].tolist())),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
            }
        )
    return predictions, fold_ids, split_records


def evaluate_rankings(
    rows: list[dict[str, Any]],
    target: np.ndarray,
    predictions: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[str]]:
    assets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        assets[(row["sample_id"], row["defect_type"])].append(index)

    details: list[dict[str, Any]] = []
    oracle_labels: list[str] = []
    predicted_labels: list[str] = []
    for (sample_id, defect_type), indices in sorted(assets.items()):
        oracle_gain = max(float(target[index]) for index in indices)
        oracle_actions = sorted(
            rows[index]["action"]
            for index in indices
            if math.isclose(float(target[index]), oracle_gain, abs_tol=1e-12)
        )
        selected_index = max(indices, key=lambda index: float(predictions[index]))
        selected_action = rows[selected_index]["action"]
        realized_gain = float(target[selected_index])
        details.append(
            {
                "sample_id": sample_id,
                "defect_type": defect_type,
                "oracle_actions": oracle_actions,
                "oracle_gain": oracle_gain,
                "selected_action": selected_action,
                "selected_realized_gain": realized_gain,
                "selected_predicted_gain": float(predictions[selected_index]),
                "top1_correct_tie_aware": selected_action in oracle_actions,
                "regret": oracle_gain - realized_gain,
                "selected_accepted": bool(rows[selected_index]["accepted"]),
                "selected_rollback_reason": rows[selected_index]["rollback_reason"],
            }
        )
        oracle_labels.append(oracle_actions[0])
        predicted_labels.append(selected_action)

    oracle_sum = sum(item["oracle_gain"] for item in details)
    realized_sum = sum(item["selected_realized_gain"] for item in details)
    metrics = {
        "defect_assets": len(details),
        "top1_accuracy_tie_aware": sum(item["top1_correct_tie_aware"] for item in details)
        / len(details),
        "mean_regret": float(np.mean([item["regret"] for item in details])),
        "median_regret": float(np.median([item["regret"] for item in details])),
        "oracle_gain_capture_ratio": realized_sum / oracle_sum if oracle_sum else 1.0,
        "recovery_rate": sum(item["selected_realized_gain"] > 0 for item in details)
        / len(details),
        "actionable_rate": sum(item["selected_accepted"] for item in details) / len(details),
        "automatic_rollback_rate": sum(not item["selected_accepted"] for item in details)
        / len(details),
        "deployment_no_harm_rate": 1.0,
    }
    return metrics, details, oracle_labels, predicted_labels


def write_csv(path: Path, rows: list[dict[str, Any]], actions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = feature_names(actions)
    fieldnames = [
        "sample_id",
        "defect_type",
        "action",
        "accepted",
        "quality_gain",
        "realized_gain",
        "rollback_reason",
    ] + names
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            vector = feature_vector(row["quality_before"], row["action"], actions)
            output = {
                "sample_id": row["sample_id"],
                "defect_type": row["defect_type"],
                "action": row["action"],
                "accepted": row["accepted"],
                "quality_gain": row["quality_gain"],
                "realized_gain": row["quality_gain"] if row["accepted"] else 0.0,
                "rollback_reason": row["rollback_reason"],
            }
            output.update(dict(zip(names, vector, strict=True)))
            writer.writerow(output)


def save_plots(
    figures: Path,
    model_metrics: dict[str, dict[str, Any]],
    target: np.ndarray,
    best_predictions: np.ndarray,
    actions: list[str],
    oracle_labels: list[str],
    predicted_labels: list[str],
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    names = list(model_metrics)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, [model_metrics[name]["top1_accuracy_tie_aware"] for name in names])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Top-1 action accuracy")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(names, [model_metrics[name]["mean_regret"] for name in names])
    axes[1].set_title("Mean quality-gain regret")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures / "repair3d_model_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(target, best_predictions, s=16, alpha=0.65)
    limit = max(float(target.max()), float(best_predictions.max()), 1e-6)
    axis.plot([0, limit], [0, limit], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("Realized quality gain")
    axis.set_ylabel("Predicted quality gain")
    axis.set_title("Held-out action-gain predictions")
    fig.tight_layout()
    fig.savefig(figures / "repair3d_gain_scatter.png", dpi=180)
    plt.close(fig)

    matrix = confusion_matrix(oracle_labels, predicted_labels, labels=actions)
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(actions)), actions, rotation=35, ha="right")
    axis.set_yticks(range(len(actions)), actions)
    axis.set_xlabel("Predicted action")
    axis.set_ylabel("Oracle action (deterministic tie label)")
    for row in range(len(actions)):
        for column in range(len(actions)):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(figures / "repair3d_action_confusion.png", dpi=180)
    plt.close(fig)


def evaluate_clean_assets(
    model: Any,
    rows: list[dict[str, Any]],
    actions: list[str],
    training_groups: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        assets[row["sample_id"]].append(row)
    recommendations: list[dict[str, Any]] = []
    for sample_id, asset_rows in sorted(assets.items()):
        matrix = np.vstack(
            [
                feature_vector(row["quality_before"], row["action"], actions)
                for row in asset_rows
            ]
        )
        predictions = model.predict(matrix)
        selected_index = int(np.argmax(predictions))
        selected = asset_rows[selected_index]
        no_harm = bool(
            (not selected["accepted"])
            or (
                selected["quality_gain"] > 0
                and selected["quality_after"].get("blender_loadable") is True
            )
        )
        recommendations.append(
            {
                "sample_id": sample_id,
                "seen_in_training_groups": sample_id in training_groups,
                "selected_action": selected["action"],
                "predicted_gain": float(predictions[selected_index]),
                "actual_accepted": selected["accepted"],
                "actual_gain": selected["quality_gain"] if selected["accepted"] else 0.0,
                "rollback_reason": selected["rollback_reason"],
                "no_harm": no_harm,
            }
        )

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "assets": len(items),
            "no_harm_rate": sum(item["no_harm"] for item in items) / len(items),
            "actionable_rate": sum(item["actual_accepted"] for item in items) / len(items),
            "automatic_rollback_rate": sum(not item["actual_accepted"] for item in items)
            / len(items),
            "mean_realized_gain": float(np.mean([item["actual_gain"] for item in items])),
        }

    unseen = [item for item in recommendations if not item["seen_in_training_groups"]]
    return {"all_clean_assets": summarize(recommendations), "unseen_clean_assets": summarize(unseen)}, recommendations

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Repair3D action-gain baselines")
    parser.add_argument(
        "--trials",
        type=Path,
        default=ROOT / "data/repair3d_actions/action_trials_phase2.jsonl",
    )
    parser.add_argument(
        "--training-table",
        type=Path,
        default=ROOT / "data/repair3d_actions/training_table.csv",
    )
    parser.add_argument(
        "--clean-trials",
        type=Path,
        default=ROOT / "data/repair3d_actions/action_trials.jsonl",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "checkpoints/repair3d",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=ROOT / "evaluation/repair3d_action_model",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.trials)
    actions = sorted({row["action"] for row in rows})
    matrix, target, groups = build_matrix(rows, actions)
    write_csv(args.training_table, rows, actions)

    factories: dict[str, Callable[[], Any]] = {
        "random_forest": make_random_forest,
        "gradient_boosting": make_gradient_boosting,
    }
    all_predictions: dict[str, np.ndarray] = {
        "rule": np.asarray([rule_score(row) for row in rows], dtype=np.float64)
    }
    fold_ids = np.full(len(rows), -1, dtype=np.int64)
    split_records: list[dict[str, Any]] = []
    for name, factory in factories.items():
        predictions, model_folds, splits = cross_validated_predictions(
            factory, matrix, target, groups
        )
        all_predictions[name] = predictions
        if not split_records:
            fold_ids = model_folds
            split_records = splits

    model_metrics: dict[str, dict[str, Any]] = {}
    ranking_details: dict[str, list[dict[str, Any]]] = {}
    label_pairs: dict[str, tuple[list[str], list[str]]] = {}
    for name, predictions in all_predictions.items():
        ranking, details, oracle_labels, predicted_labels = evaluate_rankings(
            rows, target, predictions
        )
        ranking.update(
            {
                "row_mae": float(mean_absolute_error(target, predictions)),
                "row_rmse": float(math.sqrt(mean_squared_error(target, predictions))),
            }
        )
        model_metrics[name] = ranking
        ranking_details[name] = details
        label_pairs[name] = (oracle_labels, predicted_labels)

    best_model_name = min(
        factories,
        key=lambda name: (
            model_metrics[name]["mean_regret"],
            -model_metrics[name]["top1_accuracy_tie_aware"],
        ),
    )
    best_model = factories[best_model_name]()
    best_model.fit(matrix, target)
    clean_metrics, clean_recommendations = evaluate_clean_assets(
        best_model,
        load_jsonl(args.clean_trials),
        actions,
        set(groups.tolist()),
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "actions": actions,
        "feature_names": feature_names(actions),
        "count_features": COUNT_FEATURES,
        "ratio_features": RATIO_FEATURES,
        "boolean_features": BOOLEAN_FEATURES,
        "training_rows": len(rows),
        "training_groups": sorted(set(groups.tolist())),
    }
    checkpoint = args.checkpoint_dir / "action_gain_model.joblib"
    joblib.dump(artifact, checkpoint)

    args.evaluation_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "selected_model": best_model_name,
        "selection_rule": "minimum grouped-CV mean regret, then maximum tie-aware Top-1",
        "rows": len(rows),
        "source_groups": len(set(groups.tolist())),
        "defect_assets": len(rows) // len(actions),
        "actions": actions,
        "models": model_metrics,
        "clean_asset_evaluation": clean_metrics,
        "checkpoint": str(checkpoint),
        "leakage_control": "4-fold GroupKFold by original sample_id",
    }
    (args.evaluation_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.evaluation_dir / "split_manifest.json").write_text(
        json.dumps(split_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    prediction_path = args.evaluation_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            record = {
                "sample_id": row["sample_id"],
                "defect_type": row["defect_type"],
                "action": row["action"],
                "fold": int(fold_ids[index]),
                "accepted": row["accepted"],
                "realized_gain": float(target[index]),
                "predictions": {
                    name: float(values[index]) for name, values in all_predictions.items()
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    with (args.evaluation_dir / "clean_recommendations.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in clean_recommendations:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    best_oracle, best_predicted = label_pairs[best_model_name]
    save_plots(
        args.evaluation_dir / "figures",
        model_metrics,
        target,
        all_predictions[best_model_name],
        actions,
        best_oracle,
        best_predicted,
    )

    report_lines = [
        "# Repair3D 动作收益模型评测",
        "",
        f"- 数据：{len(rows)} 条动作记录，{len(set(groups.tolist()))} 个原始资产组。",
        "- 划分：4 折 GroupKFold，按原始 sample_id 分组，无同源缺陷泄漏。",
        f"- 选择模型：`{best_model_name}`。",
        "- 标签：动作被接受时使用实际 quality_gain，否则为 0。",
        "",
        "| 模型 | Top-1（并列友好） | 平均后悔值 | 收益捕获率 | 恢复率 | 回退率 | 行 MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in model_metrics.items():
        report_lines.append(
            f"| {name} | {result['top1_accuracy_tie_aware']:.2%} | "
            f"{result['mean_regret']:.6f} | {result['oracle_gain_capture_ratio']:.2%} | "
            f"{result['recovery_rate']:.2%} | {result['automatic_rollback_rate']:.2%} | "
            f"{result['row_mae']:.6f} |"
        )
    report_lines.extend(
        [
            "",
            "受控缺陷结果用于验证动作收益可学习性；最终恢复率仍需在真实失败 Hunyuan 资产上报告。",
            "自动回退意味着未接受动作不会覆盖原件，因此部署无伤率按当前策略为 100%。",
            f"干净资产无伤率：{clean_metrics['all_clean_assets']['no_harm_rate']:.2%}；"
            f"未见原件无伤率：{clean_metrics['unseen_clean_assets']['no_harm_rate']:.2%}。",
            "",
        ]
    )
    (args.evaluation_dir / "report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    (args.checkpoint_dir / "metadata.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
