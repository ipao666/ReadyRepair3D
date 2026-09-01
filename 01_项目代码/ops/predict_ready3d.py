#!/usr/bin/env python3
"""Ready3D inference over one or more candidate images."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

OPS_DIR = Path(__file__).resolve().parent
ROOT = OPS_DIR.parent
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

from extract_ready3d_features import (
    FeatureModels, depth_normal_features, image_features, mask_occlusion_features,
)
from train_ready3d import failure_probabilities, rank_scores, regressor_uncertainty


def default_models_root() -> Path:
    models = os.environ.get("R3DGUARD_MODELS")
    if models:
        return Path(models)
    home = os.environ.get("R3DGUARD_HOME")
    if home:
        return Path(home) / "models"
    return ROOT / "models"


def default_checkpoint_path() -> Path:
    home = os.environ.get("R3DGUARD_HOME")
    root = Path(home) if home else ROOT
    return root / "checkpoints/ready3d_v2/ready3d_v2.joblib"


SCALAR_PREFIXES = ("image_", "mask_", "depth_", "normal_", "occlusion_")


def severe_input_flags(row: dict) -> list[str]:
    flags = []
    area = float(row["mask_area_ratio"])
    if area < 0.02:
        flags.append("tiny_foreground")
    if area > 0.92:
        flags.append("foreground_fills_frame")
    if float(row["occlusion_border_contact_ratio"]) > 0.10:
        flags.append("border_contact")
    if float(row["mask_largest_component_ratio"]) < 0.75:
        flags.append("fragmented_foreground")
    return flags


def apply_quality_first_selection(
    predictions: list[dict],
    scalar_records: list[dict],
    top_k: int = 1,
) -> list[dict]:
    if len(predictions) != len(scalar_records) or not predictions:
        raise ValueError("predictions and scalar records must be non-empty and aligned")
    if not 1 <= top_k <= len(predictions):
        raise ValueError(
            f"top_k must be between 1 and {len(predictions)}, got {top_k}"
        )

    output = []
    eligible = []
    rejected = []
    for index, (prediction, scalars) in enumerate(
        zip(predictions, scalar_records, strict=True)
    ):
        flags = severe_input_flags(scalars)
        output.append(
            {
                **prediction,
                "selected": False,
                "selection_rank": None,
                "severe_input_flags": flags,
                "structural_input_pass": not flags,
                "selection_fallback": None,
            }
        )
        (eligible if not flags else rejected).append(index)

    def sort_key(index: int) -> tuple[float, int]:
        candidate_index = int(output[index].get("candidate_index", index))
        return (-float(output[index]["predicted_quality"]), candidate_index)

    ranked_eligible = sorted(eligible, key=sort_key)
    ranked_rejected = sorted(rejected, key=sort_key)
    chosen = ranked_eligible[:top_k]
    chosen.extend(ranked_rejected[: top_k - len(chosen)])
    rule = (
        "structural_gate_then_predicted_quality"
        if top_k == 1
        else f"structural_gate_then_predicted_quality_top{top_k}"
    )
    for rank, index in enumerate(chosen, 1):
        output[index]["selected"] = True
        output[index]["selection_rank"] = rank
        output[index]["selection_rule"] = rule
        if index in rejected:
            output[index]["selection_fallback"] = (
                "all_candidates_failed_structure_gate"
                if not eligible
                else "insufficient_structural_pass_candidates"
            )
    return output


def assemble_matrices(
    scalar_records: list[dict], embeddings: np.ndarray, scalar_names: list[str]
) -> dict[str, np.ndarray]:
    missing = sorted({name for row in scalar_records for name in scalar_names if name not in row})
    if missing:
        raise ValueError(f"missing scalar features: {missing}")
    scalar = np.asarray([[float(row[name]) for name in scalar_names] for row in scalar_records], dtype=np.float32)
    dino = np.asarray(embeddings, dtype=np.float32)
    if len(scalar) != len(dino):
        raise ValueError("scalar and DINO feature counts differ")
    return {"engineered": scalar, "dino": dino, "full": np.concatenate([dino, scalar], axis=1)}


def predict_arrays(checkpoint: dict, matrices: dict[str, np.ndarray]) -> list[dict]:
    feature_set = checkpoint["quality_feature_set"]
    if feature_set not in matrices:
        raise ValueError(f"unknown quality feature set: {feature_set}")
    ranking_feature_set = checkpoint.get("ranking_feature_set", feature_set)
    if ranking_feature_set not in matrices:
        raise ValueError(f"unknown ranking feature set: {ranking_feature_set}")
    quality_model = checkpoint["quality_model"]
    quality = quality_model.predict(matrices[feature_set])
    uncertainty = regressor_uncertainty(quality_model, matrices[feature_set])
    failure_probability = failure_probabilities(checkpoint["failure_model"], matrices["full"])
    thresholds = checkpoint.get("failure_thresholds", {name: 0.5 for name in checkpoint["failure_types"]})
    ranking = (
        rank_scores(checkpoint["ranking_model"], matrices[ranking_feature_set])
        if len(quality) > 1 else np.ones(1)
    )
    selected = int(np.argmax(ranking))
    half_width = float(checkpoint["conformal_half_width"])
    high_quality_threshold = float(checkpoint.get("high_quality_threshold", 0.95))
    scoring_version = checkpoint.get("scoring_version", "v1_legacy")
    return [
        {
            "candidate_index": index,
            "predicted_quality": float(np.clip(value, 0.0, 1.0)),
            "raw_predicted_quality": float(value),
            "uncertainty_tree_std": float(uncertainty[index]),
            "interval_low": float(np.clip(value - half_width, 0.0, 1.0)),
            "interval_high": float(np.clip(value + half_width, 0.0, 1.0)),
            "ranking_score": float(ranking[index]),
            "selected": index == selected,
            "scoring_version": scoring_version,
            "high_quality": bool(value >= high_quality_threshold),
            "high_quality_threshold": high_quality_threshold,
            "predicted_failure_types": [
                name for probability, name in zip(failure_probability[index], checkpoint["failure_types"], strict=True)
                if probability >= thresholds[name]
            ],
            "failure_predictions": [
                {
                    "type": name, "probability": float(probability),
                    "threshold": float(thresholds[name]),
                    "predicted": bool(probability >= thresholds[name]),
                    "experimental": bool(checkpoint.get("failure_experimental", {}).get(name, False)),
                }
                for probability, name in zip(failure_probability[index], checkpoint["failure_types"], strict=True)
            ],
        }
        for index, value in enumerate(quality)
    ]


class Ready3DPredictor:
    def __init__(self, checkpoint_path: Path, device: str = "cuda"):
        self.checkpoint = joblib.load(checkpoint_path)
        models_root = default_models_root()
        self.models = FeatureModels(
            models_root / "dinov2-large",
            models_root / "Depth-Anything-V2-Large-hf",
            models_root / "BiRefNet",
            device,
        )

    def predict(self, images: list[Image.Image]) -> list[dict]:
        embeddings, scalars = [], []
        for image in images:
            embedding, depth, mask = self.models.extract(image.convert("RGB"))
            record = image_features(image)
            record.update(mask_occlusion_features(mask))
            record.update(depth_normal_features(depth, mask >= 0.5))
            embeddings.append(embedding)
            scalars.append(record)
        matrices = assemble_matrices(
            scalars, np.stack(embeddings), self.checkpoint["scalar_feature_names"]
        )
        return predict_arrays(self.checkpoint, matrices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint_path())
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= len(args.images) <= 4:
        raise ValueError("Ready3D accepts one to four candidate images")
    images = []
    for path in args.images:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    predictor = Ready3DPredictor(args.checkpoint, args.device)
    print(json.dumps(predictor.predict(images), indent=2))


if __name__ == "__main__":
    main()
