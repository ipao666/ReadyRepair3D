#!/usr/bin/env python3
"""Predict exactly one Ready3D V3 candidate for one Hunyuan3D call."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ops.extract_ready3d_features import (  # noqa: E402
    FeatureModels,
    depth_normal_features,
    image_features,
    mask_occlusion_features,
)
from ops.predict_ready3d import assemble_matrices, default_models_root  # noqa: E402
from r3dloop.ready3d_v3.ranking import (  # noqa: E402
    add_group_context,
    combine_group_scores,
    pairwise_group_scores,
    select_top1,
    validate_complete_groups,
)


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


def build_top1_output(
    rows: list[dict],
    scalar_records: list[dict],
    predicted_quality: np.ndarray,
    pairwise_scores: np.ndarray,
    weights: dict[str, float],
    resample_threshold: float,
) -> dict:
    validate_complete_groups(rows)
    if len({row["group_id"] for row in rows}) != 1:
        raise ValueError("one prediction call must contain exactly one candidate group")
    if len(scalar_records) != len(rows):
        raise ValueError("scalar records must align with candidates")
    quality = np.asarray(predicted_quality, dtype=np.float64)
    pairwise = np.asarray(pairwise_scores, dtype=np.float64)
    if quality.shape != (len(rows),) or pairwise.shape != (len(rows),):
        raise ValueError("prediction arrays must align with candidates")

    flags = [severe_input_flags(row) for row in scalar_records]
    structural_pass = np.asarray([not row for row in flags], dtype=bool)
    scores = combine_group_scores(rows, pairwise, quality, structural_pass, weights)
    selection = select_top1(rows, scores, structural_pass)[0]
    selected_id = selection["selected_sample_id"]
    selected_index = next(
        index for index, row in enumerate(rows) if row["sample_id"] == selected_id
    )
    if not structural_pass.any():
        resample = True
        reason = "all_candidates_failed_structure_gate"
    elif float(quality[selected_index]) < float(resample_threshold):
        resample = True
        reason = "selected_predicted_quality_below_validation_threshold"
    else:
        resample = False
        reason = None

    candidates = []
    for index, row in enumerate(rows):
        candidates.append(
            {
                **row,
                "predicted_quality": float(np.clip(quality[index], 0.0, 1.0)),
                "raw_predicted_quality": float(quality[index]),
                "pairwise_score": float(pairwise[index]),
                "ensemble_score": float(scores[index]),
                "structural_input_pass": bool(structural_pass[index]),
                "severe_input_flags": flags[index],
                "selected_for_3d": row["sample_id"] == selected_id,
            }
        )
    return {
        "schema_version": "r3dguard.ready3d-v3-top1-prediction.v1",
        "group_id": str(rows[0]["group_id"]),
        "selection_policy": "structure_gate_then_groupwise_ensemble_top1",
        "selected_sample_id": selected_id,
        "selected_candidate_index": selection["selected_candidate_index"],
        "selected_predicted_quality": float(quality[selected_index]),
        "selection_fallback": selection["selection_fallback"],
        "planned_3d_calls": 1,
        "top2_fallback_enabled": False,
        "resample_2d_recommended": resample,
        "resample_reason": reason,
        "resample_policy": "generate_four_new_2d_candidates_before_any_3d_call",
        "resample_threshold": float(resample_threshold),
        "candidates": candidates,
    }


def write_prediction_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


class Ready3DV3Predictor:
    def __init__(self, checkpoint_path: Path, device: str = "cuda"):
        self.checkpoint = joblib.load(checkpoint_path)
        if self.checkpoint.get("schema_version") != "r3dguard.ready3d-v3-top1-checkpoint.v1":
            raise ValueError("not a Ready3D V3 Top-1 checkpoint")
        models_root = default_models_root()
        self.models = FeatureModels(
            models_root / "dinov2-large",
            models_root / "Depth-Anything-V2-Large-hf",
            models_root / "BiRefNet",
            device,
        )

    def predict(self, images: list[Image.Image], rows: list[dict]) -> dict:
        if len(images) != 4 or len(rows) != 4:
            raise ValueError("Ready3D V3 Top-1 requires exactly four candidates")
        embeddings, scalars = [], []
        for image in images:
            embedding, depth, mask = self.models.extract(image.convert("RGB"))
            record = image_features(image)
            record.update(mask_occlusion_features(mask))
            record.update(depth_normal_features(depth, mask >= 0.5))
            embeddings.append(embedding)
            scalars.append(record)
        raw = assemble_matrices(
            scalars,
            np.stack(embeddings),
            self.checkpoint["scalar_feature_names"],
        )
        group_ids = [row["group_id"] for row in rows]
        matrices = {
            name: add_group_context(matrix, group_ids) for name, matrix in raw.items()
        }
        quality_feature_set = self.checkpoint["quality_feature_set"]
        ranking_feature_set = self.checkpoint["ranking_feature_set"]
        quality = self.checkpoint["quality_model"].predict(matrices[quality_feature_set])
        pairwise = pairwise_group_scores(
            self.checkpoint["ranking_model"], matrices[ranking_feature_set], rows
        )
        return build_top1_output(
            rows,
            scalars,
            quality,
            pairwise,
            self.checkpoint["ensemble_weights"],
            self.checkpoint["resample_suggestion_threshold"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path, nargs=4)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = []
    for path in args.images:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    rows = [
        {
            "group_id": args.group_id,
            "sample_id": f"{args.group_id}_c{index}",
            "candidate_index": index,
            "image_path": str(path),
        }
        for index, path in enumerate(args.images)
    ]
    output = Ready3DV3Predictor(args.checkpoint, args.device).predict(images, rows)
    write_prediction_atomic(args.output, output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
