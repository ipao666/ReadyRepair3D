#!/usr/bin/env python3
"""Join scored Hunyuan batches with image features into Ready3D V2 targets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


FAILURE_TYPES = (
    "silhouette_truncation",
    "depth_ambiguity",
    "background_leakage",
    "texture_lighting_conflict",
    "view_conflict",
    "insufficient_detail",
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sample_id(row: dict) -> str:
    value = row.get("sample_id")
    if value:
        return str(value)
    filename = row.get("filename")
    if filename:
        return Path(filename).stem
    source = row.get("source_path") or row.get("path")
    if source:
        return Path(source).stem
    raise ValueError("manifest row has no sample identifier")


def infer_failure_types(diagnostic: dict, feature: dict) -> dict[str, bool]:
    border_contact = float(feature.get("occlusion_border_contact_ratio", 0.0))
    bbox_margin = float(feature.get("occlusion_bbox_margin_ratio", 1.0))
    mask_components = float(feature.get("mask_component_count", 1.0))
    largest_mask = float(feature.get("mask_largest_component_ratio", 1.0))
    depth_std = float(feature.get("depth_std", 1.0))
    normal_std = float(feature.get("normal_z_std", 1.0))
    clipped = float(feature.get("image_dark_clip_ratio", 0.0)) + float(
        feature.get("image_light_clip_ratio", 0.0)
    )
    entropy = float(feature.get("image_entropy", 1.0))
    sharpness = float(feature.get("image_sharpness", 10.0))
    connected = int(
        diagnostic.get(
            "component_count_raw",
            diagnostic.get("connected_components", 1),
        )
    )
    largest_geometry = float(
        diagnostic.get("largest_component_face_ratio", 1.0)
    )
    texture_missing = 1.0 - float(
        diagnostic.get("textured_surface_ratio", 1.0)
    )
    return {
        "silhouette_truncation": border_contact > 0.02 or bbox_margin < 0.05,
        "depth_ambiguity": depth_std < 0.08 or normal_std < 0.002,
        "background_leakage": mask_components > 2 or largest_mask < 0.95,
        "texture_lighting_conflict": clipped > 0.08 or texture_missing > 0.0,
        "view_conflict": connected > 4 or largest_geometry < 0.98,
        "insufficient_detail": entropy < 0.42 or sharpness < 2.5,
    }


def build_targets(
    split_inputs: dict[str, tuple[list[dict], list[dict]]],
    features: list[dict],
) -> list[dict]:
    feature_by_id = {sample_id(row): row for row in features}
    if len(feature_by_id) != len(features):
        raise ValueError("duplicate feature sample IDs")
    targets = []
    seen: set[str] = set()
    for split, (manifest, labels) in split_inputs.items():
        manifest_by_id = {sample_id(row): row for row in manifest}
        label_by_id = {sample_id(row): row for row in labels}
        if set(manifest_by_id) != set(label_by_id):
            missing_labels = sorted(set(manifest_by_id) - set(label_by_id))
            extra_labels = sorted(set(label_by_id) - set(manifest_by_id))
            raise ValueError(
                f"{split} manifest/label mismatch: "
                f"missing={missing_labels[:4]} extra={extra_labels[:4]}"
            )
        for current_id in sorted(manifest_by_id):
            if current_id in seen:
                raise ValueError(f"sample appears in multiple splits: {current_id}")
            seen.add(current_id)
            source = manifest_by_id[current_id]
            label = label_by_id[current_id]
            if label.get("scoring_version") != "ready3d-v2":
                raise ValueError(
                    f"non-canonical scoring_version for {current_id}: "
                    f"{label.get('scoring_version')}"
                )
            if label.get("calibration_source") != "independent_validation_64":
                raise ValueError(
                    f"unfrozen calibration source for {current_id}: "
                    f"{label.get('calibration_source')}"
                )
            feature = feature_by_id.get(current_id)
            if feature is None:
                raise ValueError(f"missing image features for {current_id}")
            technically_valid = bool(label["technically_valid"])
            high_quality = bool(label["high_quality"])
            targets.append(
                {
                    "schema_version": "r3dguard.ready3d-target.v2",
                    "scoring_version": "ready3d-v2",
                    "calibration_version": "ready3d-v2-independent-validation64",
                    "sample_id": current_id,
                    "group_id": source["group_id"],
                    "candidate_index": int(source["candidate_index"]),
                    "domain": source.get("domain", "ai_sana"),
                    "split": split,
                    "source_path": source.get("source_path") or source["path"],
                    "selected_backend": "hunyuan",
                    "selected_glb_path": label["glb_path"],
                    "technically_valid": technically_valid,
                    "quality_score_v2": float(label["quality_score_v2"]),
                    "high_quality": high_quality,
                    "qualified": bool(technically_valid and high_quality),
                    "input_match": float(label["input_match"]),
                    "geometry": float(label["geometry"]),
                    "texture_available": bool(label["texture_available"]),
                    "best_view_index": int(label["best_view_index"]),
                    "view_dino_similarities": label["view_dino_similarities"],
                    "failure_types": infer_failure_types(label, feature),
                    "technical_failure_reasons": label[
                        "technical_failure_reasons"
                    ],
                    "automatic_quality_proxy": True,
                    "semantic_fidelity_covered": False,
                    "human_aesthetic_quality_covered": False,
                }
            )
    validate_targets(targets)
    return targets


def validate_targets(rows: list[dict]) -> None:
    groups: dict[str, set[str]] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        group = str(row["group_id"])
        groups.setdefault(group, set()).add(str(row["split"]))
        counts[group] += 1
    leaked = {group: splits for group, splits in groups.items() if len(splits) != 1}
    incomplete = {group: count for group, count in counts.items() if count != 4}
    if leaked:
        raise ValueError(f"group leakage: {leaked}")
    if incomplete:
        raise ValueError(f"incomplete Best-of-4 groups: {incomplete}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--validation-labels", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument("--calibration-in", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_calibration = json.loads(
        args.calibration_in.read_text(encoding="utf-8")
    )
    required_metrics = {
        "dino_similarity",
        "silhouette_iou",
        "contour_similarity",
        "largest_component_area_ratio",
        "debris_area_ratio",
        "boundary_edge_length_ratio",
        "non_manifold_edge_length_ratio",
    }
    if source_calibration.get("scoring_version") != "ready3d-v2":
        raise ValueError("calibration is not canonical Ready3D V2")
    if source_calibration.get("source") != "independent_validation_64":
        raise ValueError("calibration is not from independent_validation_64")
    if set(source_calibration.get("metrics", {})) != required_metrics:
        raise ValueError("calibration metric set does not match Ready3D V2")

    targets = build_targets(
        {
            "train": (
                read_jsonl(args.train_manifest),
                read_jsonl(args.train_labels),
            ),
            "validation": (
                read_jsonl(args.validation_manifest),
                read_jsonl(args.validation_labels),
            ),
            "test": (
                read_jsonl(args.test_manifest),
                read_jsonl(args.test_labels),
            ),
        },
        read_jsonl(args.features),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "targets.jsonl", targets)
    calibration = {
        "schema_version": "r3dguard.ready3d-calibration.v2",
        "calibration_version": "ready3d-v2-independent-validation64",
        "normalization": source_calibration["metrics"],
        "high_quality": {
            "threshold": float(source_calibration["quality_threshold"]),
            "threshold_percentile": int(
                source_calibration["quality_threshold_percentile"]
            ),
            "threshold_source_split": "validation",
        },
        "test_used_for_fitting": False,
        "source": source_calibration["source"],
    }
    write_json(args.output_dir / "calibration.json", calibration)
    support = {
        name: {
            split: sum(
                row["failure_types"][name]
                for row in targets
                if row["split"] == split
            )
            for split in ("train", "validation", "test")
        }
        for name in FAILURE_TYPES
    }
    summary = {
        "targets": len(targets),
        "groups": len({row["group_id"] for row in targets}),
        "splits": Counter(row["split"] for row in targets),
        "failure_positive_support": support,
        "automatic_quality_proxy": True,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
