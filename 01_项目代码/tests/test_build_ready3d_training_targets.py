from pathlib import Path
import sys

import pytest

OPS = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS))

from build_ready3d_training_targets import build_targets, infer_failure_types


def test_infer_failure_types_uses_image_and_mesh_signals():
    failures = infer_failure_types(
        {
            "component_count_raw": 6,
            "largest_component_face_ratio": 0.9,
            "textured_surface_ratio": 0.8,
        },
        {
            "occlusion_border_contact_ratio": 0.03,
            "occlusion_bbox_margin_ratio": 0.1,
            "mask_component_count": 1,
            "mask_largest_component_ratio": 1.0,
            "depth_std": 0.2,
            "normal_z_std": 0.01,
            "image_dark_clip_ratio": 0,
            "image_light_clip_ratio": 0,
            "image_entropy": 0.8,
            "image_sharpness": 5,
        },
    )

    assert failures["silhouette_truncation"] is True
    assert failures["view_conflict"] is True
    assert failures["background_leakage"] is False


def make_group(prefix: str, split: str):
    manifest, labels, features = [], [], []
    for index in range(4):
        current_id = f"{prefix}_c{index}"
        manifest.append(
            {
                "filename": f"{current_id}.png",
                "path": f"/tmp/{current_id}.png",
                "group_id": prefix,
                "candidate_index": index,
            }
        )
        labels.append(
            {
                "sample_id": current_id,
                "scoring_version": "ready3d-v2",
                "calibration_source": "independent_validation_64",
                "technically_valid": True,
                "high_quality": index == 0,
                "glb_path": f"/tmp/{current_id}.glb",
                "quality_score_v2": 0.8 - index * 0.1,
                "input_match": 0.7,
                "geometry": 0.9,
                "texture_available": True,
                "best_view_index": 0,
                "view_dino_similarities": [0.5] * 8,
                "technical_failure_reasons": [],
            }
        )
        features.append(
            {
                "sample_id": current_id,
                "depth_std": 0.2,
                "normal_z_std": 0.01,
            }
        )
    return split, (manifest, labels), features


def test_build_targets_preserves_complete_group_splits():
    split_inputs, features = {}, []
    for prefix, split in (
        ("train_group", "train"),
        ("validation_group", "validation"),
        ("test_group", "test"),
    ):
        current_split, pair, current_features = make_group(prefix, split)
        split_inputs[current_split] = pair
        features.extend(current_features)

    rows = build_targets(split_inputs, features)

    assert len(rows) == 12
    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    assert all(row["schema_version"] == "r3dguard.ready3d-target.v2" for row in rows)


def test_build_targets_rejects_manifest_label_mismatch():
    split, pair, features = make_group("train_group", "train")
    manifest, labels = pair
    with pytest.raises(ValueError, match="manifest/label mismatch"):
        build_targets({split: (manifest, labels[:-1])}, features)
