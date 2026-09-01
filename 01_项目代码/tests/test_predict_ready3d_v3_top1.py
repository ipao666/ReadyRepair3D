import json

import numpy as np

from ops.predict_ready3d_v3_top1 import build_top1_output, write_prediction_atomic


def _rows():
    return [
        {
            "group_id": "zh_001",
            "sample_id": f"zh_001_c{candidate}",
            "candidate_index": candidate,
            "image_path": f"images/c{candidate}.png",
        }
        for candidate in range(4)
    ]


def _healthy_scalars():
    return [
        {
            "mask_area_ratio": 0.5,
            "mask_largest_component_ratio": 0.95,
            "occlusion_border_contact_ratio": 0.01,
        }
        for _ in range(4)
    ]


def test_output_selects_exactly_one_candidate_for_one_glb_call():
    output = build_top1_output(
        _rows(),
        _healthy_scalars(),
        predicted_quality=np.asarray([0.9, 0.8, 0.7, 0.6]),
        pairwise_scores=np.asarray([0.1, 0.2, 0.95, 0.0]),
        weights={"pairwise": 0.8, "utility": 0.2},
        resample_threshold=0.5,
    )

    assert output["selected_sample_id"] == "zh_001_c2"
    assert output["planned_3d_calls"] == 1
    assert sum(row["selected_for_3d"] for row in output["candidates"]) == 1
    assert output["resample_2d_recommended"] is False


def test_all_structure_failures_select_one_but_recommend_only_2d_resampling():
    scalars = _healthy_scalars()
    for row in scalars:
        row["mask_area_ratio"] = 0.99
    output = build_top1_output(
        _rows(),
        scalars,
        predicted_quality=np.asarray([0.1, 0.9, 0.2, 0.3]),
        pairwise_scores=np.asarray([0.1, 0.9, 0.2, 0.3]),
        weights={"pairwise": 0.5, "utility": 0.5},
        resample_threshold=0.5,
    )

    assert output["selected_sample_id"] == "zh_001_c1"
    assert output["planned_3d_calls"] == 1
    assert output["resample_2d_recommended"] is True
    assert output["resample_reason"] == "all_candidates_failed_structure_gate"


def test_low_selected_quality_recommends_new_2d_round_not_top2():
    output = build_top1_output(
        _rows(),
        _healthy_scalars(),
        predicted_quality=np.asarray([0.41, 0.40, 0.39, 0.38]),
        pairwise_scores=np.asarray([0.9, 0.8, 0.7, 0.6]),
        weights={"pairwise": 0.5, "utility": 0.5},
        resample_threshold=0.5,
    )

    assert output["resample_2d_recommended"] is True
    assert output["resample_reason"] == "selected_predicted_quality_below_validation_threshold"
    assert output["planned_3d_calls"] == 1


def test_prediction_write_is_atomic_json(tmp_path):
    path = tmp_path / "prediction.json"
    payload = {"schema_version": "test", "selected_sample_id": "x"}

    write_prediction_atomic(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert not (tmp_path / "prediction.json.tmp").exists()
