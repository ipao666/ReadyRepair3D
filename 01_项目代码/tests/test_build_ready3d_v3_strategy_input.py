from ops.build_ready3d_v3_strategy_input import build_strategy_rows


def test_strategy_input_joins_test_truth_features_and_two_model_scores():
    targets = [
        {
            "sample_id": "g_c0",
            "group_id": "g",
            "candidate_index": 0,
            "split": "test",
            "quality_score_v2": 0.8,
            "qualified": True,
            "technically_valid": True,
        },
        {
            "sample_id": "train_c0",
            "group_id": "train",
            "candidate_index": 0,
            "split": "train",
            "quality_score_v2": 0.4,
            "qualified": False,
            "technically_valid": True,
        },
    ]
    features = [
        {
            "sample_id": "g_c0",
            "mask_area_ratio": 0.5,
            "mask_largest_component_ratio": 0.95,
            "occlusion_border_contact_ratio": 0.01,
        },
        {
            "sample_id": "train_c0",
            "mask_area_ratio": 0.5,
            "mask_largest_component_ratio": 0.95,
            "occlusion_border_contact_ratio": 0.01,
        },
    ]

    rows = build_strategy_rows(
        targets,
        features,
        v2_scores={"g_c0": 0.6, "train_c0": 0.2},
        v3_scores={"g_c0": 0.7, "train_c0": 0.3},
        split="test",
    )

    assert len(rows) == 1
    assert rows[0]["quality_score"] == 0.8
    assert rows[0]["ready_v2_score"] == 0.6
    assert rows[0]["ready_v3_score"] == 0.7
    assert rows[0]["structural_input_pass"] is True
