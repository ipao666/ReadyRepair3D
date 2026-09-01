import numpy as np

from ops.train_ready3d_v3_top1 import fit_top1_ensemble


def test_fit_top1_ensemble_freezes_on_validation_and_selects_one_test_candidate():
    rows = []
    qualities = []
    scalar = []
    dino = []
    splits = ["train"] * 4 + ["validation"] * 2 + ["test"] * 2
    for group, split in enumerate(splits):
        for candidate in range(4):
            signal = (candidate + group % 2) % 4 / 3.0
            rows.append(
                {
                    "group_id": f"g{group}",
                    "sample_id": f"g{group}_c{candidate}",
                    "candidate_index": candidate,
                    "split": split,
                }
            )
            qualities.append(signal)
            scalar.append([0.5, 0.95, 0.01])
            dino.append([signal, 1.0 - signal])
    data = {
        "rows": rows,
        "quality": np.asarray(qualities, dtype=np.float32),
        "scalar": np.asarray(scalar, dtype=np.float32),
        "dino": np.asarray(dino, dtype=np.float32),
        "scalar_names": [
            "mask_area_ratio",
            "mask_largest_component_ratio",
            "occlusion_border_contact_ratio",
        ],
        "scoring_version": "ready3d-v2",
    }

    checkpoint, metrics, test_records = fit_top1_ensemble(
        data, seed=7, n_estimators=20
    )

    assert checkpoint["schema_version"] == "r3dguard.ready3d-v3-top1-checkpoint.v1"
    assert checkpoint["model_selection"]["test_used_for_selection"] is False
    assert metrics["group_counts"] == {"train": 4, "validation": 2, "test": 2}
    assert metrics["test"]["groups"] == 2
    assert len(test_records) == 2
    assert len({row["group_id"] for row in test_records}) == 2
