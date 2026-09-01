import pytest

from ops.prepare_ready3d_v3_training_inputs import prepare_splits


def _rows():
    manifest = []
    labels = []
    for group, split in (("a", "train"), ("b", "validation"), ("c", "test")):
        for candidate in range(4):
            sample_id = f"{group}_c{candidate}"
            manifest.append(
                {
                    "sample_id": sample_id,
                    "group_id": group,
                    "candidate_index": candidate,
                    "split": split,
                }
            )
            labels.append({"sample_id": sample_id, "quality_score_v2": candidate / 4})
    return manifest, labels


def test_prepare_splits_keeps_groups_and_labels_aligned():
    manifest, labels = _rows()

    result = prepare_splits(manifest, labels, expected_group_counts={"train": 1, "validation": 1, "test": 1})

    assert len(result["train"]["manifest"]) == 4
    assert len(result["validation"]["labels"]) == 4
    assert {row["group_id"] for row in result["test"]["manifest"]} == {"c"}


def test_prepare_splits_rejects_missing_label():
    manifest, labels = _rows()
    with pytest.raises(ValueError, match="manifest/label mismatch"):
        prepare_splits(manifest, labels[:-1], expected_group_counts={"train": 1, "validation": 1, "test": 1})
