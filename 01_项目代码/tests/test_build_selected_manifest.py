import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "ops" / "build_selected_manifest.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_selected_manifest", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_rows_requires_one_selected_candidate_per_group():
    module = load_module()
    manifest = [
        {"sample_id": f"g1_c{i}", "group_id": "g1", "candidate_index": i, "source_path": f"/{i}.png"}
        for i in range(4)
    ]
    predictions = [
        {"sample_id": f"g1_c{i}", "group_id": "g1", "selected": i == 2, "predicted_quality": 0.5 + i / 10}
        for i in range(4)
    ]

    selected = module.select_rows(manifest, predictions)

    assert len(selected) == 1
    assert selected[0]["sample_id"] == "g1_c2"
    assert selected[0]["source_path"] == "/2.png"
    assert selected[0]["ready3d"]["predicted_quality"] == pytest.approx(0.7)


def test_select_rows_rejects_multiple_selected_candidates():
    module = load_module()
    manifest = [{"sample_id": "a", "group_id": "g", "candidate_index": 0}]
    predictions = [
        {"sample_id": "a", "group_id": "g", "selected": True},
        {"sample_id": "b", "group_id": "g", "selected": True},
    ]

    with pytest.raises(ValueError, match="exactly one"):
        module.select_rows(manifest, predictions)


def test_select_rows_supports_two_ranked_candidates_per_group():
    module = load_module()
    manifest = [
        {
            "sample_id": f"g1_c{i}",
            "group_id": "g1",
            "candidate_index": i,
            "source_path": f"/{i}.png",
        }
        for i in range(4)
    ]
    predictions = [
        {
            "sample_id": f"g1_c{i}",
            "group_id": "g1",
            "selected": i in (0, 2),
            "selection_rank": {2: 1, 0: 2}.get(i),
            "predicted_quality": 0.5 + i / 10,
        }
        for i in range(4)
    ]

    selected = module.select_rows(
        manifest, predictions, expected_per_group=2
    )

    assert [row["sample_id"] for row in selected] == ["g1_c2", "g1_c0"]
    assert [row["ready3d"]["selection_rank"] for row in selected] == [1, 2]


def test_select_rows_rejects_duplicate_selection_rank():
    module = load_module()
    manifest = [
        {"sample_id": "g_c0", "group_id": "g", "candidate_index": 0},
        {"sample_id": "g_c1", "group_id": "g", "candidate_index": 1},
    ]
    predictions = [
        {
            "sample_id": "g_c0",
            "group_id": "g",
            "selected": True,
            "selection_rank": 1,
        },
        {
            "sample_id": "g_c1",
            "group_id": "g",
            "selected": True,
            "selection_rank": 1,
        },
    ]

    with pytest.raises(ValueError, match="selection ranks"):
        module.select_rows(manifest, predictions, expected_per_group=2)


def test_select_rows_rejects_missing_second_rank():
    module = load_module()
    manifest = [
        {"sample_id": "g_c0", "group_id": "g", "candidate_index": 0},
        {"sample_id": "g_c1", "group_id": "g", "candidate_index": 1},
    ]
    predictions = [
        {
            "sample_id": "g_c0",
            "group_id": "g",
            "selected": True,
            "selection_rank": 1,
        }
    ]

    with pytest.raises(ValueError, match="exactly 2"):
        module.select_rows(manifest, predictions, expected_per_group=2)
