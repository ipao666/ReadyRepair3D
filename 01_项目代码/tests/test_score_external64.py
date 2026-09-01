import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "score_external64.py"


def load_module():
    spec = importlib.util.spec_from_file_location("score_external64", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_group_indices_require_complete_best_of_four_groups():
    module = load_module()
    rows = [
        {"group_id": "g1", "candidate_index": index, "sample_id": f"g1_c{index}"}
        for index in range(4)
    ]
    assert module.group_indices(rows) == {"g1": [0, 1, 2, 3]}
    with pytest.raises(ValueError, match="four candidates"):
        module.group_indices(rows[:3])


def test_severe_input_flags_only_trigger_on_catastrophic_structure():
    module = load_module()
    healthy = {
        "mask_area_ratio": 0.35,
        "occlusion_border_contact_ratio": 0.0,
        "mask_largest_component_ratio": 0.98,
    }
    assert module.severe_input_flags(healthy) == []
    broken = {
        "mask_area_ratio": 0.01,
        "occlusion_border_contact_ratio": 0.20,
        "mask_largest_component_ratio": 0.60,
    }
    assert set(module.severe_input_flags(broken)) == {
        "tiny_foreground", "border_contact", "fragmented_foreground"
    }


def test_parse_args_accepts_top_two(monkeypatch):
    module = load_module()
    monkeypatch.setattr("sys.argv", ["score_external64.py", "--top-k", "2"])

    args = module.parse_args()

    assert args.top_k == 2
