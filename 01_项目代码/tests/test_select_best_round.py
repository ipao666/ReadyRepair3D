import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "ops" / "select_best_round.py"


def load_module():
    spec = importlib.util.spec_from_file_location("select_best_round", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def round_report(score, valid=True, path="/round/final.glb"):
    return {
        "final_quality_score_v2": score,
        "final_technically_valid": valid,
        "final_glb": path,
        "repair3d_accepted": False,
        "v2_repair_accepted": False,
    }


def test_single_round_is_selected():
    module = load_module()

    report = module.select_rounds([round_report(0.85)])

    assert report["selected_round"] == 1
    assert report["round_count"] == 1
    assert report["meets_quality_threshold"] is True
    assert report["low_confidence"] is False


def test_valid_round_beats_invalid_higher_numeric_score():
    module = load_module()

    report = module.select_rounds(
        [round_report(0.70, valid=True), round_report(0.99, valid=False)]
    )

    assert report["selected_round"] == 1
    assert report["final_quality_score_v2"] == 0.70


def test_higher_valid_round_score_wins():
    module = load_module()

    report = module.select_rounds(
        [round_report(0.70), round_report(0.88)]
    )

    assert report["selected_round"] == 2
    assert report["final_quality_score_v2"] == 0.88


def test_exact_tie_prefers_round_one():
    module = load_module()

    report = module.select_rounds(
        [round_report(0.80), round_report(0.80)]
    )

    assert report["selected_round"] == 1


def test_two_invalid_rounds_are_low_confidence():
    module = load_module()

    report = module.select_rounds(
        [round_report(0.60, valid=False), round_report(0.70, valid=False)]
    )

    assert report["selected_round"] == 2
    assert report["low_confidence"] is True
    assert report["meets_quality_threshold"] is False
