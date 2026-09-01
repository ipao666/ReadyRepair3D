import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "ops" / "select_best_hunyuan.py"


def load_module():
    spec = importlib.util.spec_from_file_location("select_best_hunyuan", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest():
    return [
        {"sample_id": "g_c0", "candidate_index": 0},
        {"sample_id": "g_c2", "candidate_index": 2},
    ]


def label(
    sample_id,
    score,
    valid=True,
    largest=0.9,
    components=1,
    glb_path=None,
):
    return {
        "sample_id": sample_id,
        "quality_score_v2": score,
        "technically_valid": valid,
        "largest_component_area_ratio": largest,
        "effective_component_count": components,
        "glb_path": glb_path or f"/tmp/{sample_id}.glb",
    }


def test_one_valid_asset_wins_even_with_lower_score():
    module = load_module()

    result = module.select_best(
        [label("g_c0", 0.70, valid=True), label("g_c2", 0.99, valid=False)],
        manifest(),
    )

    assert result["winner_sample_id"] == "g_c0"
    assert result["round_failed"] is False
    assert result["selection_reason"] == "only_technically_valid"


def test_two_valid_assets_choose_higher_quality():
    module = load_module()

    result = module.select_best(
        [label("g_c0", 0.75), label("g_c2", 0.88)],
        manifest(),
    )

    assert result["winner_sample_id"] == "g_c2"
    assert result["selection_reason"] == "highest_quality_score_v2"
    assert result["meets_quality_threshold"] is True


def test_no_valid_assets_return_diagnostic_winner():
    module = load_module()

    result = module.select_best(
        [label("g_c0", 0.65, valid=False), label("g_c2", 0.70, valid=False)],
        manifest(),
    )

    assert result["winner_sample_id"] == "g_c2"
    assert result["round_failed"] is True
    assert result["selection_reason"] == "diagnostic_best_no_valid_assets"
    assert result["meets_quality_threshold"] is False


def test_equal_scores_use_frozen_tie_break_order():
    module = load_module()

    larger_body = module.select_best(
        [
            label("g_c0", 0.80, largest=0.90, components=1),
            label("g_c2", 0.80, largest=0.95, components=3),
        ],
        manifest(),
    )
    fewer_components = module.select_best(
        [
            label("g_c0", 0.80, largest=0.95, components=1),
            label("g_c2", 0.80, largest=0.95, components=3),
        ],
        manifest(),
    )
    smaller_index = module.select_best(
        [
            label("g_c0", 0.80, largest=0.95, components=1),
            label("g_c2", 0.80, largest=0.95, components=1),
        ],
        manifest(),
    )

    assert larger_body["winner_sample_id"] == "g_c2"
    assert fewer_components["winner_sample_id"] == "g_c0"
    assert smaller_index["winner_sample_id"] == "g_c0"


def test_report_contains_both_candidates_and_frozen_threshold():
    module = load_module()

    result = module.select_best(
        [label("g_c0", 0.75), label("g_c2", 0.76)],
        manifest(),
    )

    assert result["quality_threshold"] == 0.806912747446761
    assert [row["sample_id"] for row in result["candidates"]] == ["g_c0", "g_c2"]


def test_runtime_seeded_ids_align_by_preserved_source_path():
    module = load_module()
    runtime_manifest = [
        {
            "sample_id": "g_c0",
            "candidate_index": 0,
            "source_path": "/images/g_c0_s100.png",
        },
        {
            "sample_id": "g_c2",
            "candidate_index": 2,
            "source_path": "/images/g_c2_s102.png",
        },
    ]
    labels = [
        {
            **label("g_c0_s100", 0.75),
            "source_path": "/images/g_c0_s100.png",
        },
        {
            **label("g_c2_s102", 0.88),
            "source_path": "/images/g_c2_s102.png",
        },
    ]

    result = module.select_best(labels, runtime_manifest)

    assert result["winner_sample_id"] == "g_c2_s102"
    assert result["winner_candidate_index"] == 2


def test_manifest_without_sample_id_aligns_by_path_basename():
    module = load_module()
    runtime_manifest = [
        {
            "prompt_id": "g",
            "candidate_index": 0,
            "path": "/root/project/images/g_c0_s100.png",
        },
        {
            "prompt_id": "g",
            "candidate_index": 2,
            "path": "/root/project/images/g_c2_s102.png",
        },
    ]
    labels = [
        {
            **label("g_c0_s100", 0.75),
            "source_path": "data/images/g_c0_s100.png",
        },
        {
            **label("g_c2_s102", 0.88),
            "source_path": "data/images/g_c2_s102.png",
        },
    ]

    result = module.select_best(labels, runtime_manifest)

    assert result["winner_sample_id"] == "g_c2_s102"
    assert result["winner_candidate_index"] == 2


def test_single_semantically_valid_candidate_can_be_delivered():
    module = load_module()

    result = module.select_best(
        [label("g_c2", 0.71)],
        [{"sample_id": "g_c2", "candidate_index": 2}],
        threshold=0.75,
    )

    assert result["winner_sample_id"] == "g_c2"
    assert result["selection_reason"] == "only_candidate_after_filtering"
    assert result["meets_quality_threshold"] is False
