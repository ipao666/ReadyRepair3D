import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "ops" / "run_repair3d_auto.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_repair3d_auto", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rank_actions_orders_predicted_gain_descending(monkeypatch):
    module = load_module()

    class Model:
        def predict(self, matrix):
            assert matrix.shape[0] == 2
            return np.asarray([-0.1, 0.2])

    monkeypatch.setattr(module, "feature_vector", lambda metrics, action, actions: [actions.index(action)])
    artifact = {"actions": ["first", "second"], "model": Model(), "model_name": "fake"}

    ranking = module.rank_actions({"quality_score": 0.5}, artifact)

    assert ranking[0] == {"action": "second", "predicted_gain": 0.2}


def test_decide_candidate_rolls_back_no_change():
    module = load_module()

    accepted, reason, gain = module.decide_candidate(
        {"quality_score": 0.5}, {"quality_score": 0.6}, {}, changed=False,
        blender_loadable=True,
    )

    assert accepted is False
    assert reason == "no_change"
    assert gain == 0.1


def test_decide_candidate_requires_positive_gain(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "hard_constraints_hold", lambda before, after: (True, None))
    monkeypatch.setattr(module, "fidelity_constraints_hold", lambda fidelity: (True, None))

    accepted, reason, gain = module.decide_candidate(
        {"quality_score": 0.7}, {"quality_score": 0.7}, {}, changed=True,
        blender_loadable=True,
    )

    assert accepted is False
    assert reason == "no_quality_improvement"
    assert gain == 0.0


def test_decide_candidate_requires_frozen_quality_margin(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "hard_constraints_hold", lambda before, after: (True, None))
    monkeypatch.setattr(module, "fidelity_constraints_hold", lambda fidelity: (True, None))

    accepted, reason, gain = module.decide_candidate(
        {"quality_score": 0.7000}, {"quality_score": 0.7049}, {}, changed=True,
        blender_loadable=True,
    )

    assert accepted is False
    assert reason == "quality_gain_below_0.005"
    assert gain == 0.0049


def test_finalize_delivery_preserves_original_on_rollback(tmp_path):
    module = load_module()
    original = tmp_path / "original.glb"
    candidate = tmp_path / "candidate.glb"
    final = tmp_path / "final.glb"
    repaired = tmp_path / "repaired.glb"
    original.write_bytes(b"original")
    candidate.write_bytes(b"candidate")

    delivered = module.finalize_delivery(
        original, candidate, repaired, final, accepted=False
    )

    assert delivered == original
    assert final.read_bytes() == b"original"
    assert candidate.read_bytes() == b"candidate"
    assert not repaired.exists()


def test_clean_input_is_accept_original_and_sha_identical(tmp_path):
    module = load_module()
    original = tmp_path / "original.glb"
    candidate = tmp_path / "candidate.glb"
    final = tmp_path / "final.glb"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    original.write_bytes(b"clean-glb")
    args = Namespace(
        checkpoint=tmp_path / "missing.joblib",
        output_dir=output_dir,
    )
    metrics = {
        "trimesh_loadable": True,
        "blender_loadable": True,
        "finite": True,
        "faces": 40000,
        "geometry_count": 1,
        "largest_component_face_ratio": 1.0,
        "boundary_edges": 0,
        "non_manifold_edges": 0,
        "degenerate_faces": 0,
        "duplicate_faces": 0,
        "texture_present": True,
        "uv_coverage_ratio": 1.0,
        "quality_score": 1.0,
    }

    module.write_safe_fallback_report(
        args, original, candidate, final, metrics, blender_available=True
    )

    report = json.loads((output_dir / "repair_report.json").read_text())
    summary = json.loads((output_dir / "pipeline_summary.json").read_text())
    assert final.read_bytes() == b"clean-glb"
    assert report["mode"] == "safe_fallback"
    assert report["learned_checkpoint_available"] is False
    assert report["status"] == "accept_original"
    assert report["recommended_action"] == "accept_original"
    assert report["executed_actions"] == []
    assert report["accepted"] is False
    assert report["rollback_reason"] is None
    assert report["sha256"]["original"] == report["sha256"]["final"]
    assert candidate.exists()
    assert summary["status"] == "success"
    assert summary["mode"] == "safe_fallback"
    assert summary["repair_status"] == "accept_original"


def test_rule_actions_choose_only_low_risk_repairs():
    module = load_module()
    metrics = {
        "trimesh_loadable": True,
        "finite": True,
        "faces": 10000,
        "geometry_count": 1,
        "largest_component_face_ratio": 0.995,
        "component_face_counts": [9950, 50],
        "component_area_ratios": [0.997, 0.003],
        "boundary_edges": 4,
        "hole_count_approx": 1,
        "non_manifold_edges": 0,
        "degenerate_faces": 2,
        "duplicate_faces": 1,
        "mergeable_duplicate_vertices": 3,
        "normal_inconsistent_edges": 2,
    }

    decision = module.choose_rule_actions(metrics)

    assert decision["status"] == "repair_rollback"
    assert decision["candidate_ready_for_rescore"] is True
    assert decision["actions"] == [
        "remove_floaters",
        "merge_close_vertices",
        "cleanup_degenerate_faces",
        "fix_face_normals",
        "fill_small_holes",
    ]


def test_catastrophic_asset_requires_regeneration():
    module = load_module()
    metrics = {
        "trimesh_loadable": True,
        "finite": True,
        "faces": 1000,
        "geometry_count": 1,
        "largest_component_face_ratio": 0.40,
        "boundary_edges": 400,
        "hole_count_approx": 12,
        "non_manifold_edges": 100,
    }

    decision = module.choose_rule_actions(metrics)

    assert decision["status"] == "regenerate_required"
    assert decision["actions"] == []


def test_composite_action_names_expand_to_runtime_primitives():
    module = load_module()

    assert module.action_steps("cleanup_debris_and_merge_vertices") == (
        "remove_floaters",
        "merge_close_vertices",
        "cleanup_degenerate_faces",
    )
    assert module.action_steps("repair_small_holes_and_nonmanifold") == (
        "fill_small_holes",
        "cleanup_degenerate_faces",
        "fix_face_normals",
    )
    assert module.action_steps("fix_normals") == ("fix_face_normals",)
    assert module.action_steps("remesh_or_decimate") == ("decimate_mesh",)
    assert module.action_steps("accept") == ()
