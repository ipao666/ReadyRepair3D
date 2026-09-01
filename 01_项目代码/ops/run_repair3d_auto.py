#!/usr/bin/env python3
"""Run one Repair3D action with hard constraints and automatic rollback."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import trimesh

OPS_DIR = Path(__file__).resolve().parent
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

from repair3d_mesh import (
    ACTIONS,
    apply_blender_result,
    diagnose_glb,
    fidelity_constraints_hold,
    file_sha256,
    geometry_fidelity,
    hard_constraints_hold,
)
from run_repair3d_decimate_trials import run_decimate_jobs
from run_repair3d_diagnostics import blender_checks
from train_repair3d_action_model import feature_vector


ROOT = Path(__file__).resolve().parents[1]

COMPOSITE_ACTIONS: dict[str, tuple[str, ...]] = {
    "accept": (),
    "cleanup_debris_and_merge_vertices": (
        "remove_floaters",
        "merge_close_vertices",
        "cleanup_degenerate_faces",
    ),
    "repair_small_holes_and_nonmanifold": (
        "fill_small_holes",
        "cleanup_degenerate_faces",
        "fix_face_normals",
    ),
    "fix_normals": ("fix_face_normals",),
    "remesh_or_decimate": ("decimate_mesh",),
}


def action_steps(action: str) -> tuple[str, ...]:
    if action in COMPOSITE_ACTIONS:
        return COMPOSITE_ACTIONS[action]
    if action in ACTIONS or action == "decimate_mesh":
        return (action,)
    raise ValueError(f"unsupported Repair3D action: {action}")


def apply_geometry_steps(scene: trimesh.Scene, steps: tuple[str, ...]):
    current = scene
    changed = False
    details: dict[str, Any] = {}
    for step in steps:
        if step == "decimate_mesh":
            raise ValueError("decimate_mesh must run through Blender")
        result = ACTIONS[step](current)
        current = result.scene
        changed = changed or result.changed
        details[step] = result.details
    return current, changed, details


def rank_actions(metrics: dict[str, Any], artifact: dict[str, Any]) -> list[dict]:
    actions = artifact["actions"]
    matrix = np.vstack([feature_vector(metrics, action, actions) for action in actions])
    predictions = artifact["model"].predict(matrix)
    return sorted(
        [
            {"action": action, "predicted_gain": float(prediction)}
            for action, prediction in zip(actions, predictions, strict=True)
        ],
        key=lambda row: row["predicted_gain"],
        reverse=True,
    )


def decide_candidate(
    before: dict, after: dict, fidelity: dict, *, changed: bool,
    blender_loadable: bool,
) -> tuple[bool, str | None, float]:
    gain = round(float(after["quality_score"]) - float(before["quality_score"]), 8)
    if not changed:
        return False, "no_change", gain
    if not blender_loadable:
        return False, "blender_reload_failed", gain
    constraints_ok, reason = hard_constraints_hold(before, after)
    if constraints_ok:
        fidelity_ok, fidelity_reason = fidelity_constraints_hold(fidelity)
        if not fidelity_ok:
            constraints_ok, reason = False, fidelity_reason
    if not constraints_ok:
        return False, reason, gain
    if gain <= 0:
        return False, "no_quality_improvement", gain
    if gain < 0.005:
        return False, "quality_gain_below_0.005", gain
    return True, None, gain


def finalize_delivery(
    original: Path, candidate: Path, repaired: Path, final: Path, *, accepted: bool
) -> Path:
    repaired.unlink(missing_ok=True)
    final.unlink(missing_ok=True)
    if accepted:
        shutil.copy2(candidate, repaired)
        shutil.copy2(repaired, final)
        return repaired
    shutil.copy2(original, final)
    return original


def blender_check(path: Path, blender: str) -> dict:
    return blender_checks(
        [path], blender, Path(__file__).with_name("blender_check_glb.py")
    )[str(path.resolve())]


def write_pipeline_summary(output_dir: Path, report: dict[str, Any]) -> None:
    summary = {
        "stage": "repair3d",
        "status": "success",
        "mode": "safe_fallback",
        "repair_status": report.get("status"),
        "learned_checkpoint_available": False,
        "recommended_action": report.get("recommended_action"),
        "executed_actions": report.get("executed_actions", []),
        "accepted": report.get("accepted"),
        "rollback_reason": report.get("rollback_reason"),
        "quality_gain": report.get("quality_gain"),
        "final_path": report.get("final_path"),
        "report_path": str((output_dir / "repair_report.json").resolve()),
    }
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def choose_rule_actions(metrics: dict[str, Any]) -> dict[str, Any]:
    """Choose only the conservative, deterministic Repair3D action set."""
    faces = max(int(metrics.get("faces", 0)), 1)
    edge_instances = max(int(metrics.get("edge_instances", faces * 3)), 1)
    boundary_edges = int(metrics.get("boundary_edges", 0))
    non_manifold = int(metrics.get("non_manifold_edges", 0))
    holes = int(metrics.get("hole_count_approx", 0))
    main_ratio = float(metrics.get("largest_component_face_ratio", 0.0))

    invalid = (
        not metrics.get("trimesh_loadable")
        or not metrics.get("finite")
        or int(metrics.get("faces", 0)) <= 0
        or int(metrics.get("geometry_count", 0)) <= 0
    )
    catastrophic = (
        invalid
        or main_ratio < 0.65
        or boundary_edges / edge_instances > 0.05
        or non_manifold / edge_instances > 0.02
        or holes > 8
    )
    if catastrophic:
        return {
            "status": "regenerate_required",
            "actions": [],
            "candidate_ready_for_rescore": False,
            "reason": "catastrophic_or_incomplete_topology",
        }

    actions: list[str] = []
    component_faces = list(metrics.get("component_face_counts", []))
    component_areas = list(metrics.get("component_area_ratios", []))
    if int(metrics.get("connected_components", len(component_faces))) > 1:
        face_ratios = [value / faces for value in component_faces]
        if any(value < 0.01 for value in face_ratios) or any(
            value < 0.01 for value in component_areas
        ):
            actions.append("remove_floaters")
    if int(metrics.get("mergeable_duplicate_vertices", 0)) > 0:
        actions.append("merge_close_vertices")
    if (
        int(metrics.get("degenerate_faces", 0)) > 0
        or int(metrics.get("duplicate_faces", 0)) > 0
    ):
        actions.append("cleanup_degenerate_faces")
    if (
        int(metrics.get("normal_inconsistent_edges", 0)) > 0
        or not bool(metrics.get("is_winding_consistent", True))
    ):
        actions.append("fix_face_normals")
    if boundary_edges:
        if holes <= 2 and boundary_edges <= 16:
            actions.append("fill_small_holes")
        else:
            return {
                "status": "regenerate_required",
                "actions": [],
                "candidate_ready_for_rescore": False,
                "reason": "open_boundary_exceeds_safe_hole_limit",
            }

    if not actions:
        return {
            "status": "accept_original",
            "actions": [],
            "candidate_ready_for_rescore": False,
            "reason": None,
        }
    return {
        # A candidate is not accepted until Blender and frozen-Q validation.
        "status": "repair_rollback",
        "actions": actions,
        "candidate_ready_for_rescore": True,
        "reason": "candidate_requires_blender_and_frozen_quality_rescore",
    }


def _hashes(original: Path, candidate: Path, final: Path) -> dict[str, str | None]:
    return {
        "original": file_sha256(original) if original.is_file() else None,
        "candidate": file_sha256(candidate) if candidate.is_file() else None,
        "final": file_sha256(final) if final.is_file() else None,
    }


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "repair_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_pipeline_summary(output_dir, report)


def write_safe_fallback_report(
    args: argparse.Namespace,
    original: Path,
    candidate: Path,
    final: Path,
    before: dict,
    *,
    blender_available: bool,
) -> None:
    shutil.copy2(original, candidate)
    shutil.copy2(original, final)
    decision = choose_rule_actions(before)
    if decision["actions"]:
        raise ValueError("write_safe_fallback_report only handles unchanged inputs")
    status = decision["status"]
    reason = decision["reason"]
    report = {
        "schema_version": "r3dguard.repair3d-auto.v2",
        "mode": "safe_fallback",
        "learned_checkpoint_available": False,
        "model": None,
        "status": status,
        "recommended_action": status,
        "executed_actions": [],
        "accepted": False,
        "rollback_reason": reason,
        "quality_gain": 0.0,
        "frozen_quality_before": None,
        "frozen_quality_after": None,
        "quality_before": before,
        "quality_after": before,
        "technical_checks": {"blender_available": blender_available},
        "fidelity": {},
        "action_details": {},
        "candidate_ready_for_rescore": False,
        "original_path": str(original.resolve()),
        "candidate_path": str(candidate.resolve()),
        "final_path": str(final.resolve()),
        "delivered_source": str(original.resolve()),
        "sha256": _hashes(original, candidate, final),
    }
    _write_report(args.output_dir, report)


def run_rule_repair(
    source: Path,
    output_dir: Path,
    *,
    blender: str = "blender",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    original = output_dir / "original.glb"
    candidate = output_dir / "candidate.glb"
    repaired = output_dir / "repaired.glb"
    final = output_dir / "final.glb"
    shutil.copy2(source, original)
    candidate.unlink(missing_ok=True)
    repaired.unlink(missing_ok=True)

    before = diagnose_glb(original, "hunyuan")
    blender_available = shutil.which(blender) is not None
    if blender_available:
        apply_blender_result(before, blender_check(original, blender))
    decision = choose_rule_actions(before)

    if not decision["actions"]:
        shutil.copy2(original, candidate)
        shutil.copy2(original, final)
        report = {
            "schema_version": "r3dguard.repair3d-auto.v2",
            "mode": "safe_fallback",
            "learned_checkpoint_available": False,
            "status": decision["status"],
            "recommended_action": decision["status"],
            "executed_actions": [],
            "accepted": False,
            "rollback_reason": decision["reason"],
            "quality_gain": 0.0,
            "frozen_quality_before": None,
            "frozen_quality_after": None,
            "quality_before": before,
            "quality_after": before,
            "fidelity": {},
            "action_details": {},
            "candidate_ready_for_rescore": False,
            "technical_checks": {"blender_available": blender_available},
            "original_path": str(original.resolve()),
            "candidate_path": str(candidate.resolve()),
            "final_path": str(final.resolve()),
            "delivered_source": str(original.resolve()),
        }
        report["sha256"] = _hashes(original, candidate, final)
        _write_report(output_dir, report)
        return report

    original_scene = trimesh.load(original, force="scene", process=False)
    candidate_scene, changed, details = apply_geometry_steps(
        original_scene, tuple(decision["actions"])
    )
    if changed:
        candidate_scene.export(file_obj=candidate, file_type="glb")
    else:
        shutil.copy2(original, candidate)
    after = diagnose_glb(candidate, "hunyuan")
    if blender_available:
        apply_blender_result(after, blender_check(candidate, blender))
    fidelity = geometry_fidelity(original_scene, candidate_scene)

    constraints_ok, reason = hard_constraints_hold(before, after)
    if constraints_ok:
        constraints_ok, reason = fidelity_constraints_hold(fidelity)
    geometry_candidate_ready = bool(changed and constraints_ok)
    if geometry_candidate_ready and not blender_available:
        constraints_ok = False
        reason = "blender_validation_required"
    elif geometry_candidate_ready and not bool(after.get("blender_loadable")):
        constraints_ok = False
        reason = "blender_reload_failed"

    # Frozen Q is deliberately not approximated locally. The original is delivered
    # until the A100 post-render stage proves an improvement of at least 0.005.
    status = "repair_rollback"
    candidate_ready = bool(changed and constraints_ok and blender_available)
    rollback_reason = (
        "frozen_quality_rescore_required"
        if candidate_ready
        else (reason or "no_safe_geometry_improvement")
    )
    shutil.copy2(original, final)
    hashes = _hashes(original, candidate, final)
    if hashes["original"] != hashes["final"]:
        raise RuntimeError("rollback SHA256 mismatch")
    report = {
        "schema_version": "r3dguard.repair3d-auto.v2",
        "mode": "safe_fallback",
        "learned_checkpoint_available": False,
        "status": status,
        "recommended_action": "conservative_rule_combination",
        "executed_actions": decision["actions"],
        "accepted": False,
        "rollback_reason": rollback_reason,
        "quality_gain": None,
        "frozen_quality_before": None,
        "frozen_quality_after": None,
        "quality_before": before,
        "quality_after": after,
        "fidelity": fidelity,
        "action_details": details,
        "candidate_ready_for_rescore": candidate_ready,
        "candidate_prepared_for_server": geometry_candidate_ready,
        "technical_checks": {
            "trimesh_reload": bool(after.get("trimesh_loadable")),
            "blender_available": blender_available,
            "blender_reload": after.get("blender_loadable"),
            "hard_constraints": constraints_ok,
        },
        "original_path": str(original.resolve()),
        "candidate_path": str(candidate.resolve()),
        "final_path": str(final.resolve()),
        "delivered_source": str(original.resolve()),
        "sha256": hashes,
    }
    _write_report(output_dir, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("glb", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "disabled.joblib")
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    if not args.glb.is_file():
        raise FileNotFoundError(args.glb)
    report = run_rule_repair(args.glb, args.output_dir, blender=args.blender)
    print(
        json.dumps(
            {
                "status": report["status"],
                "actions": report["executed_actions"],
                "candidate_ready_for_rescore": report[
                    "candidate_ready_for_rescore"
                ],
                "rollback_reason": report["rollback_reason"],
                "final": report["final_path"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
