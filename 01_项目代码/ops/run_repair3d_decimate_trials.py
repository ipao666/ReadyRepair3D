from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import trimesh

from repair3d_mesh import (
    apply_blender_result,
    diagnose_glb,
    fidelity_constraints_hold,
    file_sha256,
    geometry_fidelity,
    hard_constraints_hold,
    write_jsonl,
)
from run_repair3d_diagnostics import blender_checks, blender_environment


ROOT = Path("/root/r3dguard")


def run_decimate_jobs(
    jobs: list[dict[str, Any]], blender: str, script: Path
) -> dict[str, dict[str, Any]]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as handle:
        json.dump(jobs, handle, ensure_ascii=False)
        handle.flush()
        process = subprocess.run(
            [blender, "--background", "--python", str(script), "--", handle.name],
            capture_output=True,
            text=True,
            check=False,
            env=blender_environment(),
        )
    results: dict[str, dict[str, Any]] = {}
    for line in (process.stdout + "\n" + process.stderr).splitlines():
        if line.startswith("REPAIR3D_DECIMATE="):
            result = json.loads(line.split("=", 1)[1])
            results[str(Path(result["source"]).resolve())] = result
    for job in jobs:
        source = str(Path(job["source"]).resolve())
        results.setdefault(
            source,
            {
                "source": source,
                "target": job["target"],
                "success": False,
                "error": f"Blender returned {process.returncode} without a result",
            },
        )
    return results


def decide_decimation(
    before: dict[str, Any],
    after: dict[str, Any],
    fidelity: dict[str, Any],
) -> tuple[bool, str | None, float]:
    constraints_ok, reason = hard_constraints_hold(before, after)
    if constraints_ok:
        fidelity_ok, fidelity_reason = fidelity_constraints_hold(fidelity)
        if not fidelity_ok:
            constraints_ok = False
            reason = fidelity_reason
    gain = round(after["quality_score"] - before["quality_score"], 8)
    if constraints_ok and gain <= 0:
        return False, "no_quality_improvement", gain
    return constraints_ok and gain > 0, reason, gain


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Blender decimation with rollback")
    parser.add_argument(
        "--defect-manifest",
        type=Path,
        default=ROOT / "data/repair3d_actions/defect_manifest.jsonl",
    )
    parser.add_argument(
        "--phase2-trials",
        type=Path,
        default=ROOT / "data/repair3d_actions/action_trials_phase2.jsonl",
    )
    parser.add_argument(
        "--clean-diagnostics",
        type=Path,
        default=ROOT / "evaluation/mesh_diagnostics.jsonl",
    )
    parser.add_argument(
        "--clean-trials",
        type=Path,
        default=ROOT / "data/repair3d_actions/action_trials.jsonl",
    )
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()

    defect_manifest = [
        json.loads(line) for line in args.defect_manifest.open(encoding="utf-8")
    ]
    clean_diagnostics = [
        json.loads(line) for line in args.clean_diagnostics.open(encoding="utf-8")
    ]
    items: list[dict[str, Any]] = []
    for record in defect_manifest:
        source = Path(record["defect_path"])
        target = (
            ROOT
            / "data/repair3d_actions/outputs_phase2"
            / source.parent.name
            / record["defect_type"]
            / "decimate_mesh.glb"
        )
        items.append(
            {
                "kind": "defect",
                "sample_id": record["sample_id"],
                "defect_type": record["defect_type"],
                "intended_action": record["intended_action"],
                "source": source,
                "clean": Path(record["clean_path"]),
                "before": record["quality_defect"],
                "target": target,
            }
        )
    for record in clean_diagnostics:
        source = Path(record["path"])
        target = (
            ROOT
            / "data/repair3d_actions/outputs"
            / "hunyuan"
            / record["sample_id"]
            / "decimate_mesh.glb"
        )
        items.append(
            {
                "kind": "clean",
                "sample_id": record["sample_id"],
                "defect_type": None,
                "intended_action": None,
                "source": source,
                "clean": source,
                "before": record,
                "target": target,
            }
        )

    jobs = []
    for item in items:
        item["target"].parent.mkdir(parents=True, exist_ok=True)
        temporary = item["target"].with_suffix(".candidate.glb")
        temporary.unlink(missing_ok=True)
        item["temporary"] = temporary
        item["source_sha256"] = file_sha256(item["source"])
        jobs.append(
            {
                "source": str(item["source"].resolve()),
                "target": str(temporary.resolve()),
                "ratio": args.ratio,
            }
        )

    decimate_results = run_decimate_jobs(
        jobs, args.blender, Path(__file__).with_name("blender_decimate_batch.py")
    )
    generated = [
        item["temporary"]
        for item in items
        if decimate_results[str(item["source"].resolve())]["success"]
        and item["temporary"].exists()
    ]
    reload_results = blender_checks(
        generated,
        args.blender,
        Path(__file__).with_name("blender_check_glb.py"),
    )

    defect_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    for item in items:
        source_key = str(item["source"].resolve())
        decimate = decimate_results[source_key]
        before = item["before"]
        if decimate["success"] and item["temporary"].exists():
            after = diagnose_glb(item["temporary"], "hunyuan")
            apply_blender_result(
                after, reload_results[str(item["temporary"].resolve())]
            )
            clean_scene = trimesh.load(item["clean"], force="scene", process=False)
            candidate_scene = trimesh.load(
                item["temporary"], force="scene", process=False
            )
            fidelity = geometry_fidelity(clean_scene, candidate_scene)
            accepted, reason, gain = decide_decimation(before, after, fidelity)
        else:
            after = {"trimesh_loadable": False, "quality_score": 0.0}
            fidelity = {}
            accepted, reason, gain = False, "decimate_export_failed", -before["quality_score"]

        row = {
            "sample_id": item["sample_id"],
            "backend": "hunyuan",
            "defect_type": item["defect_type"],
            "intended_action": item["intended_action"],
            "action": "decimate_mesh",
            "clean_path": str(item["clean"].resolve()),
            "source_path": source_key,
            "source_sha256": item["source_sha256"],
            "output_path": str(item["target"].resolve()) if accepted else None,
            "quality_before": before,
            "quality_after": after,
            "quality_gain": gain,
            "fidelity_after": fidelity,
            "accepted": accepted,
            "rollback_reason": None if accepted else reason,
            "action_details": decimate,
        }
        if accepted:
            item["temporary"].replace(item["target"])
            row["quality_after"]["path"] = str(item["target"].resolve())
        else:
            item["temporary"].unlink(missing_ok=True)
        if file_sha256(item["source"]) != item["source_sha256"]:
            raise RuntimeError(f"Source changed during decimation: {item['source']}")
        (defect_rows if item["kind"] == "defect" else clean_rows).append(row)

    existing_phase2 = [
        json.loads(line) for line in args.phase2_trials.open(encoding="utf-8")
    ]
    existing_clean = [json.loads(line) for line in args.clean_trials.open(encoding="utf-8")]
    write_jsonl(
        args.phase2_trials,
        [row for row in existing_phase2 if row["action"] != "decimate_mesh"]
        + defect_rows,
    )
    write_jsonl(
        args.clean_trials,
        [row for row in existing_clean if row["action"] != "decimate_mesh"] + clean_rows,
    )
    print(
        json.dumps(
            {
                "defect_trials": len(defect_rows),
                "clean_trials": len(clean_rows),
                "exports_succeeded": sum(result["success"] for result in decimate_results.values()),
                "accepted": sum(row["accepted"] for row in defect_rows + clean_rows),
                "rolled_back": sum(not row["accepted"] for row in defect_rows + clean_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
