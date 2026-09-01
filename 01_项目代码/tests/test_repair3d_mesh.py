from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import trimesh


OPS = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS))

from repair3d_mesh import (  # noqa: E402
    cleanup_degenerate_faces,
    diagnose_glb,
    fill_small_holes,
    fix_face_normals,
    geometry_fidelity,
    hard_constraints_hold,
    merge_close_vertices,
    remove_floaters,
)
from repair3d_defects import (  # noqa: E402
    add_degenerate_face,
    add_duplicate_face,
    add_floater,
    duplicate_face_vertices,
    flip_face,
    remove_face_for_small_hole,
)
from run_repair3d_diagnostics import discover  # noqa: E402


def export(scene: trimesh.Scene, path: Path) -> None:
    scene.export(file_obj=path, file_type="glb")


def test_diagnose_and_remove_small_floater(tmp_path: Path) -> None:
    main = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    floater = trimesh.creation.icosphere(subdivisions=0, radius=0.05)
    floater.apply_translation((5.0, 0.0, 0.0))
    scene = trimesh.Scene(trimesh.util.concatenate([main, floater]))
    source = tmp_path / "floaters.glb"
    export(scene, source)

    before = diagnose_glb(source)
    repaired = remove_floaters(scene, min_component_faces=21)
    target = tmp_path / "repaired.glb"
    export(repaired.scene, target)
    after = diagnose_glb(target)

    assert before["connected_components"] == 2
    assert repaired.changed
    assert repaired.details["removed_components"] == 1
    assert after["connected_components"] == 1
    assert source.exists()


def test_cleanup_degenerate_and_duplicate_faces(tmp_path: Path) -> None:
    box = trimesh.creation.box()
    box.faces = np.vstack([box.faces, box.faces[0], [0, 0, 1]])
    scene = trimesh.Scene(box)
    repaired = cleanup_degenerate_faces(scene)
    target = tmp_path / "clean.glb"
    export(repaired.scene, target)
    metrics = diagnose_glb(target)

    assert repaired.changed
    assert repaired.details["removed_degenerate_faces"] >= 1
    assert repaired.details["removed_duplicate_faces"] >= 1
    assert metrics["degenerate_faces"] == 0
    assert metrics["duplicate_faces"] == 0


def test_fix_face_normals_resolves_winding_conflict(tmp_path: Path) -> None:
    box = trimesh.creation.box()
    box.faces[0] = box.faces[0][::-1]
    scene = trimesh.Scene(box)
    repaired = fix_face_normals(scene)
    target = tmp_path / "normals.glb"
    export(repaired.scene, target)
    metrics = diagnose_glb(target)

    assert repaired.changed
    assert repaired.details["normal_conflicts_before"] > 0
    assert repaired.details["normal_conflicts_after"] == 0
    assert metrics["is_winding_consistent"]


def test_hard_constraints_force_texture_loss_rollback() -> None:
    before = {
        "texture_present": True,
        "trimesh_loadable": True,
        "finite": True,
        "faces": 10,
        "geometry_count": 1,
    }
    after = dict(before, texture_present=False)

    accepted, reason = hard_constraints_hold(before, after)

    assert not accepted
    assert reason == "texture_lost"


def test_action_does_not_modify_original_file(tmp_path: Path) -> None:
    source = tmp_path / "original.glb"
    scene = trimesh.Scene(trimesh.creation.box())
    export(scene, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    remove_floaters(trimesh.load(source, force="scene", process=False))

    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def test_default_discovery_labels_only_hunyuan_assets(tmp_path: Path) -> None:
    (tmp_path / "a.glb").touch()
    (tmp_path / "ignore.txt").touch()

    discovered = discover(tmp_path)

    assert discovered == [("hunyuan", tmp_path / "a.glb")]


def test_trellis_discovery_is_explicit_comparison_mode(tmp_path: Path) -> None:
    (tmp_path / "reference.glb").touch()

    discovered = discover(tmp_path, backend="trellis")

    assert discovered == [("trellis", tmp_path / "reference.glb")]


def test_merge_close_vertices_repairs_duplicate_vertex_defect(tmp_path: Path) -> None:
    clean = trimesh.Scene(trimesh.creation.box())
    defective = duplicate_face_vertices(clean).scene
    defect_path = tmp_path / "duplicate_vertices.glb"
    export(defective, defect_path)
    before = diagnose_glb(defect_path)

    repaired = merge_close_vertices(defective)
    repaired_path = tmp_path / "merged.glb"
    export(repaired.scene, repaired_path)
    after = diagnose_glb(repaired_path)

    assert repaired.changed
    assert repaired.details["merged_vertices"] == 3
    assert before["mergeable_duplicate_vertices"] >= 3
    assert after["mergeable_duplicate_vertices"] == 0
    assert after["faces"] == before["faces"]


def test_fill_small_triangle_hole_restores_watertight_mesh(tmp_path: Path) -> None:
    clean = trimesh.Scene(trimesh.creation.box())
    defective = remove_face_for_small_hole(clean).scene
    before_path = tmp_path / "hole.glb"
    export(defective, before_path)
    before = diagnose_glb(before_path)

    repaired = fill_small_holes(defective)
    after_path = tmp_path / "filled.glb"
    export(repaired.scene, after_path)
    after = diagnose_glb(after_path)

    assert before["boundary_edges"] == 3
    assert repaired.changed
    assert repaired.details["added_faces"] == 1
    assert after["boundary_edges"] == 0
    assert after["is_watertight"]


def test_controlled_defects_are_detectable(tmp_path: Path) -> None:
    clean = trimesh.Scene(trimesh.creation.box())
    cases = {
        "floater": (add_floater, "connected_components"),
        "degenerate": (add_degenerate_face, "degenerate_faces"),
        "duplicate_face": (add_duplicate_face, "duplicate_faces"),
        "flipped": (flip_face, "normal_inconsistent_edges"),
        "hole": (remove_face_for_small_hole, "boundary_edges"),
        "duplicate_vertices": (
            duplicate_face_vertices,
            "mergeable_duplicate_vertices",
        ),
    }
    clean_path = tmp_path / "clean.glb"
    export(clean, clean_path)
    baseline = diagnose_glb(clean_path)

    for name, (injector, metric) in cases.items():
        path = tmp_path / f"{name}.glb"
        export(injector(clean).scene, path)
        metrics = diagnose_glb(path)
        assert metrics[metric] > baseline[metric], name


def test_geometry_fidelity_is_zero_for_identical_scene() -> None:
    scene = trimesh.Scene(trimesh.creation.icosphere(subdivisions=1))

    fidelity = geometry_fidelity(scene, scene.copy())

    assert fidelity["normalized_symmetric_chamfer"] < 1e-16
    assert fidelity["bbox_center_shift"] == 0.0
    assert fidelity["bbox_extent_change"] == 0.0
    assert fidelity["area_relative_change"] == 0.0
    assert fidelity["volume_relative_change"] == 0.0


def test_remove_floater_uses_one_percent_face_and_area_limits(tmp_path: Path) -> None:
    main = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    floater = trimesh.creation.icosphere(subdivisions=0, radius=0.03)
    floater.apply_translation((3.0, 0.0, 0.0))
    scene = trimesh.Scene(trimesh.util.concatenate([main, floater]))

    result = remove_floaters(scene)
    target = tmp_path / "face_area_ratio.glb"
    export(result.scene, target)
    after = diagnose_glb(target)

    assert result.changed
    assert result.details["removed_components"] == 1
    assert after["connected_components"] == 1


def test_texture_material_and_uv_survive_low_risk_cleanup(tmp_path: Path) -> None:
    from PIL import Image

    mesh = trimesh.creation.box()
    uv = np.zeros((len(mesh.vertices), 2), dtype=float)
    uv[:, 0] = (mesh.vertices[:, 0] - mesh.vertices[:, 0].min()) / np.ptp(mesh.vertices[:, 0])
    uv[:, 1] = (mesh.vertices[:, 1] - mesh.vertices[:, 1].min()) / np.ptp(mesh.vertices[:, 1])
    material = trimesh.visual.material.PBRMaterial(
        name="kept_material",
        baseColorTexture=Image.new("RGBA", (4, 4), (40, 120, 200, 255)),
    )
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    mesh.faces = np.vstack([mesh.faces, mesh.faces[0]])
    scene = trimesh.Scene(mesh)
    before_path = tmp_path / "textured_before.glb"
    export(scene, before_path)

    cleaned = cleanup_degenerate_faces(scene)
    after_path = tmp_path / "textured_after.glb"
    export(cleaned.scene, after_path)
    after = diagnose_glb(after_path)

    assert cleaned.changed
    assert after["texture_present"]
    assert after["uv_coverage_ratio"] == 1.0
    loaded = trimesh.load(after_path, force="scene", process=False)
    loaded_mesh = next(iter(loaded.geometry.values()))
    assert getattr(loaded_mesh.visual.material, "name", None) == "kept_material"
