from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import trimesh


@dataclass
class DefectResult:
    scene: trimesh.Scene
    details: dict[str, object]
    intended_action: str


def _primary_geometry(scene: trimesh.Scene) -> tuple[str, trimesh.Trimesh]:
    name = max(scene.geometry, key=lambda key: len(scene.geometry[key].faces))
    return name, scene.geometry[name]


def _append_vertices(
    mesh: trimesh.Trimesh,
    vertices: np.ndarray,
    source_vertex_indices: np.ndarray,
) -> np.ndarray:
    start = len(mesh.vertices)
    uv = getattr(mesh.visual, "uv", None)
    uv_before = np.asarray(uv).copy() if uv is not None else None
    mesh.vertices = np.vstack((mesh.vertices, vertices))
    if uv_before is not None and len(uv_before) == start:
        mesh.visual.uv = np.vstack((uv_before, uv_before[source_vertex_indices]))
    return np.arange(start, start + len(vertices), dtype=np.int64)


def add_floater(scene: trimesh.Scene) -> DefectResult:
    result = scene.copy()
    name, mesh = _primary_geometry(result)
    face = np.asarray(mesh.faces[0])
    source = np.asarray(mesh.vertices[face])
    center = source.mean(axis=0)
    diagonal = max(float(np.linalg.norm(mesh.extents)), 1e-6)
    target = np.asarray(mesh.bounds[1]) + np.array([0.03 * diagonal, 0.0, 0.0])
    floater_vertices = (source - center) * 0.05 + target
    added = _append_vertices(mesh, floater_vertices, face)
    mesh.faces = np.vstack((mesh.faces, added))
    result.geometry[name] = mesh
    return DefectResult(
        result,
        {"added_faces": 1, "added_vertices": 3},
        "remove_floaters",
    )


def add_degenerate_face(scene: trimesh.Scene) -> DefectResult:
    result = scene.copy()
    name, mesh = _primary_geometry(result)
    face = np.asarray(mesh.faces[0])
    mesh.faces = np.vstack((mesh.faces, [face[0], face[0], face[1]]))
    result.geometry[name] = mesh
    return DefectResult(result, {"added_degenerate_faces": 1}, "cleanup_degenerate_faces")


def add_duplicate_face(scene: trimesh.Scene) -> DefectResult:
    result = scene.copy()
    name, mesh = _primary_geometry(result)
    face = np.asarray(mesh.faces[0])
    mesh.faces = np.vstack((mesh.faces, face[::-1]))
    result.geometry[name] = mesh
    return DefectResult(
        result,
        {"added_reversed_duplicate_faces": 1},
        "cleanup_degenerate_faces",
    )


def flip_face(scene: trimesh.Scene) -> DefectResult:
    result = scene.copy()
    name, mesh = _primary_geometry(result)
    faces = mesh.faces.copy()
    faces[0] = faces[0][::-1]
    mesh.faces = faces
    result.geometry[name] = mesh
    return DefectResult(result, {"flipped_faces": 1}, "fix_face_normals")


def remove_face_for_small_hole(scene: trimesh.Scene) -> DefectResult:
    result = scene.copy()
    name, mesh = _primary_geometry(result)
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[0] = False
    mesh.update_faces(keep)
    result.geometry[name] = mesh
    return DefectResult(result, {"removed_faces": 1}, "fill_small_holes")


def duplicate_face_vertices(scene: trimesh.Scene) -> DefectResult:
    result = scene.copy()
    name, mesh = _primary_geometry(result)
    faces = mesh.faces.copy()
    face = np.asarray(faces[0])
    added = _append_vertices(mesh, np.asarray(mesh.vertices[face]).copy(), face)
    faces[0] = added
    mesh.faces = faces
    result.geometry[name] = mesh
    return DefectResult(
        result,
        {"duplicated_vertices": 3},
        "merge_close_vertices",
    )


DEFECTS: dict[str, Callable[[trimesh.Scene], DefectResult]] = {
    "floater": add_floater,
    "degenerate_face": add_degenerate_face,
    "non_manifold_duplicate_face": add_duplicate_face,
    "flipped_face": flip_face,
    "small_triangle_hole": remove_face_for_small_hole,
    "duplicate_vertices": duplicate_face_vertices,
}
