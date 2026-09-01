"""Deterministic topology and texture-coverage metrics for GLB assets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh


def _has_valid_texture(mesh: trimesh.Trimesh) -> bool:
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    if uv is None or len(uv) != len(mesh.vertices) or material is None:
        return False
    image = getattr(material, "image", None)
    base_color = getattr(material, "baseColorTexture", None)
    return image is not None or base_color is not None


def _face_components(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    if len(mesh.faces) == 0:
        return []
    topology = mesh.copy()
    topology.merge_vertices(merge_tex=True, merge_norm=True)
    return list(
        trimesh.graph.connected_components(
            topology.face_adjacency,
            nodes=np.arange(len(topology.faces)),
            min_len=1,
            engine="scipy",
        )
    )


def measure_meshes(meshes: Iterable[trimesh.Trimesh], debris_fraction: float = 0.005) -> dict:
    geometries = list(meshes)
    if not geometries:
        raise ValueError("Asset contains no mesh geometry")
    faces = sum(len(mesh.faces) for mesh in geometries)
    vertices = sum(len(mesh.vertices) for mesh in geometries)
    if faces <= 0:
        raise ValueError("Asset contains no faces")
    finite = all(
        np.isfinite(mesh.vertices).all() and np.isfinite(mesh.faces).all()
        for mesh in geometries
    )

    components = [
        (len(component), float(mesh.area_faces[component].sum()))
        for mesh in geometries
        for component in _face_components(mesh)
    ]
    component_face_sizes = [component[0] for component in components]

    boundary_edges = 0
    non_manifold_edges = 0
    unique_edges = 0
    boundary_edge_length = 0.0
    non_manifold_edge_length = 0.0
    total_edge_length = 0.0
    for mesh in geometries:
        if len(mesh.faces) == 0:
            continue
        edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
        lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
        unique_edges += len(counts)
        boundary_edges += int(np.sum(counts == 1))
        non_manifold_edges += int(np.sum(counts > 2))
        total_edge_length += float(lengths.sum())
        boundary_edge_length += float(lengths[counts == 1].sum())
        non_manifold_edge_length += float(lengths[counts > 2].sum())

    total_area = float(sum(float(mesh.area) for mesh in geometries))
    textured_area = float(
        sum(float(mesh.area) for mesh in geometries if _has_valid_texture(mesh))
    )
    component_areas = [component[1] for component in components]
    area_cutoff = debris_fraction * total_area
    face_cutoff = debris_fraction * faces
    debris_area = sum(area for area in component_areas if area < area_cutoff)
    debris_faces = sum(size for size in component_face_sizes if size < face_cutoff)
    return {
        "finite": bool(finite),
        "vertices": int(vertices),
        "faces": int(faces),
        "component_count_raw": len(components),
        "effective_component_count": int(sum(area >= area_cutoff for area in component_areas)),
        "largest_component_area_ratio": max(component_areas, default=0.0) / total_area,
        "debris_area_ratio": debris_area / total_area,
        "largest_component_face_ratio": max(component_face_sizes, default=0) / faces,
        "debris_face_ratio": debris_faces / faces,
        "boundary_edge_ratio": boundary_edges / unique_edges if unique_edges else 1.0,
        "non_manifold_ratio": non_manifold_edges / unique_edges if unique_edges else 1.0,
        "boundary_edge_length_ratio": (
            boundary_edge_length / total_edge_length if total_edge_length else 1.0
        ),
        "non_manifold_edge_length_ratio": (
            non_manifold_edge_length / total_edge_length if total_edge_length else 1.0
        ),
        "textured_surface_ratio": textured_area / total_area if total_area > 0 else 0.0,
    }


def inspect_glb(path: Path) -> dict:
    path = Path(path)
    try:
        scene = trimesh.load(path, force="scene", process=False)
    except Exception as error:
        return {
            "glb_loadable": False,
            "load_error": f"{type(error).__name__}: {error}",
        }
    result = measure_meshes(scene.geometry.values())
    result.update({"glb_loadable": True, "glb_bytes": path.stat().st_size})
    return result
