from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh
from scipy.spatial import cKDTree


JsonDict = dict[str, Any]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _topology_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    topology = mesh.copy()
    topology.merge_vertices(merge_tex=True, merge_norm=True)
    return topology


def _connected_face_components(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    if len(mesh.faces) == 0:
        return []
    topology = _topology_mesh(mesh)
    return list(
        trimesh.graph.connected_components(
            topology.face_adjacency,
            nodes=np.arange(len(topology.faces)),
            min_len=1,
            engine="scipy",
        )
    )


def _edge_metrics_raw(mesh: trimesh.Trimesh) -> JsonDict:
    if len(mesh.faces) == 0:
        return {
            "boundary_edges": 0,
            "boundary_loops_approx": 0,
            "non_manifold_edges": 0,
            "manifold_edges": 0,
            "normal_inconsistent_edges": 0,
        }

    oriented = np.asarray(mesh.edges, dtype=np.int64)
    sorted_edges = np.sort(oriented, axis=1)
    unique_edges, inverse, counts = np.unique(
        sorted_edges, axis=0, return_inverse=True, return_counts=True
    )
    boundary_mask = counts == 1
    boundary_edges = unique_edges[boundary_mask]

    boundary_loops = 0
    if len(boundary_edges):
        vertices = np.unique(boundary_edges)
        remap = {int(vertex): index for index, vertex in enumerate(vertices)}
        parent = np.arange(len(vertices), dtype=np.int64)

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        for left, right in boundary_edges:
            a, b = find(remap[int(left)]), find(remap[int(right)])
            if a != b:
                parent[b] = a
        boundary_loops = len({find(index) for index in range(len(vertices))})

    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(np.r_[0, counts[:-1]])
    manifold_groups = np.flatnonzero(counts == 2)
    inconsistent = 0
    for group in manifold_groups:
        first = order[starts[group]]
        second = order[starts[group] + 1]
        if np.array_equal(oriented[first], oriented[second]):
            inconsistent += 1

    return {
        "boundary_edges": int(boundary_mask.sum()),
        "boundary_loops_approx": int(boundary_loops),
        "non_manifold_edges": int((counts > 2).sum()),
        "manifold_edges": int(len(manifold_groups)),
        "normal_inconsistent_edges": int(inconsistent),
    }


def _edge_metrics(mesh: trimesh.Trimesh) -> JsonDict:
    return _edge_metrics_raw(_topology_mesh(mesh))


def _has_texture(mesh: trimesh.Trimesh) -> bool:
    material = getattr(mesh.visual, "material", None)
    return bool(
        getattr(material, "baseColorTexture", None) is not None
        or getattr(material, "image", None) is not None
    )


def _mesh_metrics(mesh: trimesh.Trimesh) -> JsonDict:
    face_count = len(mesh.faces)
    topology = _topology_mesh(mesh)
    components = _connected_face_components(mesh)
    component_sizes = [int(len(component)) for component in components]
    area_faces = np.asarray(mesh.area_faces, dtype=float)
    component_areas = [
        float(area_faces[component].sum()) for component in components
    ]
    total_area = max(float(area_faces.sum()), 1e-12)
    edge = _edge_metrics(mesh)

    nondegenerate = np.asarray(mesh.nondegenerate_faces(), dtype=bool)
    unique = np.asarray(mesh.unique_faces(), dtype=bool)
    degenerate_faces = int(face_count - nondegenerate.sum())
    duplicate_faces = int(face_count - unique.sum())
    mergeable = mesh.copy()
    vertices_before_merge = len(mergeable.vertices)
    mergeable.merge_vertices(
        merge_tex=False,
        merge_norm=False,
        digits_vertex=6,
        digits_uv=6,
        digits_norm=4,
    )
    mergeable_duplicate_vertices = vertices_before_merge - len(mergeable.vertices)

    normal_inconsistent = edge["normal_inconsistent_edges"]
    manifold_edges = edge["manifold_edges"]
    normal_inconsistent_ratio = (
        float(normal_inconsistent / manifold_edges) if manifold_edges else 0.0
    )

    reversed_ratio = 0.0
    if face_count:
        offsets = np.asarray(mesh.triangles_center) - np.asarray(mesh.centroid)
        dots = np.einsum("ij,ij->i", np.asarray(mesh.face_normals), offsets)
        valid = np.isfinite(dots) & (np.abs(dots) > 1e-12)
        if valid.any():
            reversed_ratio = float((dots[valid] < 0).mean())

    texture = _has_texture(mesh)
    uv = getattr(mesh.visual, "uv", None)
    uv_valid_faces = 0
    if uv is not None and len(uv) >= len(mesh.vertices) and face_count:
        uv_array = np.asarray(uv)
        face_uv = uv_array[np.asarray(mesh.faces)]
        uv_valid_faces = int(np.isfinite(face_uv).all(axis=(1, 2)).sum())

    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(face_count),
        "component_face_counts": component_sizes,
        "component_area_ratios": [area / total_area for area in component_areas],
        "is_watertight": bool(topology.is_watertight),
        "is_winding_consistent": bool(topology.is_winding_consistent),
        "degenerate_faces": degenerate_faces,
        "duplicate_faces": duplicate_faces,
        "mergeable_duplicate_vertices": int(mergeable_duplicate_vertices),
        "normal_inconsistent_ratio": normal_inconsistent_ratio,
        "reversed_normal_ratio_approx": reversed_ratio,
        "has_texture": texture,
        "uv_valid_faces": uv_valid_faces,
        **edge,
    }


def quality_score(metrics: JsonDict) -> float:
    if not metrics.get("trimesh_loadable") or not metrics.get("finite", False):
        return 0.0
    if metrics.get("blender_loadable") is False:
        return 0.0
    faces = max(int(metrics.get("faces", 0)), 1)
    edges = max(int(metrics.get("edge_instances", 0)), 1)
    components = max(int(metrics.get("connected_components", 0)), 1)

    component_score = float(metrics.get("largest_component_face_ratio", 0.0))
    fragmentation_score = 1.0 / math.sqrt(components)
    boundary_score = 1.0 - min(1.0, metrics.get("boundary_edges", 0) / edges)
    non_manifold_score = 1.0 - min(
        1.0, metrics.get("non_manifold_edges", 0) / edges
    )
    clean_face_score = 1.0 - min(
        1.0,
        (metrics.get("degenerate_faces", 0) + metrics.get("duplicate_faces", 0))
        / faces,
    )
    clean_vertex_score = 1.0 - min(
        1.0,
        metrics.get("mergeable_duplicate_vertices", 0)
        / max(int(metrics.get("vertices", 0)), 1),
    )
    normal_score = 1.0 - min(
        1.0, float(metrics.get("normal_inconsistent_ratio", 0.0))
    )
    self_intersection_score = 1.0 - min(
        1.0, float(metrics.get("self_intersection_proxy_ratio", 0.0))
    )
    texture_score = 1.0 - min(1.0, float(metrics.get("texture_missing_ratio", 1.0)))

    score = (
        0.15 * component_score
        + 0.10 * fragmentation_score
        + 0.10 * boundary_score
        + 0.10 * non_manifold_score
        + 0.10 * clean_face_score
        + 0.05 * clean_vertex_score
        + 0.10 * normal_score
        + 0.10 * self_intersection_score
        + 0.10 * texture_score
        + 0.10 * float(bool(metrics.get("is_watertight")))
    )
    return round(float(np.clip(score, 0.0, 1.0)), 8)


def diagnose_glb(path: Path, backend: str | None = None) -> JsonDict:
    path = path.resolve()
    base: JsonDict = {
        "sample_id": path.stem,
        "backend": backend,
        "path": str(path),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": file_sha256(path) if path.exists() else None,
        "trimesh_loadable": False,
        "blender_loadable": None,
    }
    try:
        scene = trimesh.load(path, force="scene", process=False)
        geometries = list(scene.geometry.values())
        if not geometries:
            raise ValueError("GLB contains no mesh geometry")
        per_geometry = [_mesh_metrics(mesh) for mesh in geometries]
    except Exception as exc:
        base["load_error"] = f"{type(exc).__name__}: {exc}"
        base["quality_score"] = 0.0
        return base

    faces = sum(item["faces"] for item in per_geometry)
    vertices = sum(item["vertices"] for item in per_geometry)
    component_sizes = [
        size for item in per_geometry for size in item["component_face_counts"]
    ]
    component_area_ratios = [
        ratio for item in per_geometry for ratio in item["component_area_ratios"]
    ]
    boundary_edges = sum(item["boundary_edges"] for item in per_geometry)
    non_manifold_edges = sum(item["non_manifold_edges"] for item in per_geometry)
    manifold_edges = sum(item["manifold_edges"] for item in per_geometry)
    inconsistent_edges = sum(
        item["normal_inconsistent_edges"] for item in per_geometry
    )
    degenerate_faces = sum(item["degenerate_faces"] for item in per_geometry)
    duplicate_faces = sum(item["duplicate_faces"] for item in per_geometry)
    mergeable_duplicate_vertices = sum(
        item["mergeable_duplicate_vertices"] for item in per_geometry
    )
    texture_faces = sum(
        item["faces"] if item["has_texture"] else 0 for item in per_geometry
    )
    uv_valid_faces = sum(item["uv_valid_faces"] for item in per_geometry)
    edge_instances = faces * 3
    self_proxy = degenerate_faces + duplicate_faces + non_manifold_edges

    base.update(
        {
            "trimesh_loadable": True,
            "geometry_count": len(per_geometry),
            "vertices": vertices,
            "faces": faces,
            "finite": all(
                np.isfinite(mesh.vertices).all() and np.isfinite(mesh.faces).all()
                for mesh in geometries
            ),
            "connected_components": len(component_sizes),
            "component_face_counts": component_sizes,
            "component_area_ratios": component_area_ratios,
            "largest_component_faces": max(component_sizes, default=0),
            "largest_component_face_ratio": (
                max(component_sizes, default=0) / faces if faces else 0.0
            ),
            "boundary_edges": boundary_edges,
            "hole_count_approx": sum(
                item["boundary_loops_approx"] for item in per_geometry
            ),
            "non_manifold_edges": non_manifold_edges,
            "edge_instances": edge_instances,
            "is_watertight": all(item["is_watertight"] for item in per_geometry),
            "is_winding_consistent": all(
                item["is_winding_consistent"] for item in per_geometry
            ),
            "normal_inconsistent_edges": inconsistent_edges,
            "normal_inconsistent_ratio": (
                inconsistent_edges / manifold_edges if manifold_edges else 0.0
            ),
            "reversed_normal_ratio_approx": (
                sum(
                    item["reversed_normal_ratio_approx"] * item["faces"]
                    for item in per_geometry
                )
                / faces
                if faces
                else 0.0
            ),
            "degenerate_faces": degenerate_faces,
            "duplicate_faces": duplicate_faces,
            "mergeable_duplicate_vertices": mergeable_duplicate_vertices,
            "self_intersection_proxy_count": self_proxy,
            "self_intersection_proxy_ratio": self_proxy / max(faces, 1),
            "texture_present": texture_faces > 0,
            "textured_geometry_count": sum(
                int(item["has_texture"]) for item in per_geometry
            ),
            "texture_missing_ratio": 1.0 - texture_faces / max(faces, 1),
            "uv_coverage_ratio": uv_valid_faces / max(faces, 1),
            "uv_coverage_method": "fraction_of_faces_with_finite_per_vertex_uv",
        }
    )
    base["quality_score"] = quality_score(base)
    return base


def apply_blender_result(metrics: JsonDict, result: JsonDict) -> None:
    metrics["blender_loadable"] = bool(result.get("blender_loadable"))
    metrics["blender_mesh_objects"] = int(result.get("mesh_objects", 0))
    metrics["blender_materials"] = int(result.get("materials", 0))
    metrics["blender_images"] = int(result.get("images", 0))
    if result.get("error"):
        metrics["blender_error"] = result["error"]
    metrics["quality_score"] = quality_score(metrics)


@dataclass
class ActionResult:
    scene: trimesh.Scene
    changed: bool
    details: JsonDict


def remove_floaters(
    scene: trimesh.Scene,
    min_component_faces: int | None = None,
    max_component_ratio: float = 0.01,
    max_component_area_ratio: float = 0.01,
) -> ActionResult:
    result = scene.copy()
    removed_faces = 0
    removed_components = 0
    for name, mesh in list(result.geometry.items()):
        components = _connected_face_components(mesh)
        if len(components) <= 1:
            continue
        face_total = max(len(mesh.faces), 1)
        face_areas = np.asarray(mesh.area_faces, dtype=float)
        total_area = max(float(face_areas.sum()), 1e-12)
        keep = np.zeros(len(mesh.faces), dtype=bool)
        largest_index = int(np.argmax([len(component) for component in components]))
        for index, component in enumerate(components):
            face_ratio = len(component) / face_total
            area_ratio = float(face_areas[component].sum()) / total_area
            below_absolute_limit = (
                min_component_faces is not None
                and len(component) < min_component_faces
            )
            removable = (
                face_ratio < max_component_ratio
                or area_ratio < max_component_area_ratio
                or below_absolute_limit
            )
            if index == largest_index or not removable:
                keep[component] = True
            else:
                removed_faces += len(component)
                removed_components += 1
        if not keep.all():
            mesh.update_faces(keep)
            mesh.remove_unreferenced_vertices()
            result.geometry[name] = mesh
    return ActionResult(
        scene=result,
        changed=removed_faces > 0,
        details={
            "removed_faces": int(removed_faces),
            "removed_components": int(removed_components),
            "component_face_threshold_policy": {
                "minimum": min_component_faces,
                "maximum_total_face_ratio": max_component_ratio,
                "maximum_total_area_ratio": max_component_area_ratio,
                "logic": "remove_non_main_component_when_face_or_area_ratio_below_limit",
            },
        },
    )


def cleanup_degenerate_faces(scene: trimesh.Scene) -> ActionResult:
    result = scene.copy()
    removed_degenerate = 0
    removed_duplicate = 0
    for name, mesh in list(result.geometry.items()):
        nondegenerate = np.asarray(mesh.nondegenerate_faces(), dtype=bool)
        unique = np.asarray(mesh.unique_faces(), dtype=bool)
        removed_degenerate += int(len(mesh.faces) - nondegenerate.sum())
        removed_duplicate += int(len(mesh.faces) - unique.sum())
        keep = nondegenerate & unique
        if not keep.all():
            mesh.update_faces(keep)
            mesh.remove_unreferenced_vertices()
            result.geometry[name] = mesh
    removed = removed_degenerate + removed_duplicate
    return ActionResult(
        scene=result,
        changed=removed > 0,
        details={
            "removed_degenerate_faces": removed_degenerate,
            "removed_duplicate_faces": removed_duplicate,
        },
    )


def fix_face_normals(scene: trimesh.Scene) -> ActionResult:
    result = scene.copy()
    before_conflicts = 0
    after_conflicts = 0
    for name, mesh in list(result.geometry.items()):
        before_conflicts += _edge_metrics(mesh)["normal_inconsistent_edges"]
        topology = _topology_mesh(mesh)
        faces_before = topology.faces.copy()
        trimesh.repair.fix_normals(topology, multibody=True)
        flipped = np.all(topology.faces == faces_before[:, ::-1], axis=1)
        if flipped.any():
            faces = mesh.faces.copy()
            faces[flipped] = faces[flipped, ::-1]
            mesh.faces = faces
        after_conflicts += _edge_metrics(mesh)["normal_inconsistent_edges"]
        result.geometry[name] = mesh
    return ActionResult(
        scene=result,
        changed=after_conflicts < before_conflicts,
        details={
            "normal_conflicts_before": before_conflicts,
            "normal_conflicts_after": after_conflicts,
        },
    )


def merge_close_vertices(scene: trimesh.Scene) -> ActionResult:
    result = scene.copy()
    merged_vertices = 0
    for name, mesh in list(result.geometry.items()):
        before = len(mesh.vertices)
        mesh.merge_vertices(
            merge_tex=False,
            merge_norm=False,
            digits_vertex=6,
            digits_uv=6,
            digits_norm=4,
        )
        merged_vertices += before - len(mesh.vertices)
        result.geometry[name] = mesh
    return ActionResult(
        scene=result,
        changed=merged_vertices > 0,
        details={
            "merged_vertices": int(merged_vertices),
            "position_digits": 6,
            "uv_digits": 6,
            "normal_digits": 4,
        },
    )


def fill_small_holes(scene: trimesh.Scene) -> ActionResult:
    result = scene.copy()
    added_faces = 0
    closed_boundary_edges = 0
    for name, mesh in list(result.geometry.items()):
        topology = _topology_mesh(mesh)
        faces_before = len(topology.faces)
        boundary_before = _edge_metrics_raw(topology)["boundary_edges"]
        trimesh.repair.fill_holes(topology)
        new_faces = np.asarray(topology.faces[faces_before:], dtype=np.int64)
        if len(new_faces) == 0:
            continue

        representatives: dict[tuple[float, float, float], int] = {}
        for index, vertex in enumerate(np.asarray(mesh.vertices)):
            representatives.setdefault(tuple(float(value) for value in vertex), index)
        mapped_faces = []
        for face in new_faces:
            mapped = []
            for vertex_index in face:
                key = tuple(float(value) for value in topology.vertices[vertex_index])
                if key not in representatives:
                    mapped = []
                    break
                mapped.append(representatives[key])
            if len(mapped) == 3:
                mapped_faces.append(mapped)
        if not mapped_faces:
            continue

        face_colors = None
        if mesh.visual.defined and mesh.visual.kind == "face":
            face_colors = np.asarray(mesh.visual.face_colors).copy()
        mesh.faces = np.vstack((mesh.faces, np.asarray(mapped_faces, dtype=np.int64)))
        if face_colors is not None:
            mesh.visual.face_colors = np.vstack(
                (face_colors, np.tile(face_colors[-1], (len(mapped_faces), 1)))
            )
        added_faces += len(mapped_faces)
        closed_boundary_edges += max(
            0,
            boundary_before - _edge_metrics_raw(topology)["boundary_edges"],
        )
        result.geometry[name] = mesh
    return ActionResult(
        scene=result,
        changed=added_faces > 0 and closed_boundary_edges > 0,
        details={
            "added_faces": int(added_faces),
            "closed_boundary_edges": int(closed_boundary_edges),
            "supported_hole_edges": [3, 4],
        },
    )


def _scene_world_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    meshes: list[trimesh.Trimesh] = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if not meshes:
        raise ValueError("Scene contains no mesh geometry")
    return trimesh.util.concatenate(meshes)


def _comparison_points(mesh: trimesh.Trimesh, maximum: int = 4096) -> np.ndarray:
    points = np.vstack((np.asarray(mesh.vertices), np.asarray(mesh.triangles_center)))
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, num=maximum, dtype=np.int64)
    return points[indices]


def geometry_fidelity(reference: trimesh.Scene, candidate: trimesh.Scene) -> JsonDict:
    reference_mesh = _scene_world_mesh(reference)
    candidate_mesh = _scene_world_mesh(candidate)
    reference_topology = _topology_mesh(reference_mesh)
    candidate_topology = _topology_mesh(candidate_mesh)

    reference_points = _comparison_points(reference_mesh)
    candidate_points = _comparison_points(candidate_mesh)
    ref_to_candidate = cKDTree(candidate_points).query(reference_points, k=1)[0]
    candidate_to_ref = cKDTree(reference_points).query(candidate_points, k=1)[0]
    diagonal = max(float(np.linalg.norm(reference_mesh.extents)), 1e-12)
    chamfer = float(
        (np.square(ref_to_candidate).mean() + np.square(candidate_to_ref).mean())
        / (diagonal * diagonal)
    )

    reference_center = np.asarray(reference_mesh.bounds).mean(axis=0)
    candidate_center = np.asarray(candidate_mesh.bounds).mean(axis=0)
    center_shift = float(np.linalg.norm(candidate_center - reference_center) / diagonal)
    extent_change = float(
        np.linalg.norm(candidate_mesh.extents - reference_mesh.extents) / diagonal
    )
    area_change = abs(float(candidate_mesh.area - reference_mesh.area)) / max(
        float(reference_mesh.area), 1e-12
    )
    reference_volume = abs(float(reference_topology.volume))
    candidate_volume = abs(float(candidate_topology.volume))
    volume_change = (
        abs(candidate_volume - reference_volume) / reference_volume
        if reference_volume > 1e-12
        else None
    )
    return {
        "normalized_symmetric_chamfer": chamfer,
        "bbox_center_shift": center_shift,
        "bbox_extent_change": extent_change,
        "area_relative_change": area_change,
        "volume_relative_change": volume_change,
        "reference_watertight": bool(reference_topology.is_watertight),
        "candidate_watertight": bool(candidate_topology.is_watertight),
        "point_count_per_side": 4096,
    }


def fidelity_constraints_hold(fidelity: JsonDict) -> tuple[bool, str | None]:
    thresholds = {
        "normalized_symmetric_chamfer": 0.01,
        "bbox_center_shift": 0.01,
        "bbox_extent_change": 0.03,
        "area_relative_change": 0.05,
        "volume_relative_change": 0.05,
    }
    for metric, threshold in thresholds.items():
        value = fidelity.get(metric)
        if value is not None and value > threshold:
            return False, f"fidelity_{metric}"
    return True, None


ACTIONS: dict[str, Callable[[trimesh.Scene], ActionResult]] = {
    "remove_floaters": remove_floaters,
    "cleanup_degenerate_faces": cleanup_degenerate_faces,
    "fix_face_normals": fix_face_normals,
    "merge_close_vertices": merge_close_vertices,
    "fill_small_holes": fill_small_holes,
}


def hard_constraints_hold(before: JsonDict, after: JsonDict) -> tuple[bool, str | None]:
    if not after.get("trimesh_loadable"):
        return False, "trimesh_reload_failed"
    if not after.get("finite") or after.get("faces", 0) <= 0:
        return False, "invalid_or_empty_geometry"
    if before.get("texture_present") and not after.get("texture_present"):
        return False, "texture_lost"
    if int(before.get("blender_materials", 0)) > int(
        after.get("blender_materials", 0)
    ):
        return False, "material_lost"
    if int(before.get("blender_images", 0)) > int(after.get("blender_images", 0)):
        return False, "texture_image_lost"
    if (
        float(before.get("uv_coverage_ratio", 0.0)) > 0.0
        and float(after.get("uv_coverage_ratio", 0.0))
        < float(before.get("uv_coverage_ratio", 0.0)) - 0.001
    ):
        return False, "uv_coverage_decreased"
    if after.get("geometry_count", 0) <= 0:
        return False, "geometry_lost"
    if float(after.get("largest_component_face_ratio", 0.0)) + 0.01 < float(
        before.get("largest_component_face_ratio", 0.0)
    ):
        return False, "main_component_ratio_decreased"
    for metric in (
        "non_manifold_edges",
        "degenerate_faces",
        "duplicate_faces",
        "normal_inconsistent_edges",
    ):
        if int(after.get(metric, 0)) > int(before.get(metric, 0)):
            return False, f"{metric}_increased"
    if int(after.get("connected_components", 0)) > int(
        before.get("connected_components", 0)
    ):
        return False, "floating_components_increased"
    return True, None


def write_jsonl(path: Path, records: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)
