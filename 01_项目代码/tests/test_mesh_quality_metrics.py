import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from mesh_quality_metrics import measure_meshes  # noqa: E402


def test_watertight_single_component_has_clean_topology():
    result = measure_meshes([trimesh.creation.box()])
    assert result["component_count_raw"] == 1
    assert result["effective_component_count"] == 1
    assert result["largest_component_area_ratio"] == 1.0
    assert result["debris_area_ratio"] == 0.0
    assert result["boundary_edge_length_ratio"] == 0.0
    assert result["non_manifold_edge_length_ratio"] == 0.0


def test_tiny_detached_triangle_is_counted_as_debris():
    body = trimesh.creation.box(extents=[2, 2, 2])
    fragment = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
    fragment.apply_translation([10, 0, 0])
    result = measure_meshes([body, fragment])
    assert result["component_count_raw"] == 2
    assert result["effective_component_count"] == 1
    assert result["debris_area_ratio"] > 0
    assert 0.99 < result["largest_component_area_ratio"] < 1
    # The tiny dense sphere owns almost all faces, proving face-count ratios are misleading.
    assert result["largest_component_face_ratio"] > 0.99


def test_three_faces_sharing_an_edge_are_non_manifold():
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]]),
        process=False,
    )
    result = measure_meshes([mesh])
    assert result["non_manifold_edge_length_ratio"] > 0
    assert result["boundary_edge_length_ratio"] > 0


def test_textured_surface_ratio_is_area_weighted():
    textured = trimesh.creation.box()
    uv = np.zeros((len(textured.vertices), 2), dtype=float)
    textured.visual = trimesh.visual.texture.TextureVisuals(
        uv=uv,
        image=Image.new("RGB", (4, 4), "red"),
    )
    plain = trimesh.creation.box()
    result = measure_meshes([textured, plain])
    assert 0.49 <= result["textured_surface_ratio"] <= 0.51
