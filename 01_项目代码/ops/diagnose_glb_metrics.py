from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


path = Path("/root/r3dguard/data/trellis_16_test/glb/sana_000_c0_s2026071500.glb")
print({"step": "start", "bytes": path.stat().st_size}, flush=True)
scene = trimesh.load(path, force="scene")
print({"step": "loaded", "geometries": len(scene.geometry)}, flush=True)
geometries = list(scene.geometry.values())
print(
    {
        "step": "finite",
        "finite": all(np.isfinite(mesh.vertices).all() and np.isfinite(mesh.faces).all() for mesh in geometries),
    },
    flush=True,
)
print(
    {
        "step": "texture",
        "textured": sum(
            getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None) is not None
            for mesh in geometries
        ),
    },
    flush=True,
)
print(
    {"step": "watertight", "values": [bool(mesh.is_watertight) for mesh in geometries]},
    flush=True,
)
print(
    {"step": "face_adjacency", "sizes": [len(mesh.face_adjacency) for mesh in geometries]},
    flush=True,
)
component_counts = []
for mesh in geometries:
    topology = mesh.copy()
    topology.merge_vertices(merge_tex=True, merge_norm=True)
    components = trimesh.graph.connected_components(
        topology.face_adjacency,
        nodes=np.arange(len(topology.faces)),
        min_len=1,
        engine="scipy",
    )
    component_counts.append(len(components))
print({"step": "component_count", "values": component_counts}, flush=True)
print({"step": "complete"}, flush=True)
