from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh


path = Path(sys.argv[1])
scene = trimesh.load(path, force="scene")
geometries = list(scene.geometry.values())

vertices = sum(len(mesh.vertices) for mesh in geometries)
faces = sum(len(mesh.faces) for mesh in geometries)
finite = all(
    np.isfinite(mesh.vertices).all() and np.isfinite(mesh.faces).all()
    for mesh in geometries
)
textured = sum(
    getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None)
    is not None
    for mesh in geometries
)

report = {
    "path": str(path),
    "bytes": path.stat().st_size,
    "geometries": len(geometries),
    "vertices": vertices,
    "faces": faces,
    "finite": finite,
    "textured_geometries": textured,
}
print(json.dumps(report, ensure_ascii=False))

if not geometries or not vertices or not faces or not finite or not textured:
    raise SystemExit(1)
