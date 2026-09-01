from __future__ import annotations

import json
import sys

import bpy


path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=path)

meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
report = {
    "path": path,
    "mesh_objects": len(meshes),
    "vertices": sum(len(obj.data.vertices) for obj in meshes),
    "polygons": sum(len(obj.data.polygons) for obj in meshes),
    "materials": len(bpy.data.materials),
    "images": len(bpy.data.images),
}
print("BLENDER_GLB_REPORT=" + json.dumps(report, ensure_ascii=False))

if not meshes or not report["vertices"] or not report["polygons"]:
    raise SystemExit(1)
if not report["materials"] or not report["images"]:
    raise SystemExit(1)
