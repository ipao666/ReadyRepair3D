from __future__ import annotations

import json
import sys
import traceback

import bpy


def check(path: str) -> dict[str, object]:
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=path)
        meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        vertices = sum(len(obj.data.vertices) for obj in meshes)
        polygons = sum(len(obj.data.polygons) for obj in meshes)
        return {
            "path": path,
            "blender_loadable": bool(meshes and vertices and polygons),
            "mesh_objects": len(meshes),
            "vertices": vertices,
            "polygons": polygons,
            "materials": len(bpy.data.materials),
            "images": len(bpy.data.images),
        }
    except Exception as exc:
        return {
            "path": path,
            "blender_loadable": False,
            "mesh_objects": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=3),
        }


arguments = sys.argv[sys.argv.index("--") + 1 :]
for argument in arguments:
    print("REPAIR3D_BLENDER=" + json.dumps(check(argument), ensure_ascii=False), flush=True)
