from __future__ import annotations

import json
import sys
import traceback

import bpy


def decimate(job: dict[str, object]) -> dict[str, object]:
    source = str(job["source"])
    target = str(job["target"])
    ratio = float(job["ratio"])
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=source)
        meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        faces_before = sum(len(obj.data.polygons) for obj in meshes)
        for obj in meshes:
            if len(obj.data.polygons) < 8:
                continue
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            modifier = obj.modifiers.new(name="Repair3DDecimate", type="DECIMATE")
            modifier.decimate_type = "COLLAPSE"
            modifier.ratio = ratio
            modifier.use_collapse_triangulate = True
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            obj.select_set(False)
        faces_after = sum(len(obj.data.polygons) for obj in meshes)
        bpy.ops.export_scene.gltf(
            filepath=target,
            export_format="GLB",
            export_texcoords=True,
            export_normals=True,
            export_materials="EXPORT",
            export_image_format="AUTO",
        )
        return {
            "source": source,
            "target": target,
            "success": True,
            "faces_before": faces_before,
            "faces_after": faces_after,
            "ratio": ratio,
        }
    except Exception as exc:
        return {
            "source": source,
            "target": target,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=4),
        }


jobs_path = sys.argv[sys.argv.index("--") + 1]
with open(jobs_path, encoding="utf-8") as handle:
    jobs = json.load(handle)
for current_job in jobs:
    print(
        "REPAIR3D_DECIMATE=" + json.dumps(decimate(current_job), ensure_ascii=False),
        flush=True,
    )
