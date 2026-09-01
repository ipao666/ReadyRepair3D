from __future__ import annotations

import json
from importlib import import_module

import torch


report: dict[str, object] = {
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}

if torch.cuda.is_available():
    lhs = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    rhs = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    report["cuda_matmul_finite"] = bool(torch.isfinite(lhs @ rhs).all())

modules = {
    "shape_pipeline": "hy3dshape.pipelines",
    "texture_pipeline": "textureGenPipeline",
    "custom_rasterizer": "custom_rasterizer_kernel",
    "mesh_inpaint": "DifferentiableRenderer.mesh_inpaint_processor",
    "bpy": "bpy",
}
for label, module in modules.items():
    try:
        imported = import_module(module)
        report[label] = "ok"
        if label == "bpy":
            report["blender_version"] = imported.app.version_string
            report["bpy_path"] = imported.__file__
    except Exception as exc:
        report[label] = f"error: {type(exc).__name__}: {exc}"

print(json.dumps(report, ensure_ascii=False, indent=2))

required = [
    "cuda_matmul_finite",
    "shape_pipeline",
    "texture_pipeline",
    "custom_rasterizer",
    "mesh_inpaint",
    "bpy",
]
if not all(report.get(key) is True or report.get(key) == "ok" for key in required):
    raise SystemExit(1)
