from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import torch


def check_import(name: str) -> None:
    importlib.import_module(name)
    print(f"IMPORT_OK {name}")


print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"gpu_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"gpu_name={torch.cuda.get_device_name(0)}")
    a = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    b = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    c = a @ b
    torch.cuda.synchronize()
    print(f"cuda_matmul_finite={bool(torch.isfinite(c).all().item())}")

for module_name in (
    "flash_attn",
    "nvdiffrast.torch",
    "nvdiffrec_render",
    "cumesh",
    "flex_gemm",
    "o_voxel",
    "trellis2",
):
    check_import(module_name)

models_root = Path("/root/r3dguard/models")
required_files = {
    "TRELLIS.2-4B": "pipeline.json",
    "TRELLIS-image-large": "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    "dinov2-large": "model.safetensors",
    "Depth-Anything-V2-Large-hf": "model.safetensors",
    "sam2-hiera-large": "model.safetensors",
}
for model_dir, relative_file in required_files.items():
    path = models_root / model_dir / relative_file
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing model artifact: {path}")
    print(f"MODEL_FILE_OK {model_dir}/{relative_file} {path.stat().st_size}")

pipeline_config = json.loads(
    (models_root / "TRELLIS.2-4B" / "pipeline.json").read_text(encoding="utf-8")
)
print(f"pipeline_name={pipeline_config['name']}")
print("VERIFY_OK")
