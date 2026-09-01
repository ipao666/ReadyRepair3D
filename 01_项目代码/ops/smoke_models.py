from __future__ import annotations

import gc
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForDepthEstimation, Sam2Model


ROOT = Path("/root/r3dguard/models")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


tests = (
    ("dinov2-large", AutoModel),
    ("Depth-Anything-V2-Large-hf", AutoModelForDepthEstimation),
    ("sam2-hiera-large", Sam2Model),
)

for directory, model_class in tests:
    path = ROOT / directory
    model = model_class.from_pretrained(
        path,
        local_files_only=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print(f"LOAD_OK {directory} params={count_parameters(model)}")
    del model
    gc.collect()

print("MODEL_SMOKE_OK")
