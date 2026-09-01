from __future__ import annotations

from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


MODEL = "/root/r3dguard/models/SDXL-base-1.0"
OUTPUT = Path("/root/r3dguard/outputs/smoke_sdxl.png")

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    local_files_only=True,
)
pipe.to("cuda")

image = pipe(
    prompt="a single matte ceramic toy robot, full object visible, studio background",
    height=512,
    width=512,
    num_inference_steps=2,
    guidance_scale=5.0,
    generator=torch.Generator(device="cuda").manual_seed(20260714),
).images[0]
image.save(OUTPUT)

print(f"SDXL_SMOKE_OK path={OUTPUT} size={OUTPUT.stat().st_size}")
