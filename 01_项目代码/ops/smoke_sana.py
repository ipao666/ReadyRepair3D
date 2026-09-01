from __future__ import annotations

from pathlib import Path

import torch
from diffusers import SanaPipeline


MODEL = "/root/r3dguard/models/SANA1.5_1.6B_1024px_diffusers"
OUTPUT = Path("/root/r3dguard/outputs/sample_sana_1024.png")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

pipe = SanaPipeline.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=False)

generator = torch.Generator(device="cuda").manual_seed(20260714)
image = pipe(
    prompt=(
        "A single matte ceramic toy robot, full object visible, centered, "
        "plain light gray studio background, soft diffuse lighting, "
        "three-quarter view, high geometric clarity"
    ),
    height=1024,
    width=1024,
    guidance_scale=4.5,
    num_inference_steps=20,
    generator=generator,
).images[0]
image.save(OUTPUT)

print(f"SANA_QUALITY_SAMPLE_OK path={OUTPUT} size={OUTPUT.stat().st_size}")
