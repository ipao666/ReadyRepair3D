from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import SanaPipeline


MODEL = "/root/r3dguard/models/SANA1.5_1.6B_1024px_diffusers"
OUTPUT_DIR = Path("/root/r3dguard/outputs/sana_gallery")

SAMPLES = [
    {
        "name": "01_robot",
        "seed": 20260715,
        "prompt": (
            "A single premium retro-futuristic toy robot, matte painted metal, "
            "full object visible from head to feet, centered, three-quarter view, "
            "plain light gray studio background, soft diffuse lighting, crisp silhouette, "
            "high geometric clarity, product photography, no text, no extra objects"
        ),
    },
    {
        "name": "02_sneaker",
        "seed": 20260716,
        "prompt": (
            "A single futuristic high-top sneaker, white leather with cobalt blue accents, "
            "the entire shoe visible, centered, three-quarter side view, plain light gray "
            "studio background, soft diffuse lighting, crisp silhouette, detailed sole and "
            "laces, product photography, no text, no extra objects"
        ),
    },
    {
        "name": "03_teapot",
        "seed": 20260717,
        "prompt": (
            "A single elegant ceramic teapot, celadon glaze with subtle carved patterns, "
            "complete handle spout lid and body visible, centered, three-quarter view, "
            "plain light gray studio background, soft diffuse lighting, crisp silhouette, "
            "high geometric clarity, museum product photography, no text, no extra objects"
        ),
    },
    {
        "name": "04_sports_car",
        "seed": 20260718,
        "prompt": (
            "A single compact retro sports car, glossy deep red paint, full vehicle visible, "
            "centered, front three-quarter view, plain light gray studio background, soft "
            "diffuse lighting, crisp silhouette, detailed wheels and body panels, premium "
            "automotive product photography, no driver, no text, no extra objects"
        ),
    },
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe = SanaPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)

    manifest = []
    for sample in SAMPLES:
        generator = torch.Generator(device="cuda").manual_seed(sample["seed"])
        image = pipe(
            prompt=sample["prompt"],
            height=1024,
            width=1024,
            guidance_scale=4.5,
            num_inference_steps=20,
            generator=generator,
        ).images[0]
        output = OUTPUT_DIR / f"{sample['name']}.png"
        image.save(output)
        manifest.append(
            {
                **sample,
                "path": str(output),
                "bytes": output.stat().st_size,
                "size": list(image.size),
                "mode": image.mode,
            }
        )

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
