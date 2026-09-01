from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image


MODEL = "/root/r3dguard/models/SANA1.5_1.6B_1024px_diffusers"
OUTPUT_DIR = Path("/root/r3dguard/data/ai_sana_200")
BASE_SEED = 2026071500

CATALOG = [
    ("kitchenware", "easy", "an elegant celadon ceramic teapot with a curved handle and short spout"),
    ("kitchenware", "easy", "a brushed stainless steel electric kettle with a black handle"),
    ("kitchenware", "medium", "a glossy red countertop stand mixer with a metal mixing bowl"),
    ("kitchenware", "easy", "a compact retro cream-colored two-slot toaster"),
    ("kitchenware", "medium", "a vintage manual coffee grinder with a wooden base and metal crank"),
    ("electronics", "medium", "a vintage black rangefinder camera with a silver lens"),
    ("electronics", "easy", "a compact 1960s portable radio with rounded corners and two knobs"),
    ("electronics", "medium", "a futuristic white and blue wireless game controller"),
    ("electronics", "hard", "a pair of premium over-ear headphones with a padded headband"),
    ("electronics", "easy", "a compact matte black desktop video projector with a large lens"),
    ("tools", "medium", "a cordless power drill with a red body and black battery pack"),
    ("tools", "easy", "a heavy blue cast-iron bench vise with closed jaws"),
    ("tools", "medium", "a vintage metal camping lantern with a protective wire cage"),
    ("tools", "medium", "a classic black mechanical sewing machine on a small wooden base"),
    ("tools", "hard", "a compact orange chainsaw with a complete guide bar and handles"),
    ("furniture", "easy", "a modern mustard-yellow upholstered armchair with wooden legs"),
    ("furniture", "medium", "a sculpted walnut wooden bar stool with four legs"),
    ("furniture", "hard", "a black ergonomic office swivel chair with armrests and five wheels"),
    ("furniture", "easy", "a small mid-century walnut bedside cabinet with one drawer"),
    ("furniture", "hard", "an adjustable green metal desk lamp with an articulated arm"),
    ("vehicles", "medium", "a compact retro deep-red sports car with detailed wheels"),
    ("vehicles", "easy", "a small cream-colored vintage delivery van with closed doors"),
    ("vehicles", "hard", "a green agricultural tractor with large rear tires"),
    ("vehicles", "hard", "a yellow tracked bulldozer with a raised front blade"),
    ("vehicles", "hard", "a turquoise 1960s motor scooter with a black seat"),
    ("wearables", "medium", "a futuristic white high-top sneaker with cobalt blue accents"),
    ("wearables", "medium", "a rugged brown leather hiking boot with thick tread"),
    ("wearables", "easy", "a glossy white full-face motorcycle helmet with a dark visor"),
    ("wearables", "easy", "a compact teal hard-shell travel suitcase with four wheels"),
    ("wearables", "hard", "a stainless steel analog wristwatch with a blue dial and metal bracelet"),
    ("decor", "easy", "a tall cobalt blue ceramic vase with subtle carved patterns"),
    ("decor", "medium", "an art deco brass mantel clock with a round white face"),
    ("decor", "hard", "a vintage tabletop globe on a dark wooden stand"),
    ("decor", "medium", "an art deco table lamp with a frosted glass dome shade"),
    ("decor", "medium", "a stylized bronze owl figurine with geometric feathers"),
    ("toys", "medium", "a premium retro-futuristic matte metal toy robot"),
    ("toys", "hard", "a detailed red and black toy steam locomotive"),
    ("toys", "medium", "a compact silver retro toy spaceship with three landing legs"),
    ("toys", "hard", "a green articulated toy tyrannosaurus dinosaur figure"),
    ("toys", "hard", "a small yellow toy excavator with tracks and a raised bucket"),
    ("containers", "easy", "a sturdy red metal toolbox with a closed lid and top handle"),
    ("containers", "easy", "a rounded retro blue metal lunchbox with a folding handle"),
    ("containers", "medium", "a classic green metal watering can with a long spout"),
    ("containers", "medium", "a red fire extinguisher with a black hose and handle"),
    ("containers", "easy", "a gray industrial storage bin with a fitted lid"),
    ("challenging", "hard", "a natural wood acoustic guitar with all strings and tuning pegs visible"),
    ("challenging", "hard", "a vintage laboratory microscope with brass adjustment knobs"),
    ("challenging", "hard", "a blue city bicycle with thin wheels, pedals, and handlebars"),
    ("challenging", "hard", "a retro metal pedestal fan with a circular wire grille"),
    ("challenging", "hard", "a polished brass alto saxophone with visible keys and curved bell"),
]


def full_prompt(subject: str) -> str:
    return (
        f"A single {subject}, full object visible and entirely inside the frame, centered, "
        "three-quarter product view, plain light gray seamless studio background, soft diffuse "
        "lighting, crisp clean silhouette, realistic materials, high geometric clarity, "
        "high-end product photography, no people, no hands, no text, no logo, no watermark, "
        "no extra objects, no cropped parts"
    )


def expected_records(limit_groups: int) -> list[dict]:
    records = []
    for group_index, (category, difficulty, subject) in enumerate(CATALOG[:limit_groups]):
        group_id = f"sana_{group_index:03d}"
        prompt = full_prompt(subject)
        for candidate_index in range(4):
            seed = BASE_SEED + group_index * 10 + candidate_index
            filename = f"{group_id}_c{candidate_index}_s{seed}.png"
            records.append(
                {
                    "group_id": group_id,
                    "candidate_index": candidate_index,
                    "category": category,
                    "difficulty": difficulty,
                    "subject": subject,
                    "prompt": prompt,
                    "seed": seed,
                    "filename": filename,
                }
            )
    return records


def valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.load()
            return image.size == (1024, 1024) and image.mode == "RGB"
    except (OSError, ValueError):
        return False


def atomic_json(path: Path, payload: object, *, jsonl: bool = False) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if jsonl:
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload)
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-groups", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_groups <= len(CATALOG):
        raise ValueError(f"max-groups must be between 1 and {len(CATALOG)}")

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records = expected_records(args.max_groups)
    missing = [row for row in records if not valid_image(images_dir / row["filename"])]
    started = time.perf_counter()

    if missing:
        import torch
        from diffusers import SanaPipeline

        pipe = SanaPipeline.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)

        for completed, row in enumerate(missing, start=1):
            output = images_dir / row["filename"]
            if output.exists():
                output.unlink()
            generator = torch.Generator(device="cuda").manual_seed(row["seed"])
            image = pipe(
                prompt=row["prompt"],
                height=1024,
                width=1024,
                guidance_scale=4.5,
                num_inference_steps=20,
                generator=generator,
            ).images[0].convert("RGB")
            image.save(output)
            if not valid_image(output):
                raise RuntimeError(f"generated image failed validation: {output}")
            print(
                json.dumps(
                    {
                        "event": "generated",
                        "completed_missing": completed,
                        "total_missing": len(missing),
                        "group_id": row["group_id"],
                        "candidate_index": row["candidate_index"],
                        "path": str(output),
                    }
                ),
                flush=True,
            )

    finished_records = []
    for row in records:
        path = images_dir / row["filename"]
        if not valid_image(path):
            raise RuntimeError(f"missing or invalid output: {path}")
        finished_records.append(
            {
                **row,
                "model": "SANA1.5_1.6B_1024px_diffusers",
                "width": 1024,
                "height": 1024,
                "guidance_scale": 4.5,
                "num_inference_steps": 20,
                "path": str(path),
                "bytes": path.stat().st_size,
            }
        )

    elapsed = time.perf_counter() - started
    atomic_json(args.output_dir / "manifest.jsonl", finished_records, jsonl=True)
    atomic_json(
        args.output_dir / "summary.json",
        {
            "groups": args.max_groups,
            "images": len(finished_records),
            "generated_this_run": len(missing),
            "elapsed_seconds": round(elapsed, 2),
            "output_dir": str(args.output_dir),
        },
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "groups": args.max_groups,
                "images": len(finished_records),
                "generated_this_run": len(missing),
                "elapsed_seconds": round(elapsed, 2),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
