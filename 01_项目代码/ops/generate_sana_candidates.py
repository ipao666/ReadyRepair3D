#!/usr/bin/env python3
"""Generate a resumable SANA Best-of-4 set for one object prompt."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image


MODEL = Path("/root/r3dguard/models/SANA1.5_1.6B_1024px_diffusers")


def full_prompt(subject: str) -> str:
    subject = " ".join(subject.strip().split())
    if not subject:
        raise ValueError("prompt must not be empty")
    return (
        f"A single {subject}, full object visible and entirely inside the frame, "
        "centered, three-quarter product view, generous margin around the object, "
        "plain light gray seamless studio background, soft diffuse lighting, "
        "crisp clean silhouette, single connected solid object, realistic materials, "
        "high geometric clarity, high-end product photography, no people, no hands, "
        "no text, no logo, no watermark, no extra objects, no cropped parts"
    )


def candidate_records(subject: str, base_seed: int) -> list[dict]:
    group_id = f"request_{int(base_seed)}"
    prompt = full_prompt(subject)
    return [
        {
            "schema_version": "r3dguard.pipeline-candidate.v1",
            "sample_id": f"{group_id}_c{index}",
            "group_id": group_id,
            "candidate_index": index,
            "subject": subject,
            "prompt": prompt,
            "seed": int(base_seed) + index,
            "filename": f"{group_id}_c{index}_s{int(base_seed) + index}.png",
            "domain": "ai_sana_pipeline",
            "split": "inference",
            "generator": "SANA1.5_1.6B_1024px_diffusers",
        }
        for index in range(4)
    ]


def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return path.stat().st_size > 0
    except (FileNotFoundError, OSError):
        return False


def finished_record(row: dict, path: Path) -> dict:
    if not valid_image(path):
        raise ValueError(f"invalid candidate image: {path}")
    with Image.open(path) as image:
        width, height = image.size
    return {
        **row,
        "width": width,
        "height": height,
        "guidance_scale": 4.5,
        "num_inference_steps": 20,
        "path": str(path.resolve()),
        "source_path": str(path.resolve()),
        "bytes": path.stat().st_size,
    }


def atomic_json(path: Path, payload: object, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        if jsonl:
            for row in payload:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = candidate_records(args.prompt, args.seed)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "planned_manifest.jsonl", rows, jsonl=True)
    missing = [row for row in rows if not valid_image(images_dir / row["filename"])]
    started = time.perf_counter()
    if missing:
        import torch
        from diffusers import SanaPipeline

        pipeline = SanaPipeline.from_pretrained(
            str(MODEL), torch_dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        pipeline.set_progress_bar_config(disable=True)
        for completed, row in enumerate(missing, 1):
            target = images_dir / row["filename"]
            target.unlink(missing_ok=True)
            generator = torch.Generator(device="cuda").manual_seed(row["seed"])
            image = pipeline(
                prompt=row["prompt"], height=1024, width=1024,
                guidance_scale=4.5, num_inference_steps=20, generator=generator,
            ).images[0].convert("RGB")
            image.save(target)
            if not valid_image(target):
                raise RuntimeError(f"SANA produced an invalid image: {target}")
            print(json.dumps({"event": "generated", "completed": completed, "sample_id": row["sample_id"]}), flush=True)
    finished = [finished_record(row, images_dir / row["filename"]) for row in rows]
    if any(row["width"] != 1024 or row["height"] != 1024 for row in finished):
        raise RuntimeError("all SANA candidates must be 1024x1024")
    atomic_json(args.output_dir / "manifest.jsonl", finished, jsonl=True)
    atomic_json(args.output_dir / "summary.json", {
        "schema_version": "r3dguard.pipeline-candidates-summary.v1",
        "groups": 1, "images": 4, "generated_this_run": len(missing),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    })
    print(json.dumps({"event": "complete", "images": 4, "generated_this_run": len(missing)}))


if __name__ == "__main__":
    main()
