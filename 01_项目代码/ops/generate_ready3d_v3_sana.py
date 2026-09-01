#!/usr/bin/env python3
"""Resume-safe SANA generation for the fixed Ready3D V3 candidate manifest."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def default_model_path() -> Path:
    return Path(
        os.environ.get(
            "R3D_SANA_MODEL",
            str(ROOT / "models" / "SANA1.5_1.6B_1024px_diffusers"),
        )
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.load()
            return image.size == (1024, 1024) and image.mode == "RGB"
    except (OSError, ValueError):
        return False


def build_generation_records(
    candidates: list[dict], optimized: list[dict], images_dir: Path
) -> list[dict]:
    optimized_by_id = {}
    for row in optimized:
        prompt_id = str(row.get("prompt_id", ""))
        if not prompt_id:
            raise ValueError("optimized prompt is missing prompt_id")
        if prompt_id in optimized_by_id:
            raise ValueError(f"duplicate optimized prompt: {prompt_id}")
        optimized_by_id[prompt_id] = row
    output = []
    for candidate in candidates:
        group_id = str(candidate["group_id"])
        prompt = optimized_by_id.get(group_id)
        if prompt is None:
            raise ValueError(f"missing optimized prompt for {group_id}")
        if prompt.get("pipeline_ready") is not True:
            raise ValueError(f"{group_id}: optimizer output must have pipeline_ready=true")
        sana_prompt = prompt.get("sana_prompt")
        if not isinstance(sana_prompt, str) or not sana_prompt.strip():
            raise ValueError(f"{group_id}: optimized output is missing sana_prompt")
        image_path = (images_dir / str(candidate["filename"])).resolve()
        output.append(
            {
                **candidate,
                "source_prompt": candidate.get("prompt_zh"),
                "prompt": sana_prompt.strip(),
                "sana_prompt": sana_prompt.strip(),
                "optimizer_model": prompt.get("optimizer_model", "Qwen3-8B"),
                "optimizer_validated": bool(prompt.get("validated", False)),
                "optimizer_used_fallback": bool(prompt.get("used_fallback", False)),
                "path": str(image_path),
                "source_path": str(image_path),
                "generator": "SANA1.5_1.6B_1024px_diffusers",
            }
        )
    return output


def pending_records(
    rows: list[dict], validator: Callable[[Path], bool] = valid_image
) -> list[dict]:
    return [row for row in rows if not validator(Path(row["source_path"]))]


def _finalize(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        path = Path(row["source_path"])
        if not valid_image(path):
            raise RuntimeError(f"invalid or missing generated image: {path}")
        output.append(
            {
                **row,
                "width": 1024,
                "height": 1024,
                "bytes": path.stat().st_size,
                "guidance_scale": 4.5,
                "num_inference_steps": 20,
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--optimized-prompts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model_path(),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = args.output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = build_generation_records(
        read_jsonl(args.candidates), read_jsonl(args.optimized_prompts), images_dir
    )
    write_jsonl_atomic(args.output_root / "planned_manifest.jsonl", rows)
    pending = pending_records(rows)
    if pending:
        import torch
        from diffusers import SanaPipeline

        pipeline = SanaPipeline.from_pretrained(
            str(args.model), torch_dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        pipeline.set_progress_bar_config(disable=True)
        for completed, row in enumerate(pending, 1):
            path = Path(row["source_path"])
            path.unlink(missing_ok=True)
            generator = torch.Generator(device="cuda").manual_seed(int(row["seed"]))
            image = pipeline(
                prompt=row["prompt"],
                height=1024,
                width=1024,
                guidance_scale=4.5,
                num_inference_steps=20,
                generator=generator,
            ).images[0].convert("RGB")
            image.save(path)
            if not valid_image(path):
                raise RuntimeError(f"SANA produced an invalid image: {path}")
            print(
                json.dumps(
                    {
                        "event": "generated",
                        "completed": completed,
                        "pending_total": len(pending),
                        "sample_id": row["sample_id"],
                    }
                ),
                flush=True,
            )
    final = _finalize(rows)
    write_jsonl_atomic(args.output_root / "manifest.jsonl", final)
    print(
        json.dumps(
            {
                "event": "complete",
                "images": len(final),
                "generated_this_run": len(pending),
            }
        )
    )


if __name__ == "__main__":
    main()
