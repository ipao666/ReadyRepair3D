#!/usr/bin/env python3
"""Resume-safe SANA generation with one frozen LoRA adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ops.generate_ready3d_v3_sana import (
    _finalize,
    build_generation_records,
    pending_records,
    read_jsonl,
    valid_image,
    write_jsonl_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--optimized-prompts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = args.output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = build_generation_records(
        read_jsonl(args.candidates), read_jsonl(args.optimized_prompts), images_dir
    )
    rows = [
        {
            **row,
            "generator": "SANA1.5_1.6B_1024px_diffusers+quality_lora",
            "lora_path": str(args.lora.resolve()),
        }
        for row in rows
    ]
    write_jsonl_atomic(args.output_root / "planned_manifest.jsonl", rows)
    pending = pending_records(rows)
    if pending:
        import torch
        from diffusers import SanaPipeline

        pipeline = SanaPipeline.from_pretrained(
            str(args.model), torch_dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        pipeline.load_lora_weights(str(args.lora), adapter_name="quality")
        pipeline.set_adapters("quality")
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
                raise RuntimeError(f"SANA LoRA produced an invalid image: {path}")
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
            {"event": "complete", "images": len(final), "generated_this_run": len(pending)}
        )
    )


if __name__ == "__main__":
    main()
