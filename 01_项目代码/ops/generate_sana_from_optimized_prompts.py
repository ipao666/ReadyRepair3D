from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image


DEFAULT_MODEL = Path("/root/r3dguard/models/SANA1.5_1.6B_1024px_diffusers")
DEFAULT_BASE_SEED = 2026072100


def resolve_runtime_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _seed_base(prompt_id: str, base_seed: int) -> int:
    offset = int(hashlib.sha256(prompt_id.encode("utf-8")).hexdigest()[:8], 16) % 1_000_000
    return base_seed + offset * 10


def build_candidate_records(
    optimized_rows: list[dict[str, Any]],
    *,
    base_seed: int = DEFAULT_BASE_SEED,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in optimized_rows:
        if row.get("pipeline_ready") is not True:
            raise ValueError("every optimized row must have pipeline_ready=true")
        prompt_id = row.get("prompt_id")
        sana_prompt = row.get("sana_prompt")
        if not isinstance(prompt_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", prompt_id):
            raise ValueError("prompt_id must contain only letters, digits, dot, underscore, or hyphen")
        if prompt_id in seen:
            raise ValueError(f"duplicate prompt_id: {prompt_id}")
        seen.add(prompt_id)
        if not isinstance(sana_prompt, str) or not sana_prompt.strip():
            raise ValueError(f"{prompt_id}: missing sana_prompt")
        first_seed = _seed_base(prompt_id, base_seed)
        for candidate_index in range(4):
            seed = first_seed + candidate_index
            sample_id = f"{prompt_id}_c{candidate_index}_s{seed}"
            records.append(
                {
                    "sample_id": sample_id,
                    "prompt_id": prompt_id,
                    "group_id": prompt_id,
                    "candidate_index": candidate_index,
                    "seed": seed,
                    "filename": f"{sample_id}.png",
                    "source_prompt": row.get("source_prompt"),
                    "sana_prompt": sana_prompt.strip(),
                    "validated": bool(row.get("validated")),
                    "used_fallback": bool(row.get("used_fallback")),
                    "fallback_reason": row.get("fallback_reason"),
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


def pending_records(
    records: list[dict[str, Any]],
    images_dir: Path,
    validator: Callable[[Path], bool] = valid_image,
) -> list[dict[str, Any]]:
    return [row for row in records if not validator(images_dir / row["filename"])]


def build_manifest_row(
    record: dict[str, Any],
    image_path: Path,
    *,
    model_name: str,
) -> dict[str, Any]:
    """Add the stable path fields consumed by feature extraction and scoring."""
    path = str(image_path)
    return {
        **record,
        "path": path,
        "source_path": path,
        "model": model_name,
        "width": 1024,
        "height": 1024,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def _atomic_json(path: Path, payload: Any, *, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if jsonl:
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload)
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SANA Best-of-4 from validated optimizer JSONL")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input = resolve_runtime_path(args.input)
    args.output_dir = resolve_runtime_path(args.output_dir)
    args.model = resolve_runtime_path(args.model)
    rows = _read_jsonl(args.input)
    records = build_candidate_records(rows, base_seed=args.base_seed)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    pending = pending_records(records, images_dir)
    started = time.perf_counter()
    if pending:
        import torch
        from diffusers import SanaPipeline

        pipe = SanaPipeline.from_pretrained(
            str(args.model),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        for index, row in enumerate(pending, start=1):
            output = images_dir / row["filename"]
            generator = torch.Generator(device="cuda").manual_seed(row["seed"])
            image = pipe(
                prompt=row["sana_prompt"],
                height=1024,
                width=1024,
                guidance_scale=4.5,
                num_inference_steps=20,
                generator=generator,
            ).images[0].convert("RGB")
            image.save(output)
            if not valid_image(output):
                raise RuntimeError(f"invalid generated image: {output}")
            print(json.dumps({"event": "generated", "done": index, "total": len(pending), "path": str(output)}), flush=True)
        del pipe
        torch.cuda.empty_cache()
    finished = []
    for row in records:
        path = images_dir / row["filename"]
        if not valid_image(path):
            raise RuntimeError(f"missing output: {path}")
        finished.append(build_manifest_row(row, path, model_name=args.model.name))
    elapsed = time.perf_counter() - started
    _atomic_json(args.output_dir / "manifest.jsonl", finished, jsonl=True)
    summary = {
        "prompts": len(rows),
        "images": len(finished),
        "generated_this_run": len(pending),
        "fallback_prompts": sum(bool(row.get("used_fallback")) for row in rows),
        "elapsed_seconds": round(elapsed, 2),
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"event": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
