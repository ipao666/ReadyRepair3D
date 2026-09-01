#!/usr/bin/env python3
"""Join frozen candidates, 3D labels, and prompt adherence into LoRA weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from r3dloop.sana_lora.data_protocol import (  # noqa: E402
    read_jsonl,
    validate_lora_samples,
    write_jsonl_atomic,
)
from r3dloop.sana_lora.weights import build_weighted_record  # noqa: E402


def _unique_by_sample(rows: list[dict], source: str) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in output:
            raise ValueError(f"{source} has duplicate or empty sample_id: {sample_id}")
        output[sample_id] = row
    return output


def join_training_manifest(
    candidates: list[dict],
    labels: list[dict],
    adherence: list[dict],
    *,
    expected: int | None = 720,
) -> list[dict]:
    label_by_id = _unique_by_sample(labels, "labels")
    adherence_by_id = _unique_by_sample(adherence, "adherence")
    train_candidates = [row for row in candidates if row.get("split") == "train"]
    unexpected = sorted(
        str(row.get("sample_id"))
        for row in candidates
        if row.get("split") not in {"train", "validation", "dev_test", "final_test"}
    )
    if unexpected:
        raise ValueError(f"candidates contain invalid splits: {unexpected[:5]}")
    if expected is not None and len(train_candidates) != expected:
        raise ValueError(f"expected {expected} train candidates, found {len(train_candidates)}")
    output = []
    for candidate in train_candidates:
        sample_id = str(candidate["sample_id"])
        if sample_id not in label_by_id:
            raise ValueError(f"missing 3D label for {sample_id}")
        canonical = {
            **candidate,
            "prompt_group_id": str(
                candidate.get("prompt_group_id")
                or candidate.get("group_id")
                or candidate.get("prompt_id")
                or ""
            ),
            "prompt_zh": str(candidate.get("prompt_zh") or candidate.get("source_prompt") or ""),
            "caption_en": str(
                candidate.get("caption_en")
                or candidate.get("sana_prompt")
                or candidate.get("prompt")
                or ""
            ),
            "image_path": str(
                candidate.get("image_path")
                or candidate.get("path")
                or candidate.get("source_path")
                or ""
            ),
            "base_model": str(
                candidate.get("base_model") or "SANA1.5_1.6B_1024px_diffusers"
            ),
        }
        output.append(
            build_weighted_record(
                canonical,
                label_by_id[sample_id],
                adherence_by_id.get(sample_id),
            )
        )
    validate_lora_samples(output, training_only=True)
    return sorted(output, key=lambda row: str(row["sample_id"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--adherence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = join_training_manifest(
        read_jsonl(args.candidates),
        read_jsonl(args.labels),
        read_jsonl(args.adherence),
        expected=args.expect,
    )
    write_jsonl_atomic(args.output, rows)
    summary = {
        "samples": len(rows),
        "min_weight": min(row["sample_weight"] for row in rows),
        "max_weight": max(row["sample_weight"] for row in rows),
        "hunyuan_labels": sum(row["label_source"] == "hunyuan_frozen_q" for row in rows),
        "ready3d_labels": sum(
            row["label_source"] == "ready3d_v3_calibrated" for row in rows
        ),
        "adherence_missing": sum(bool(row["adherence_missing"]) for row in rows),
        "test_samples": sum(row["split"] != "train" for row in rows),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
