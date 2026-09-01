#!/usr/bin/env python3
"""Expand the frozen 240-group development catalog into 960 SANA candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from r3dloop.sana_lora.data_protocol import (  # noqa: E402
    build_optimizer_inputs,
    expand_prompt_candidates,
    read_jsonl,
    validate_prompt_catalog,
    write_jsonl_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=2026083100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = read_jsonl(args.prompts)
    catalog = validate_prompt_catalog(
        groups,
        expected_split_counts={"train": 180, "validation": 30, "dev_test": 30},
    )
    candidates = expand_prompt_candidates(groups, base_seed=args.base_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.output_dir / "candidates960.jsonl", candidates)
    write_jsonl_atomic(
        args.output_dir / "prompt_optimizer_input.jsonl", build_optimizer_inputs(groups)
    )
    summary = {
        **catalog,
        "candidates": len(candidates),
        "candidate_split_counts": dict(Counter(row["split"] for row in candidates)),
        "candidates_per_group": 4,
        "base_seed": args.base_seed,
        "final_test_used": False,
    }
    temporary = args.output_dir / "summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
