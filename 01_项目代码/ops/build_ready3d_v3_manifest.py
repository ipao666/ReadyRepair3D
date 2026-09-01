#!/usr/bin/env python3
"""Validate 80 Ready3D V3 prompt groups and expand them to 320 candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from r3dloop.ready3d_v3.data_protocol import (
    build_optimizer_inputs,
    expand_candidates,
    read_jsonl,
    validate_prompt_groups,
    write_jsonl_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=2026082700)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = read_jsonl(args.prompts)
    summary = validate_prompt_groups(groups)
    candidates = expand_candidates(groups, args.base_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.output_dir / "candidates320.jsonl", candidates)
    write_jsonl_atomic(
        args.output_dir / "prompt_optimizer_input.jsonl",
        build_optimizer_inputs(groups),
    )
    payload = {
        **summary,
        "candidate_schema_version": candidates[0]["schema_version"],
        "candidates": len(candidates),
        "candidates_per_group": 4,
        "base_seed": args.base_seed,
        "test_used_for_model_selection": False,
    }
    temporary = args.output_dir / "summary.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output_dir / "summary.json")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
