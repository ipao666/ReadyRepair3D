from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from r3dloop.prompt_optimizer import optimize_prompt  # noqa: E402
from r3dloop.prompt_optimizer.batch import build_pipeline_row, optimize_jsonl  # noqa: E402
from r3dloop.prompt_optimizer.runtime import Qwen3Generator  # noqa: E402


DEFAULT_MODEL = Path("/root/r3dguard/models/Qwen3-8B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict Chinese-to-SANA prompt optimizer")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="One Chinese prompt; JSON is printed to stdout")
    source.add_argument("--input", type=Path, help="Input JSONL with a prompt field")
    parser.add_argument("--output", type=Path, help="Resume-safe optimized JSONL output")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input is not None and args.output is None:
        raise SystemExit("--output is required with --input")
    generator = Qwen3Generator(args.model)
    if args.prompt is not None:
        result = optimize_prompt(args.prompt, generator, min_confidence=args.min_confidence)
        payload = build_pipeline_row(result)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, args.output)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    summary = optimize_jsonl(
        args.input,
        args.output,
        generator,
        prompt_field=args.prompt_field,
        min_confidence=args.min_confidence,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
