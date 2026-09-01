#!/usr/bin/env python3
"""Build a one-row Hunyuan-style status file for post-repair V2 scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_status(
    labels: list[dict],
    selection_report: dict,
    repaired_glb: Path,
) -> dict:
    if not repaired_glb.is_file() or repaired_glb.stat().st_size == 0:
        raise FileNotFoundError(repaired_glb)
    winner_id = str(selection_report["winner_sample_id"])
    matches = [row for row in labels if str(row.get("sample_id")) == winner_id]
    if not matches:
        raise ValueError(f"winner label not found: {winner_id}")
    if len(matches) != 1:
        raise ValueError(f"winner must have exactly one quality label: {winner_id}")
    winner = matches[0]
    digest = hashlib.sha256(repaired_glb.read_bytes()).hexdigest()
    return {
        "sample_id": f"{winner_id}_postrepair_{digest[:12]}",
        "source_path": str(winner["source_path"]),
        "artifact_sha256": digest,
        "paint": {
            "status": "success",
            "artifact_path": str(repaired_glb.resolve()),
        },
    }


def write_jsonl_atomic(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--repaired-glb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = build_status(
        read_jsonl(args.labels),
        read_json(args.selection_report),
        args.repaired_glb,
    )
    write_jsonl_atomic(args.output, row)
    print(json.dumps({"sample_id": row["sample_id"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
