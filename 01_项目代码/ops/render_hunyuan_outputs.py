#!/usr/bin/env python3
"""Render eight fixed views for an arbitrary successful Hunyuan status file."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


OPS_DIR = Path(__file__).resolve().parent
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

from render_ready3d_v2_views import render_one, write_jsonl


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def collect_assets(
    status_rows: list[dict], expected: int | None = None
) -> list[dict]:
    assets = []
    for row in status_rows:
        sample_id = str(row["sample_id"])
        paint = row.get("paint", {})
        if paint.get("status") != "success":
            raise ValueError(f"{sample_id} paint is not successful")
        glb_path = Path(paint["artifact_path"])
        if not glb_path.is_file() or glb_path.stat().st_size == 0:
            raise FileNotFoundError(glb_path)
        assets.append(
            {
                "sample_id": sample_id,
                "source_path": str(row["source_path"]),
                "glb_path": str(glb_path),
            }
        )
    unique = {asset["sample_id"] for asset in assets}
    if len(unique) != len(assets):
        raise ValueError("Hunyuan status must contain unique sample IDs")
    if expected is not None and len(assets) != expected:
        raise ValueError(f"expected {expected} assets, found {len(assets)}")
    return sorted(assets, key=lambda asset: asset["sample_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expect", type=int)
    parser.add_argument("--blender", type=Path, default=Path("/usr/bin/blender"))
    parser.add_argument("--xvfb-run", type=Path, default=Path("/usr/bin/xvfb-run"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets = collect_assets(read_jsonl(args.status), expected=args.expect)
    args.output_root.mkdir(parents=True, exist_ok=True)
    worker = OPS_DIR / "blender_render_ready3d_v2.py"
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                render_one,
                asset,
                args.output_root,
                args.blender,
                worker,
                args.timeout,
                args.xvfb_run,
            ): asset
            for asset in assets
        }
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            results[row["sample_id"]] = row
            write_jsonl(
                args.output_root.parent / "render_status.jsonl",
                list(results.values()),
            )
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": len(assets),
                        "sample_id": row["sample_id"],
                        "status": row["status"],
                        "cache_hit": row["cache_hit"],
                    }
                ),
                flush=True,
            )
    selected = [results[asset["sample_id"]] for asset in assets]
    failures = [row for row in selected if row["status"] != "success"]
    summary = {
        "schema_version": "r3dguard.hunyuan-renders.v1",
        "assets": len(selected),
        "successful": len(selected) - len(failures),
        "failed": len(failures),
        "views": sum(row.get("render_count", 0) for row in selected),
    }
    (args.output_root.parent / "render_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
