#!/usr/bin/env python3
"""Resume-safe renderer for the 320 existing Ready3D Hunyuan GLBs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_assets(status_paths: list[Path], expected: int = 320) -> list[dict]:
    assets = []
    for status_path in status_paths:
        domain = "ai_sana" if "ready3d_200" in str(status_path) else "real_photo_objectron"
        for row in read_jsonl(status_path):
            paint = row.get("paint", {})
            if paint.get("status") != "success":
                raise ValueError(f"paint is not successful: {row.get('sample_id')}")
            path = Path(paint["artifact_path"])
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
            assets.append({"sample_id": row["sample_id"], "domain": domain, "glb_path": str(path)})
    if len(assets) != expected or len({row["sample_id"] for row in assets}) != expected:
        raise ValueError(f"expected {expected} unique existing GLBs, found {len(assets)}")
    return sorted(assets, key=lambda row: row["sample_id"])


def complete_views(output_dir: Path) -> list[Path]:
    expected = [output_dir / f"shaded_{index:02d}.png" for index in range(8)]
    return expected if all(path.is_file() and path.stat().st_size > 0 for path in expected) else []


def existing_view_count(output_dir: Path) -> int:
    return sum(
        path.is_file() and path.stat().st_size > 0
        for path in (output_dir / f"shaded_{index:02d}.png" for index in range(8))
    )


def render_one(asset: dict, output_root: Path, blender: Path, worker: Path, timeout: int, xvfb_run: Path | None = None) -> dict:
    output_dir = output_root / asset["sample_id"]
    existing = complete_views(output_dir)
    if existing:
        return {**asset, "status": "success", "cache_hit": True, "render_count": 8, "render_paths": [str(path) for path in existing]}
    command = [str(blender), "--background", "--python", str(worker), "--", "--glb", asset["glb_path"], "--output", str(output_dir)]
    if xvfb_run is not None:
        command = [str(xvfb_run), "-a", *command]
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PATH"] = "/usr/bin:/bin"
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            **asset,
            "status": "failed",
            "cache_hit": False,
            "returncode": None,
            "error": f"render timed out after {timeout}s: {exc}",
            "render_count": existing_view_count(output_dir),
        }
    outputs = complete_views(output_dir)
    if len(outputs) != 8:
        return {**asset, "status": "failed", "cache_hit": False, "returncode": completed.returncode, "error": completed.stderr[-4000:], "render_count": existing_view_count(output_dir)}
    return {**asset, "status": "success", "cache_hit": False, "returncode": completed.returncode, "render_count": 8, "render_paths": [str(path) for path in outputs]}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: item["sample_id"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/r3dguard"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--blender", type=Path, default=Path("/usr/bin/blender"))
    parser.add_argument("--xvfb-run", type=Path, default=Path("/usr/bin/xvfb-run"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); root = args.root.resolve()
    output_root = args.output_root or root / "evaluation/ready3d_v2/renders"
    status_path = output_root.parent / "render_status.jsonl"
    assets = collect_assets([
        root / "data/hunyuan_ready3d_200/status.jsonl",
        root / "data/hunyuan_objectron_120/status.jsonl",
    ])
    if args.limit is not None: assets = assets[:args.limit]
    worker = root / "ops/blender_render_ready3d_v2.py"
    previous = {row["sample_id"]: row for row in read_jsonl(status_path)} if status_path.is_file() else {}
    results = dict(previous)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(render_one, asset, output_root, args.blender, worker, args.timeout, args.xvfb_run): asset for asset in assets}
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result(); row["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            results[row["sample_id"]] = row; write_jsonl(status_path, list(results.values()))
            print(json.dumps({"completed": completed, "total": len(assets), "sample_id": row["sample_id"], "status": row["status"], "cache_hit": row["cache_hit"]}), flush=True)
    selected = [results[row["sample_id"]] for row in assets]
    failures = [row for row in selected if row["status"] != "success"]
    summary = {"schema_version": "r3dguard.ready3d-v2-renders.v1", "assets": len(selected), "successful": len(selected) - len(failures), "failed": len(failures), "views": sum(row.get("render_count", 0) for row in selected), "existing_glbs_only": True}
    (output_root.parent / "render_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()
