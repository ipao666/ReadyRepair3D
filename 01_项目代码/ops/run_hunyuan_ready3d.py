#!/usr/bin/env python3
"""Run the resumable Hunyuan pipeline over the full Ready3D candidate manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

OPS_DIR = Path(__file__).resolve().parent
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

import run_hunyuan_16 as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "ai_sana_200" / "manifest.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "hunyuan_ready3d_200"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_candidates(
    rows: list[dict], limit: int | None = None, require_complete_groups: bool = True
) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (str(row["group_id"]), int(row["candidate_index"])))
    seen: set[tuple[str, int]] = set()
    counts: dict[str, int] = {}
    for row in ordered:
        key = (str(row["group_id"]), int(row["candidate_index"]))
        if key in seen:
            raise ValueError(f"duplicate candidate: {key[0]} c{key[1]}")
        seen.add(key)
        counts[key[0]] = counts.get(key[0], 0) + 1
    incomplete = {group: count for group, count in counts.items() if count != 4}
    if require_complete_groups and incomplete:
        raise ValueError(f"every Best-of-4 group must have four candidates: {incomplete}")
    if limit is None:
        return ordered
    if limit < 1 or limit > len(ordered):
        raise ValueError(f"limit must be between 1 and {len(ordered)}")
    return ordered[:limit]


def configure_runner(manifest: Path, output_root: Path) -> None:
    runner.SOURCE_MANIFEST = manifest
    runner.OUTPUT_ROOT = output_root
    runner.STATUS_PATH = output_root / "status.jsonl"
    runner.GPU_CSV_PATH = output_root / "gpu_memory.csv"
    runner.CONTROL_PATH = output_root / "gpu_control.json"


def stage_command(args: argparse.Namespace, stage: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage", stage,
        "--manifest", str(args.manifest),
        "--output-root", str(args.output_root),
        "--stop-vram-mib", str(args.stop_vram_mib),
    ]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    if getattr(args, "allow_partial_groups", False):
        command.append("--allow-partial-groups")
    return command


def monitor_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(OPS_DIR / "monitor_hunyuan_gpu.py"),
        "--output", str(args.output_root / "gpu_memory.csv"),
        "--control", str(args.output_root / "gpu_control.json"),
        "--interval", "1",
    ]


def resolve_batch_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.manifest = args.manifest.resolve()
    args.output_root = args.output_root.resolve()
    return args


def wait_for_monitor_ready(process: subprocess.Popen, telemetry_path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"GPU monitor exited early with code {process.returncode}")
        if telemetry_path.exists() and telemetry_path.stat().st_size > 0:
            return
        time.sleep(0.1)
    raise RuntimeError(f"GPU monitor did not create telemetry within {timeout:.1f}s: {telemetry_path}")


def stop_monitor(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["shape", "paint", "all"], default="all")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stop-vram-mib", type=int, default=38_000)
    parser.add_argument("--allow-partial-groups", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = resolve_batch_paths(parse_args())
    if args.stage == "all":
        args.output_root.mkdir(parents=True, exist_ok=True)
        telemetry_path = args.output_root / "gpu_memory.csv"
        with (args.output_root / "monitor.log").open("a", encoding="utf-8") as monitor_log:
            monitor = subprocess.Popen(
                monitor_command(args),
                stdout=monitor_log,
                stderr=subprocess.STDOUT,
            )
            try:
                wait_for_monitor_ready(monitor, telemetry_path)
                subprocess.run(stage_command(args, "shape"), check=True)
                runner.wait_for_shape_release()
                subprocess.run(stage_command(args, "paint"), check=True)
            finally:
                stop_monitor(monitor)
        print(json.dumps({"event": "ready3d_complete", "stage": "all"}), flush=True)
        return

    configure_runner(args.manifest, args.output_root)
    selected = select_candidates(
        read_jsonl(args.manifest), args.limit,
        require_complete_groups=not args.allow_partial_groups,
    )
    status_by_id = runner.load_status()
    for row in selected:
        status_by_id.setdefault(runner.sample_id_from_row(row), runner.base_status(row))
    runner.persist_status(selected, status_by_id)
    control = args.output_root / "gpu_control.json"
    if args.stage == "shape":
        runner.run_shape_stage(selected, status_by_id, control, args.stop_vram_mib)
    else:
        runner.run_paint_stage(selected, status_by_id, control, args.stop_vram_mib)
    print(json.dumps({"event": "ready3d_complete", "stage": args.stage, "samples": len(selected)}), flush=True)


if __name__ == "__main__":
    main()
