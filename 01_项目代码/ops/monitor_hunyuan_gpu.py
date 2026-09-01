from __future__ import annotations

import argparse
import csv
import json
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from hunyuan_batch_common import GPU_CSV_HEADER, parse_gpu_sample


QUERY_COMMAND = [
    "nvidia-smi",
    "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
    "--format=csv,noheader,nounits",
]
STOP_REQUESTED = False


def read_control(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"stage": "idle", "sample_id": ""}
    return {
        "stage": str(payload.get("stage", "idle")),
        "sample_id": str(payload.get("sample_id", "")),
    }


def run_nvidia_smi() -> str:
    result = subprocess.run(
        QUERY_COMMAND,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[0]


def sample_once(
    control_path: Path,
    runner: Callable[[], str] = run_nvidia_smi,
    timestamp: str | None = None,
) -> dict:
    control = read_control(control_path)
    return parse_gpu_sample(
        runner(),
        timestamp=timestamp or datetime.now().astimezone().isoformat(timespec="seconds"),
        stage=control["stage"],
        sample_id=control["sample_id"],
    )


def request_stop(_signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    write_header = not args.output.exists() or args.output.stat().st_size == 0
    with args.output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GPU_CSV_HEADER)
        if write_header:
            writer.writeheader()
            handle.flush()
        while not STOP_REQUESTED:
            try:
                writer.writerow(sample_once(args.control))
                handle.flush()
            except Exception as error:
                print(f"GPU monitor sample failed: {type(error).__name__}: {error}", flush=True)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
