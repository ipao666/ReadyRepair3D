from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


VRAM_STOP_MIB = 38_000
GPU_CSV_HEADER = [
    "timestamp",
    "stage",
    "sample_id",
    "index",
    "memory_total_mib",
    "memory_used_mib",
    "memory_free_mib",
    "utilization_gpu_percent",
    "temperature_c",
]


def select_records(rows: list[dict], group_ids: list[str]) -> list[dict]:
    selected = [row for row in rows if row.get("group_id") in group_ids]
    seen: set[tuple[str, int]] = set()
    counts = {group_id: 0 for group_id in group_ids}
    for row in selected:
        key = (str(row["group_id"]), int(row["candidate_index"]))
        if key in seen:
            raise ValueError(f"duplicate candidate: {key[0]} c{key[1]}")
        seen.add(key)
        counts[key[0]] += 1
    for group_id, count in counts.items():
        if count != 4:
            raise ValueError(f"expected 4 candidates for {group_id}, got {count}")
    if len(selected) != len(group_ids) * 4:
        raise ValueError(f"expected {len(group_ids) * 4} records, got {len(selected)}")
    return selected


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_gpu_sample(
    line: str,
    timestamp: str,
    stage: str,
    sample_id: str,
) -> dict:
    values = [int(value.strip()) for value in line.strip().split(",")]
    if len(values) != 6:
        raise ValueError(f"expected 6 GPU fields, got {len(values)}: {line!r}")
    return dict(
        zip(
            GPU_CSV_HEADER,
            [timestamp, stage, sample_id, *values],
            strict=True,
        )
    )


def is_safe_peak(peak_used_mib: int, limit_mib: int = VRAM_STOP_MIB) -> bool:
    return peak_used_mib < limit_mib
