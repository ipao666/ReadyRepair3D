from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


OPS = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS))

from hunyuan_batch_common import (  # noqa: E402
    GPU_CSV_HEADER,
    atomic_write_jsonl,
    is_safe_peak,
    parse_gpu_sample,
    select_records,
)


GROUP_IDS = ["sana_000", "sana_005", "sana_022", "sana_047"]


def make_rows() -> list[dict]:
    return [
        {
            "group_id": group_id,
            "candidate_index": candidate_index,
            "filename": f"{group_id}_c{candidate_index}.png",
        }
        for group_id in GROUP_IDS
        for candidate_index in range(4)
    ]


def test_select_records_returns_four_candidates_per_group_in_manifest_order() -> None:
    rows = make_rows()
    selected = select_records(rows, GROUP_IDS)

    assert selected == rows
    assert len(selected) == 16
    assert {row["group_id"] for row in selected} == set(GROUP_IDS)


def test_select_records_rejects_duplicate_candidate() -> None:
    rows = make_rows()
    rows.append(dict(rows[0]))

    with pytest.raises(ValueError, match="duplicate"):
        select_records(rows, GROUP_IDS)


def test_select_records_rejects_missing_candidate() -> None:
    with pytest.raises(ValueError, match="expected 4"):
        select_records(make_rows()[:-1], GROUP_IDS)


def test_atomic_write_jsonl_replaces_previous_content(tmp_path: Path) -> None:
    target = tmp_path / "status.jsonl"
    target.write_text('{"stale": true}\n', encoding="utf-8")

    atomic_write_jsonl(target, [{"sample_id": "one"}, {"sample_id": "two"}])

    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"sample_id": "one"}, {"sample_id": "two"}]


def test_parse_gpu_sample_returns_typed_csv_row() -> None:
    row = parse_gpu_sample(
        "0, 40960, 12345, 28615, 71, 55",
        timestamp="2026-07-15T15:00:00+08:00",
        stage="shape",
        sample_id="sana_000_c0",
    )

    assert row == {
        "timestamp": "2026-07-15T15:00:00+08:00",
        "stage": "shape",
        "sample_id": "sana_000_c0",
        "index": 0,
        "memory_total_mib": 40960,
        "memory_used_mib": 12345,
        "memory_free_mib": 28615,
        "utilization_gpu_percent": 71,
        "temperature_c": 55,
    }


def test_vram_safety_boundary_is_strictly_below_38000_mib() -> None:
    assert is_safe_peak(37_999)
    assert not is_safe_peak(38_000)


def test_gpu_csv_header_is_fixed() -> None:
    assert GPU_CSV_HEADER == [
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
