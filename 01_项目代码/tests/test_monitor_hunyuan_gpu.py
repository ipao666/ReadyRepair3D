from __future__ import annotations

import json
import sys
from pathlib import Path


OPS = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS))

from monitor_hunyuan_gpu import read_control, sample_once  # noqa: E402


def test_read_control_defaults_to_idle_when_file_is_missing(tmp_path: Path) -> None:
    assert read_control(tmp_path / "missing.json") == {
        "stage": "idle",
        "sample_id": "",
    }


def test_sample_once_combines_control_and_nvidia_smi_output(tmp_path: Path) -> None:
    control = tmp_path / "control.json"
    control.write_text(
        json.dumps({"stage": "paint", "sample_id": "sana_000_c0"}),
        encoding="utf-8",
    )

    row = sample_once(
        control,
        runner=lambda: "0, 40960, 23456, 16987, 91, 62",
        timestamp="2026-07-15T15:30:00+08:00",
    )

    assert row["stage"] == "paint"
    assert row["sample_id"] == "sana_000_c0"
    assert row["memory_used_mib"] == 23456
    assert row["utilization_gpu_percent"] == 91
