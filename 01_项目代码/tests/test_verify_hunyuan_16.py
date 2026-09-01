from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace


OPS = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS))

from verify_hunyuan_16 import (  # noqa: E402
    camera_positions,
    choose_render_engine,
    ensure_world,
    summarize_telemetry,
    runtime_path,
)


def test_camera_positions_returns_eight_evenly_spaced_views() -> None:
    positions = camera_positions(center=(1.0, 2.0, 3.0), distance=5.0, count=8)

    assert len(positions) == 8
    horizontal_radii = [
        math.hypot(position[0] - 1.0, position[1] - 2.0)
        for position in positions
    ]
    assert max(horizontal_radii) - min(horizontal_radii) < 1e-9
    assert len({tuple(round(value, 6) for value in position) for position in positions}) == 8


def test_summarize_telemetry_reports_stage_and_overall_peaks(tmp_path: Path) -> None:
    path = tmp_path / "gpu.csv"
    fieldnames = [
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
    rows = [
        ["t1", "shape", "a", 0, 40960, 12000, 28960, 70, 51],
        ["t2", "paint", "a", 0, 40960, 25000, 15960, 95, 64],
        ["t3", "idle", "", 0, 40960, 10, 40950, 0, 35],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    summary = summarize_telemetry(path)

    assert summary["peak_memory_used_mib"] == 25000
    assert summary["shape_peak_memory_used_mib"] == 12000
    assert summary["paint_peak_memory_used_mib"] == 25000
    assert summary["max_utilization_gpu_percent"] == 95
    assert summary["max_temperature_c"] == 64


def test_render_engine_falls_back_for_blender_40() -> None:
    assert choose_render_engine({"BLENDER_EEVEE", "CYCLES"}) == "BLENDER_EEVEE"
    assert (
        choose_render_engine({"BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"})
        == "BLENDER_EEVEE_NEXT"
    )


def test_ensure_world_creates_one_after_empty_factory_reset() -> None:
    scene = SimpleNamespace(world=None)

    class Worlds:
        @staticmethod
        def new(name: str) -> SimpleNamespace:
            return SimpleNamespace(name=name, color=None)

    world = ensure_world(scene, Worlds())

    assert scene.world is world
    assert world.name == "BenchmarkWorld"
    assert world.color == (0.8, 0.8, 0.8)


def test_runtime_path_uses_environment_override() -> None:
    assert runtime_path(
        "R3D_TEST_ROOT", "/default", {"R3D_TEST_ROOT": "/validation64"}
    ) == Path("/validation64")
