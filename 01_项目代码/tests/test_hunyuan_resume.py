from __future__ import annotations

import sys
from pathlib import Path

import pytest


OPS = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS))

from run_hunyuan_16 import (  # noqa: E402
    REPO,
    build_stage_command,
    load_shape_components,
    parse_group_ids,
    needs_stage,
    sample_id_from_row,
)


def test_sample_id_comes_from_manifest_filename() -> None:
    assert sample_id_from_row({"filename": "folder/sana_000_c0.png"}) == "sana_000_c0"


def test_successful_valid_artifact_is_skipped(tmp_path: Path) -> None:
    artifact = tmp_path / "shape.obj"
    artifact.write_text("valid", encoding="utf-8")
    row = {"filename": "sana_000_c0.png"}
    status = {
        "sana_000_c0": {
            "shape": {"status": "success", "artifact_path": str(artifact)}
        }
    }

    assert not needs_stage(row, status, "shape", lambda path: path.read_text() == "valid")


def test_missing_artifact_is_scheduled_again(tmp_path: Path) -> None:
    row = {"filename": "sana_000_c0.png"}
    status = {
        "sana_000_c0": {
            "shape": {
                "status": "success",
                "artifact_path": str(tmp_path / "missing.obj"),
            }
        }
    }

    assert needs_stage(row, status, "shape", lambda path: path.is_file())


def test_failed_stage_is_scheduled_again() -> None:
    row = {"filename": "sana_000_c0.png"}
    status = {"sana_000_c0": {"paint": {"status": "failed"}}}

    assert needs_stage(row, status, "paint", lambda _path: True)


@pytest.mark.integration
@pytest.mark.skipif(not REPO.exists(), reason="Hunyuan3D repository is not packaged")
def test_shape_components_follow_official_repository_import_paths() -> None:
    components = load_shape_components()

    assert components["BackgroundRemover"].__module__ == "hy3dshape.rembg"
    assert components["Hunyuan3DDiTFlowMatchingPipeline"].__module__.startswith(
        "hy3dshape"
    )


def test_all_mode_builds_isolated_stage_subprocess_commands() -> None:
    command = build_stage_command(
        stage="paint",
        limit=16,
        stop_vram_mib=38_000,
        control_path=Path("/tmp/control.json"),
    )

    assert command[1].endswith("run_hunyuan_16.py")
    assert command[2:] == [
        "--stage",
        "paint",
        "--limit",
        "16",
        "--stop-vram-mib",
        "38000",
        "--monitor-control",
        str(Path("/tmp/control.json")),
    ]


def test_group_ids_can_be_configured_for_larger_batches() -> None:
    assert parse_group_ids("sana_001,sana_008,sana_011") == [
        "sana_001",
        "sana_008",
        "sana_011",
    ]
