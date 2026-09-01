import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "ops" / "render_hunyuan_outputs.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "render_hunyuan_outputs", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_assets_reads_successful_paint_outputs(tmp_path):
    module = load_module()
    glb = tmp_path / "g_c1.glb"
    glb.write_bytes(b"glTF")
    rows = [
        {
            "sample_id": "g_c1",
            "source_path": "/images/g_c1.png",
            "paint": {"status": "success", "artifact_path": str(glb)},
        }
    ]

    assets = module.collect_assets(rows, expected=1)

    assert assets == [
        {
            "sample_id": "g_c1",
            "source_path": "/images/g_c1.png",
            "glb_path": str(glb),
        }
    ]


def test_collect_assets_rejects_failed_paint(tmp_path):
    module = load_module()
    rows = [
        {
            "sample_id": "g_c1",
            "source_path": "/images/g_c1.png",
            "paint": {"status": "failed"},
        }
    ]

    with pytest.raises(ValueError, match="paint is not successful"):
        module.collect_assets(rows, expected=1)


def test_collect_assets_rejects_missing_glb(tmp_path):
    module = load_module()
    rows = [
        {
            "sample_id": "g_c1",
            "source_path": "/images/g_c1.png",
            "paint": {
                "status": "success",
                "artifact_path": str(tmp_path / "missing.glb"),
            },
        }
    ]

    with pytest.raises(FileNotFoundError):
        module.collect_assets(rows, expected=1)


def test_collect_assets_rejects_duplicate_ids(tmp_path):
    module = load_module()
    glb = tmp_path / "g.glb"
    glb.write_bytes(b"glTF")
    row = {
        "sample_id": "g",
        "source_path": "/images/g.png",
        "paint": {"status": "success", "artifact_path": str(glb)},
    }

    with pytest.raises(ValueError, match="unique"):
        module.collect_assets([row, row], expected=2)


def test_collect_assets_enforces_expected_count(tmp_path):
    module = load_module()
    glb = tmp_path / "g.glb"
    glb.write_bytes(b"glTF")
    row = {
        "sample_id": "g",
        "source_path": "/images/g.png",
        "paint": {"status": "success", "artifact_path": str(glb)},
    }

    with pytest.raises(ValueError, match="expected 2"):
        module.collect_assets([row], expected=2)
