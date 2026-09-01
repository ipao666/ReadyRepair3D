import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "ops" / "build_post_repair_status.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "build_post_repair_status", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_status_preserves_winner_source_and_uses_repaired_glb(tmp_path):
    module = load_module()
    repaired = tmp_path / "final.glb"
    repaired.write_bytes(b"glTF")
    labels = [
        {"sample_id": "g_c0_s10", "source_path": "/images/c0.png"},
        {"sample_id": "g_c2_s12", "source_path": "/images/c2.png"},
    ]

    row = module.build_status(
        labels,
        {"winner_sample_id": "g_c2_s12"},
        repaired,
    )

    assert row["sample_id"].startswith("g_c2_s12_postrepair_")
    assert row["source_path"] == "/images/c2.png"
    assert row["artifact_sha256"]
    assert row["paint"]["status"] == "success"
    assert row["paint"]["artifact_path"] == str(repaired.resolve())


def test_build_status_rejects_missing_winner(tmp_path):
    module = load_module()
    repaired = tmp_path / "final.glb"
    repaired.write_bytes(b"glTF")

    with pytest.raises(ValueError, match="winner"):
        module.build_status(
            [{"sample_id": "other", "source_path": "/images/other.png"}],
            {"winner_sample_id": "missing"},
            repaired,
        )


def test_build_status_rejects_duplicate_winner_labels(tmp_path):
    module = load_module()
    repaired = tmp_path / "final.glb"
    repaired.write_bytes(b"glTF")
    label = {"sample_id": "winner", "source_path": "/images/winner.png"}

    with pytest.raises(ValueError, match="exactly one"):
        module.build_status(
            [label, label],
            {"winner_sample_id": "winner"},
            repaired,
        )


def test_build_status_requires_existing_repaired_glb(tmp_path):
    module = load_module()

    with pytest.raises(FileNotFoundError):
        module.build_status(
            [{"sample_id": "winner", "source_path": "/images/winner.png"}],
            {"winner_sample_id": "winner"},
            tmp_path / "missing.glb",
        )
