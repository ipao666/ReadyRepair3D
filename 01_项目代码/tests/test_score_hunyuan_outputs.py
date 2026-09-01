import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from score_hunyuan_outputs import collect_samples, finalize_scored  # noqa: E402


def write_status(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def success_row(sample_id: str, source: Path, glb: Path) -> dict:
    return {
        "sample_id": sample_id,
        "source_path": str(source),
        "paint": {"status": "success", "artifact_path": str(glb)},
    }


def test_collect_samples_is_sorted_and_requires_eight_views(tmp_path):
    render_root = tmp_path / "renders"
    rows = []
    for sample_id in ("sample_b", "sample_a"):
        source, glb = tmp_path / f"{sample_id}.png", tmp_path / f"{sample_id}.glb"
        source.touch()
        glb.touch()
        rows.append(success_row(sample_id, source, glb))
        directory = render_root / sample_id
        directory.mkdir(parents=True)
        for index in range(8):
            (directory / f"shaded_{index:02d}.png").touch()
    status = tmp_path / "status.jsonl"
    write_status(status, rows)

    samples = collect_samples(status, render_root, expect=2)

    assert [row["sample_id"] for row in samples] == ["sample_a", "sample_b"]
    assert all(len(row["render_paths"]) == 8 for row in samples)


def test_collect_samples_rejects_failed_paint(tmp_path):
    status = tmp_path / "status.jsonl"
    write_status(
        status,
        [{"sample_id": "bad", "paint": {"status": "failed"}, "source_path": "x"}],
    )
    with pytest.raises(ValueError, match="paint status"):
        collect_samples(status, tmp_path / "renders", expect=1)


def test_collect_samples_rejects_incomplete_views(tmp_path):
    source, glb = tmp_path / "source.png", tmp_path / "asset.glb"
    source.touch()
    glb.touch()
    status = tmp_path / "status.jsonl"
    write_status(status, [success_row("sample", source, glb)])
    directory = tmp_path / "renders" / "sample"
    directory.mkdir(parents=True)
    for index in range(7):
        (directory / f"shaded_{index:02d}.png").touch()
    with pytest.raises(ValueError, match="exactly 8"):
        collect_samples(status, tmp_path / "renders", expect=1)


def test_bootstrap_scores_do_not_claim_absolute_high_quality():
    rows = [{"quality_score_v2": 0.8, "technically_valid": True}]
    finalized = finalize_scored(rows, threshold=0.7, calibration_source="bootstrap_16")
    assert finalized[0]["technically_valid"] is True
    assert finalized[0]["relative_top30"] is True
    assert finalized[0]["high_quality"] is None


def test_frozen_validation_threshold_produces_high_quality_label():
    rows = [{"quality_score_v2": 0.8, "technically_valid": True}]
    finalized = finalize_scored(
        rows, threshold=0.7, calibration_source="independent_validation_64"
    )
    assert finalized[0]["high_quality"] is True
