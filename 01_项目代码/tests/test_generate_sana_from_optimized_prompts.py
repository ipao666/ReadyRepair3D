from __future__ import annotations

from pathlib import Path

import pytest

from ops.generate_sana_from_optimized_prompts import (
    build_candidate_records,
    build_manifest_row,
    pending_records,
    resolve_runtime_path,
)


def _optimized_row(*, fallback: bool = False) -> dict:
    return {
        "prompt_id": "zh_0123456789ab",
        "source_prompt": "一个红色金属机器人",
        "sana_prompt": "a red metal robot, fixed constraints",
        "validated": not fallback,
        "pipeline_ready": True,
        "used_fallback": fallback,
        "fallback_reason": "low_confidence:0.200" if fallback else None,
    }


def test_builds_four_stable_candidates_with_provenance() -> None:
    first = build_candidate_records([_optimized_row()], base_seed=2026072100)
    second = build_candidate_records([_optimized_row()], base_seed=2026072100)

    assert first == second
    assert [row["candidate_index"] for row in first] == [0, 1, 2, 3]
    assert len({row["seed"] for row in first}) == 4
    assert all(row["prompt_id"] == "zh_0123456789ab" for row in first)
    assert all(row["group_id"] == "zh_0123456789ab" for row in first)
    assert all(row["sample_id"] == Path(row["filename"]).stem for row in first)
    assert all(row["used_fallback"] is False for row in first)


def test_safe_fallback_is_allowed_and_remains_traceable() -> None:
    records = build_candidate_records([_optimized_row(fallback=True)])

    assert len(records) == 4
    assert all(row["used_fallback"] is True for row in records)
    assert all(row["fallback_reason"].startswith("low_confidence") for row in records)


def test_manifest_row_satisfies_ready3d_downstream_contract(tmp_path: Path) -> None:
    record = build_candidate_records([_optimized_row()])[0]
    image_path = tmp_path / record["filename"]

    row = build_manifest_row(record, image_path, model_name="SANA-test")

    assert row["sample_id"] == record["sample_id"]
    assert row["source_path"] == str(image_path)
    assert row["path"] == str(image_path)
    assert row["group_id"] == record["group_id"]


def test_refuses_rows_not_emitted_as_pipeline_ready() -> None:
    row = _optimized_row()
    row.pop("pipeline_ready")

    with pytest.raises(ValueError, match="pipeline_ready"):
        build_candidate_records([row])


def test_pending_records_only_returns_missing_or_invalid_files(tmp_path: Path) -> None:
    records = build_candidate_records([_optimized_row()])
    existing = tmp_path / records[0]["filename"]
    existing.write_bytes(b"valid")

    pending = pending_records(records, tmp_path, lambda path: path == existing)

    assert [row["candidate_index"] for row in pending] == [1, 2, 3]


def test_runtime_paths_are_absolute_for_downstream_subprocesses(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert resolve_runtime_path(Path("outputs/demo")) == tmp_path / "outputs" / "demo"
