from pathlib import Path
import sys

import pytest

OPS = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS))

from extract_ready3d_features import normalize_manifest_rows


def test_normalize_manifest_rows_accepts_sana_generation_schema():
    rows = normalize_manifest_rows(
        [
            {
                "filename": "sana_000_c0_s1.png",
                "path": "/tmp/sana_000_c0_s1.png",
                "group_id": "sana_000",
            }
        ]
    )

    assert rows[0]["sample_id"] == "sana_000_c0_s1"
    assert rows[0]["source_path"] == "/tmp/sana_000_c0_s1.png"
    assert rows[0]["domain"] == "ai_sana"
    assert rows[0]["split"] == "unassigned"


def test_normalize_manifest_rows_rejects_duplicate_derived_ids():
    row = {
        "filename": "same.png",
        "path": "/tmp/same.png",
        "group_id": "group",
    }
    with pytest.raises(ValueError, match="duplicate sample_id"):
        normalize_manifest_rows([row, row])


def test_normalize_manifest_rows_derives_filename_from_source_path():
    rows = normalize_manifest_rows(
        [{"source_path": str(Path("/tmp/example.png")), "group_id": "group"}]
    )
    assert rows[0]["sample_id"] == "example"


def test_normalize_manifest_rows_rejects_missing_group_id():
    with pytest.raises(ValueError, match="non-empty group_id"):
        normalize_manifest_rows([{"source_path": "/tmp/example.png"}])
