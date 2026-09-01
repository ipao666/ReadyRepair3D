import csv
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "ops" / "build_backend_blind_eval.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_backend_blind_eval", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assignments_are_deterministic_balanced_and_anonymous():
    module = load_module()
    sample_ids = [f"sample_{index:02d}" for index in range(16)]

    first = module.build_assignments(sample_ids, seed=20260715)
    second = module.build_assignments(sample_ids, seed=20260715)

    assert first == second
    assert [row["evaluation_id"] for row in first] == [f"case_{i:02d}" for i in range(1, 17)]
    assert sum(row["method_a"] == "hunyuan" for row in first) == 8
    assert sum(row["method_a"] == "trellis" for row in first) == 8
    assert all({row["method_a"], row["method_b"]} == {"hunyuan", "trellis"} for row in first)

    public_rows = module.build_public_manifest(first)
    forbidden = {"sample_id", "method_a", "method_b"}
    assert all(not forbidden.intersection(row) for row in public_rows)
    assert [row["evaluation_id"] for row in public_rows] == [f"case_{i:02d}" for i in range(1, 17)]


def test_score_columns_and_templates(tmp_path):
    module = load_module()
    assignments = module.build_assignments([f"sample_{i:02d}" for i in range(16)], 20260715)
    output = tmp_path / "scores.csv"

    module.write_score_template(output, assignments)

    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 16
    assert list(rows[0]) == module.SCORE_COLUMNS
    assert rows[0]["evaluation_id"] == "case_01"
    assert all(not value for key, value in rows[0].items() if key != "evaluation_id")


def test_require_eight_views_rejects_incomplete_inputs(tmp_path):
    module = load_module()
    seven = []
    for index in range(7):
        path = tmp_path / f"view_{index}.png"
        path.touch()
        seven.append(path)

    try:
        module.require_eight_views(seven, "sample_00", "hunyuan")
    except ValueError as exc:
        assert "8" in str(exc)
        assert "sample_00" in str(exc)
    else:
        raise AssertionError("Incomplete view sets must be rejected")
