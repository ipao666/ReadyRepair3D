import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "ops" / "bootstrap_ready3d_labels.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bootstrap_ready3d_labels", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pending_rows_preserve_both_backends_without_premature_selection():
    module = load_module()
    trellis = [{
        "sample_id": "s1", "group_id": "g1", "source_path": "/data/s1.png",
        "glb_path": "/trellis/s1.glb", "render_dir": "/trellis/renders/s1",
    }]
    hunyuan = [{
        "sample_id": "s1", "group_id": "g1", "source_path": "/data/s1.png",
        "paint": {"artifact_path": "/hunyuan/s1.glb"},
    }]

    rows = module.build_pending_rows(trellis, hunyuan, "/hunyuan/renders")

    assert len(rows) == 1
    row = rows[0]
    assert row["selected_backend"] is None
    assert row["label_status"] == "pending_backend_selection"
    assert set(row["backend_artifacts"]) == {"trellis", "hunyuan"}
    assert row["ready_target"]["quality"] is None
    assert row["ready_target"]["failure_types"] is None


def test_mismatched_samples_are_rejected():
    module = load_module()
    try:
        module.build_pending_rows(
            [{"sample_id": "only_trellis"}],
            [{"sample_id": "only_hunyuan"}],
            "/renders",
        )
    except ValueError as exc:
        assert "same sample" in str(exc).lower()
    else:
        raise AssertionError("Mismatched backend samples must be rejected")
