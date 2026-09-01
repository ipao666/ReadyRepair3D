import importlib.util
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).parents[1] / "ops" / "run_hunyuan_ready3d.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_hunyuan_ready3d", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_is_group_ordered_and_complete():
    module = load_module()
    rows = [
        {"group_id": group, "candidate_index": candidate, "filename": f"{group}_{candidate}.png"}
        for group in ["g2", "g1"] for candidate in [3, 1, 0, 2]
    ]

    selected = module.select_candidates(rows)

    assert [(row["group_id"], row["candidate_index"]) for row in selected[:4]] == [
        ("g1", 0), ("g1", 1), ("g1", 2), ("g1", 3)
    ]


def test_incomplete_best_of_four_group_is_rejected():
    module = load_module()
    rows = [{"group_id": "g1", "candidate_index": index} for index in range(3)]
    try:
        module.select_candidates(rows)
    except ValueError as exc:
        assert "four candidates" in str(exc).lower()
    else:
        raise AssertionError("Incomplete groups must be rejected")


def test_stage_command_preserves_batch_configuration():
    module = load_module()
    args = SimpleNamespace(
        manifest=Path("/data/manifest.jsonl"), output_root=Path("/data/output"),
        stop_vram_mib=37000, limit=12,
    )

    command = module.stage_command(args, "shape")

    assert "shape" in command
    assert str(Path("/data/manifest.jsonl")) in command
    assert str(Path("/data/output")) in command
    assert command[-2:] == ["--limit", "12"]


def test_monitor_command_uses_same_output_and_control_files():
    module = load_module()
    args = SimpleNamespace(output_root=Path("/data/output"))

    command = module.monitor_command(args)

    assert command[1].endswith("monitor_hunyuan_gpu.py")
    assert command[command.index("--output") + 1] == str(
        Path("/data/output/gpu_memory.csv")
    )
    assert command[command.index("--control") + 1] == str(
        Path("/data/output/gpu_control.json")
    )


def test_stop_monitor_terminates_and_waits_for_process():
    module = load_module()

    class FakeProcess:
        def __init__(self):
            self.events = []

        def poll(self):
            return None

        def terminate(self):
            self.events.append("terminate")

        def wait(self, timeout=None):
            self.events.append(("wait", timeout))

    process = FakeProcess()
    module.stop_monitor(process)

    assert process.events == ["terminate", ("wait", 5)]


def test_batch_paths_are_absolute_before_backend_changes_working_directory(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(manifest=Path("data/manifest.jsonl"), output_root=Path("data/output"))

    module.resolve_batch_paths(args)

    assert args.manifest == tmp_path / "data/manifest.jsonl"
    assert args.output_root == tmp_path / "data/output"


def test_selected_single_candidate_is_allowed_only_when_explicit():
    module = load_module()
    rows = [{"group_id": "g1", "candidate_index": 2, "sample_id": "g1_c2"}]

    selected = module.select_candidates(rows, require_complete_groups=False)

    assert selected == rows


def test_partial_group_flag_is_forwarded_to_stage_process():
    module = load_module()
    args = SimpleNamespace(
        manifest=Path("/data/manifest.jsonl"), output_root=Path("/data/output"),
        stop_vram_mib=37000, limit=None, allow_partial_groups=True,
    )

    command = module.stage_command(args, "paint")

    assert "--allow-partial-groups" in command
