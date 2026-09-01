from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SHELL_ENTRYPOINTS = (
    ROOT / "run_demo.sh",
    ROOT / "activate_sana.sh",
    ROOT / "activate_hunyuan21.sh",
    ROOT / "ops" / "_env.sh",
    ROOT / "ops" / "run_full_quality_pipeline.sh",
    ROOT / "ops" / "run_repaired_quality_round.sh",
    ROOT / "ops" / "run_top2_dual3d_stage1.sh",
)


def test_ppt_metrics_remain_frozen() -> None:
    metrics = json.loads(
        (ROOT / "evaluation_summary" / "final_acceptance" / "final_metrics.json")
        .read_text(encoding="utf-8")
    )

    assert list(metrics["strategies"]) == [
        "direct",
        "ready_top2",
        "all",
    ]
    all_strategy = metrics["strategies"]["all"]
    assert all_strategy["mean_quality_score_v2"] == 0.7495
    assert all_strategy["qualified_rate"] == 0.472
    assert all_strategy["mean_3d_calls"] == 4.0
    assert all_strategy["mean_gpu_generation_seconds"] == 457.2


def test_active_shell_entrypoints_do_not_require_original_server_paths() -> None:
    for path in ACTIVE_SHELL_ENTRYPOINTS:
        text = path.read_text(encoding="utf-8")
        assert "/opt/conda" not in text, path
        assert "/root/r3dguard" not in text, path
        assert '"$ROOT/models' not in text, path


def test_stage1_passes_portable_feature_model_paths() -> None:
    script = (ROOT / "ops" / "run_top2_dual3d_stage1.sh").read_text(encoding="utf-8")
    assert '--model "$R3DGUARD_MODELS/Qwen3-8B"' in script
    assert '--model "$R3DGUARD_MODELS/SANA1.5_1.6B_1024px_diffusers"' in script
    assert '--dino "$R3DGUARD_MODELS/dinov2-large"' in script
    assert '--depth "$R3DGUARD_MODELS/Depth-Anything-V2-Large-hf"' in script
    assert '--birefnet "$R3DGUARD_MODELS/BiRefNet"' in script


def test_hunyuan_backend_uses_environment_paths() -> None:
    script = (ROOT / "ops" / "run_hunyuan_16.py").read_text(encoding="utf-8")
    assert '"HUNYUAN21_MODEL"' in script
    assert '"HUNYUAN21_REPO"' in script
    assert "/root/r3dguard" not in script
    wrapper = (ROOT / "ops" / "run_hunyuan_ready3d.py").read_text(encoding="utf-8")
    assert "/root/r3dguard" not in wrapper


def test_python_dependency_metadata_is_present() -> None:
    assert (ROOT / "pyproject.toml").is_file()
    assert (ROOT / "requirements.txt").is_file()
    assert (ROOT / "requirements-dev.txt").is_file()


def test_delivery_verifier_does_not_require_gpu_environment() -> None:
    script = (ROOT / "verify_delivery.sh").read_text(encoding="utf-8")
    assert "activate_hunyuan21.sh" not in script
    assert "build_sha256s.py" in script


def test_release_manifest_only_references_existing_paths() -> None:
    manifest = (ROOT / "RELEASE_MANIFEST.md").read_text(encoding="utf-8")
    assert "config/pipeline.local.reference.json" not in manifest


def test_submission_contains_viewable_example_assets() -> None:
    manifest_path = ROOT / "examples" / "showcase_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(manifest["assets"]) >= 3
    for asset in manifest["assets"]:
        assert (ROOT / asset["glb"]).is_file()
        assert (ROOT / asset["preview"]).is_file()
