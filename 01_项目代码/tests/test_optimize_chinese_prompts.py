from __future__ import annotations

import json
from pathlib import Path

from r3dloop.prompt_optimizer.batch import build_pipeline_row, optimize_jsonl, stable_prompt_id
from r3dloop.prompt_optimizer.qwen import PromptResult
from r3dloop.prompt_optimizer.runtime import Qwen3Generator, SYSTEM_PROMPT


def _response(source: str) -> str:
    return json.dumps(
        {
            "source_prompt": source,
            "subject_zh": source,
            "english_subject": "a red metal robot",
            "preserved_attributes": {
                "count": ["一个"],
                "colors": ["红色"],
                "materials": ["金属"],
                "structure": [],
            },
            "confidence": 0.98,
        },
        ensure_ascii=False,
    )


def test_stable_prompt_id_is_content_based() -> None:
    first = stable_prompt_id("一个红色金属机器人")
    second = stable_prompt_id("  一个红色金属机器人  ")

    assert first == second
    assert first.startswith("zh_")
    assert len(first) == 15


def test_jsonl_optimization_is_resume_safe(tmp_path) -> None:
    source = "一个红色金属机器人"
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(json.dumps({"prompt": source}, ensure_ascii=False) + "\n", encoding="utf-8")
    calls: list[str] = []

    def generator(prompt: str) -> str:
        calls.append(prompt)
        return _response(prompt)

    first = optimize_jsonl(input_path, output_path, generator)
    second = optimize_jsonl(input_path, output_path, generator)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert first == {"total": 1, "written": 1, "skipped": 0}
    assert second == {"total": 1, "written": 0, "skipped": 1}
    assert calls == [source]
    assert rows[0]["prompt_id"] == stable_prompt_id(source)
    assert rows[0]["validated"] is True
    assert rows[0]["sana_prompt"].startswith("a red metal robot,")


def test_runtime_constructor_does_not_load_model() -> None:
    runtime = Qwen3Generator("/models/Qwen3-8B")

    assert runtime.is_loaded is False


def test_system_prompt_forbids_head_noun_summaries() -> None:
    assert "subject_zh MUST equal source_prompt exactly" in SYSTEM_PROMPT
    assert "english_subject MUST translate the entire source_prompt" in SYSTEM_PROMPT
    assert "every item in preserved_attributes" in SYSTEM_PROMPT.lower()
    assert "Never return only the head noun" in SYSTEM_PROMPT


def test_single_prompt_result_becomes_pipeline_ready_row() -> None:
    source = "一个红色金属机器人"
    result = PromptResult(
        source_prompt=source,
        normalized_zh=source,
        english_subject="a red metal robot",
        sana_prompt="a red metal robot, fixed constraints",
        preserved_attributes={"colors": ["红色"], "materials": ["金属"]},
        confidence=0.98,
        used_fallback=False,
        fallback_reason=None,
    )

    row = build_pipeline_row(result)

    assert row["prompt_id"] == stable_prompt_id(source)
    assert row["pipeline_ready"] is True
    assert row["validated"] is True
    assert row["optimizer_model"] == "Qwen3-8B"
    assert row["sana_prompt"] == result.sana_prompt


def test_main_pipeline_optimizes_before_sana_generation() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "ops" / "run_top2_dual3d_stage1.sh"
    ).read_text(encoding="utf-8")

    optimizer = 'python "$ROOT/ops/optimize_chinese_prompts.py"'
    generator = 'python "$ROOT/ops/generate_sana_from_optimized_prompts.py"'
    assert optimizer in script
    assert generator in script
    assert script.index(optimizer) < script.index(generator)
    assert '--input "$OPTIMIZED_PROMPT"' in script


def test_download_script_activates_project_environment_before_hf_cli() -> None:
    script = (Path(__file__).resolve().parents[1] / "ops" / "download_qwen3_8b.sh").read_text(
        encoding="utf-8"
    )

    assert script.index("source \"${ROOT}/activate.sh\"") < script.index("hf download")
    assert 'HF_ENDPOINT:-https://hf-mirror.com' in script
    assert 'HF_HUB_ENABLE_HF_TRANSFER:-0' in script
    assert 'HF_HUB_DISABLE_XET:-1' in script
