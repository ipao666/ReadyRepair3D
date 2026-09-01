from __future__ import annotations

import json

import pytest

from r3dloop.prompt_optimizer import (
    ENGLISH_RECONSTRUCTION_SUFFIX,
    PromptValidationError,
    build_normalized_chinese,
    build_sana_prompt,
    validate_translation,
    optimize_prompt,
    parse_qwen_response,
)


def test_normalized_chinese_keeps_source_verbatim_at_start() -> None:
    source = "一个深红色黄铜机械猫，三条腿，齿轮结构"

    normalized = build_normalized_chinese(source)

    assert normalized.startswith(source)
    assert "不得改变主体、数量、颜色、材质和结构" in normalized
    assert "主体完整" in normalized


def test_sana_prompt_appends_only_frozen_reconstruction_suffix() -> None:
    translated_subject = "a dark red brass mechanical cat with three legs and a gear structure"

    prompt = build_sana_prompt(translated_subject)

    assert prompt == f"{translated_subject}, {ENGLISH_RECONSTRUCTION_SUFFIX}"
    assert "no cropped parts" in prompt


@pytest.mark.parametrize("source", ["", "   ", "\n\t"])
def test_empty_source_is_rejected(source: str) -> None:
    with pytest.raises(PromptValidationError, match="empty"):
        build_normalized_chinese(source)


def test_empty_translation_is_rejected() -> None:
    with pytest.raises(PromptValidationError, match="empty"):
        build_sana_prompt("  ")


def test_translation_preserves_common_color_material_and_count() -> None:
    source = "一个深红色黄铜机械猫，三条腿，齿轮结构"
    translated = "a dark red brass mechanical cat with three legs and a gear structure"

    validate_translation(source, translated)


def test_longer_chinese_attribute_does_not_require_its_shorter_substring_translation() -> None:
    validate_translation("一个深红色黄铜机械猫", "a burgundy brass mechanical cat")


@pytest.mark.parametrize(
    ("translated", "reason"),
    [
        ("a blue brass mechanical cat with three legs", "color"),
        ("a dark red steel mechanical cat with three legs", "material"),
        ("a dark red brass mechanical cat with two legs", "count"),
    ],
)
def test_translation_rejects_attribute_changes(translated: str, reason: str) -> None:
    source = "一个深红色黄铜机械猫，三条腿"

    with pytest.raises(PromptValidationError, match=reason):
        validate_translation(source, translated)


def test_unknown_material_does_not_cause_a_false_rejection() -> None:
    validate_translation("一个珐琅彩狮子摆件", "an enamel-painted lion figurine")


@pytest.mark.parametrize(
    ("source", "translated"),
    [
        ("一辆黄色金属玩具挖掘机", "a yellow metal toy excavator"),
        ("一双棕色皮革靴子", "a pair of brown leather boots"),
    ],
)
def test_common_chinese_classifiers_are_preserved(source: str, translated: str) -> None:
    validate_translation(source, translated)


def test_pair_classifier_rejects_singular_translation() -> None:
    with pytest.raises(PromptValidationError, match="count"):
        validate_translation("一双棕色皮革靴子", "a brown leather boot")


def _model_json(source: str, english: str, *, confidence: float = 0.96) -> str:
    return json.dumps(
        {
            "source_prompt": source,
            "subject_zh": source,
            "english_subject": english,
            "preserved_attributes": {
                "count": ["一个"],
                "colors": ["红色"],
                "materials": ["金属"],
                "structure": [],
            },
            "confidence": confidence,
        },
        ensure_ascii=False,
    )


def test_parse_qwen_response_accepts_plain_and_fenced_json() -> None:
    source = "一个红色金属机器人"
    raw = _model_json(source, "a red metal robot")

    assert parse_qwen_response(raw)["english_subject"] == "a red metal robot"
    assert parse_qwen_response(f"```json\n{raw}\n```")["source_prompt"] == source


def test_valid_qwen_output_builds_a_non_fallback_result() -> None:
    source = "一个红色金属机器人"

    result = optimize_prompt(source, lambda _: _model_json(source, "a red metal robot"))

    assert result.used_fallback is False
    assert result.normalized_zh.startswith(source)
    assert result.english_subject == "a red metal robot"
    assert result.sana_prompt.startswith("a red metal robot,")
    assert result.confidence == pytest.approx(0.96)


@pytest.mark.parametrize(
    ("raw_factory", "reason"),
    [
        (lambda source: "not json", "invalid_json"),
        (
            lambda source: _model_json(source + "被篡改", "a red metal robot"),
            "source_mismatch",
        ),
        (lambda source: _model_json(source, "a red metal robot", confidence=0.2), "low_confidence"),
        (lambda source: _model_json(source, "a blue plastic robot"), "validation_failed"),
    ],
)
def test_invalid_qwen_output_uses_traceable_fallback(raw_factory, reason: str) -> None:
    source = "一个红色金属机器人"

    result = optimize_prompt(source, lambda _: raw_factory(source))

    assert result.used_fallback is True
    assert result.fallback_reason.startswith(reason)
    assert result.sana_prompt.startswith(source)
    assert result.normalized_zh.startswith(source)
