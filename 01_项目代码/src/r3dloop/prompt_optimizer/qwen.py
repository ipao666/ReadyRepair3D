from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from .core import (
    ENGLISH_RECONSTRUCTION_SUFFIX,
    PromptValidationError,
    build_normalized_chinese,
    build_sana_prompt,
    validate_translation,
)


@dataclass(frozen=True)
class PromptResult:
    source_prompt: str
    normalized_zh: str
    english_subject: str
    sana_prompt: str
    preserved_attributes: dict[str, list[str]]
    confidence: float
    used_fallback: bool
    fallback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PromptResponseError(ValueError):
    """Internal error carrying a stable fallback reason code."""


def parse_qwen_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise PromptResponseError(f"invalid_json:{exc}") from exc
    if not isinstance(payload, dict):
        raise PromptResponseError("invalid_json:root_not_object")
    return payload


def _validated_payload(source: str, payload: dict[str, Any], min_confidence: float) -> tuple[str, dict[str, list[str]], float]:
    if payload.get("source_prompt") != source:
        raise PromptResponseError("source_mismatch")
    english = payload.get("english_subject")
    subject_zh = payload.get("subject_zh")
    attributes = payload.get("preserved_attributes")
    confidence = payload.get("confidence")
    if not isinstance(english, str) or not english.strip():
        raise PromptResponseError("invalid_schema:english_subject")
    if not isinstance(subject_zh, str) or not subject_zh.strip():
        raise PromptResponseError("invalid_schema:subject_zh")
    if not isinstance(attributes, dict) or not all(
        isinstance(key, str) and isinstance(value, list) and all(isinstance(item, str) for item in value)
        for key, value in attributes.items()
    ):
        raise PromptResponseError("invalid_schema:preserved_attributes")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise PromptResponseError("invalid_schema:confidence")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise PromptResponseError("invalid_schema:confidence_range")
    if confidence < min_confidence:
        raise PromptResponseError(f"low_confidence:{confidence:.3f}")
    try:
        validate_translation(source, english)
    except PromptValidationError as exc:
        raise PromptResponseError(f"validation_failed:{exc}") from exc
    return english.strip(), attributes, confidence


def _fallback(source: str, reason: str) -> PromptResult:
    return PromptResult(
        source_prompt=source,
        normalized_zh=build_normalized_chinese(source),
        english_subject=source,
        sana_prompt=f"{source}, {ENGLISH_RECONSTRUCTION_SUFFIX}",
        preserved_attributes={},
        confidence=0.0,
        used_fallback=True,
        fallback_reason=reason,
    )


def optimize_prompt(
    source_prompt: str,
    generator: Callable[[str], str],
    *,
    min_confidence: float = 0.75,
) -> PromptResult:
    source = source_prompt.strip()
    if not source:
        raise PromptValidationError("source prompt is empty")
    try:
        raw = generator(source)
        payload = parse_qwen_response(raw)
        english, attributes, confidence = _validated_payload(source, payload, min_confidence)
    except PromptResponseError as exc:
        return _fallback(source, str(exc))
    except Exception as exc:  # Runtime failures must not abort a resumable batch.
        return _fallback(source, f"runtime_error:{type(exc).__name__}:{exc}")
    return PromptResult(
        source_prompt=source,
        normalized_zh=build_normalized_chinese(source),
        english_subject=english,
        sana_prompt=build_sana_prompt(english),
        preserved_attributes=attributes,
        confidence=confidence,
        used_fallback=False,
        fallback_reason=None,
    )

