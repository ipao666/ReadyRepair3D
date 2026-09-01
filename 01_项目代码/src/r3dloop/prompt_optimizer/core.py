from __future__ import annotations

import re


CHINESE_RECONSTRUCTION_SUFFIX = (
    "不得改变主体、数量、颜色、材质和结构，不得添加或删除部件；"
    "单一主体，主体完整且全部位于画面内，居中，三分之四产品视角，"
    "浅灰色无缝影棚背景，柔和漫射光，轮廓清晰，几何结构明确，"
    "无遮挡，无人物、手、文字、标志、水印、额外物体或裁切。"
)

ENGLISH_RECONSTRUCTION_SUFFIX = (
    "strictly preserve the described subject, count, colors, materials, and structure; "
    "a single isolated object, the full object entirely visible within the frame, centered, "
    "three-quarter product view, plain light gray seamless studio background, soft diffuse "
    "lighting, clean silhouette, high geometric clarity, no occlusion, no people, no hands, "
    "no text, no logo, no watermark, no extra objects, no cropped parts"
)


class PromptValidationError(ValueError):
    """Raised when a prompt cannot safely enter the generation pipeline."""


def _clean_nonempty(value: str, *, field: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise PromptValidationError(f"{field} is empty")
    return cleaned


def build_normalized_chinese(source_prompt: str) -> str:
    source = _clean_nonempty(source_prompt, field="source prompt")
    return f"{source}。{CHINESE_RECONSTRUCTION_SUFFIX}"


def build_sana_prompt(translated_subject: str) -> str:
    subject = _clean_nonempty(translated_subject, field="translated subject")
    return f"{subject}, {ENGLISH_RECONSTRUCTION_SUFFIX}"


_COLOR_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("深红色", ("dark red", "deep red", "burgundy")),
    ("浅红色", ("light red", "pale red")),
    ("红色", ("red",)),
    ("蓝色", ("blue",)),
    ("绿色", ("green",)),
    ("黄色", ("yellow",)),
    ("黑色", ("black",)),
    ("白色", ("white",)),
    ("灰色", ("gray", "grey")),
    ("银色", ("silver",)),
    ("金色", ("gold", "golden")),
    ("棕色", ("brown",)),
    ("紫色", ("purple", "violet")),
    ("粉色", ("pink",)),
    ("橙色", ("orange",)),
    ("青色", ("cyan", "teal")),
    ("米色", ("beige", "cream")),
)

_MATERIAL_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("黄铜", ("brass",)),
    ("不锈钢", ("stainless steel",)),
    ("木质", ("wood", "wooden")),
    ("木头", ("wood", "wooden")),
    ("金属", ("metal", "metallic")),
    ("陶瓷", ("ceramic", "porcelain")),
    ("皮革", ("leather",)),
    ("玻璃", ("glass",)),
    ("塑料", ("plastic",)),
    ("橡胶", ("rubber",)),
    ("布料", ("fabric", "textile")),
    ("石材", ("stone",)),
    ("钢", ("steel",)),
)

_COUNT_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("一双", ("a pair", "pair of", "two", "2")),
    ("一对", ("a pair", "pair of", "two", "2")),
    ("一个", ("a", "an", "one", "single")),
    ("一辆", ("a", "an", "one", "single")),
    ("一把", ("a", "an", "one", "single")),
    ("一只", ("a", "an", "one", "single")),
    ("一张", ("a", "an", "one", "single")),
    ("一台", ("a", "an", "one", "single")),
    ("一件", ("a", "an", "one", "single")),
    ("单个", ("one", "single")),
    ("两个", ("two", "2")),
    ("二个", ("two", "2")),
    ("两条", ("two", "2")),
    ("三条", ("three", "3")),
    ("四条", ("four", "4")),
    ("五条", ("five", "5")),
    ("六条", ("six", "6")),
    ("两只", ("two", "2")),
    ("三只", ("three", "3")),
    ("四只", ("four", "4")),
)


def _contains_english(text: str, alternatives: tuple[str, ...]) -> bool:
    return any(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) for term in alternatives)


def _present_source_terms(
    source: str,
    terms: tuple[tuple[str, tuple[str, ...]], ...],
) -> list[tuple[str, tuple[str, ...]]]:
    candidates: list[tuple[int, int, str, tuple[str, ...]]] = []
    for zh, alternatives in terms:
        for match in re.finditer(re.escape(zh), source):
            candidates.append((match.start(), match.end(), zh, alternatives))
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: list[tuple[int, int, str, tuple[str, ...]]] = []
    for candidate in candidates:
        start, end = candidate[0], candidate[1]
        if any(start < chosen_end and end > chosen_start for chosen_start, chosen_end, _, _ in selected):
            continue
        selected.append(candidate)
    return [(zh, alternatives) for _, _, zh, alternatives in selected]


def _validate_term_group(
    source: str,
    translated: str,
    terms: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    label: str,
    reject_conflicts: bool,
) -> None:
    present = _present_source_terms(source, terms)
    present_terms = {zh for zh, _ in present}
    for zh, alternatives in present:
        if not _contains_english(translated, alternatives):
            raise PromptValidationError(f"{label} was not preserved: {zh}")
    if reject_conflicts and present:
        allowed = {term for _, alternatives in present for term in alternatives}
        for zh, alternatives in terms:
            is_component_of_allowed = any(
                _contains_english(allowed_phrase, (candidate_term,))
                for candidate_term in alternatives
                for allowed_phrase in allowed
            )
            if zh in present_terms or is_component_of_allowed:
                continue
            if _contains_english(translated, alternatives):
                raise PromptValidationError(f"conflicting {label} introduced: {alternatives[0]}")


def validate_translation(source_prompt: str, translated_subject: str) -> None:
    source = _clean_nonempty(source_prompt, field="source prompt")
    translated = _clean_nonempty(translated_subject, field="translated subject").lower()
    _validate_term_group(source, translated, _COLOR_TERMS, label="color", reject_conflicts=True)
    _validate_term_group(source, translated, _MATERIAL_TERMS, label="material", reject_conflicts=True)
    _validate_term_group(source, translated, _COUNT_TERMS, label="count", reject_conflicts=False)
