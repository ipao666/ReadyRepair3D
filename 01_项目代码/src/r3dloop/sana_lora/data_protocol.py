"""Stable data contracts for the single-operator SANA LoRA experiment."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


PROMPT_SCHEMA_VERSION = "r3dguard.sana-lora-prompt-group.v1"
CANDIDATE_SCHEMA_VERSION = "r3dguard.sana-lora-candidate.v1"
PROMPT_SPLITS = {"train", "validation", "dev_test", "final_test"}
TRAINABLE_SPLIT = "train"
QUALITY_FIELDS = {
    "quality",
    "hunyuan_quality",
    "ready3d_calibrated_quality",
    "calibrated_3d_quality",
    "prompt_adherence",
}


def read_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    rows: list[dict] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{line_number}: row must be a JSON object")
        rows.append(row)
    return rows


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)


def validate_unit_interval(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number in [0, 1]")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be in [0, 1], got {number}")
    return number


def validate_lora_samples(rows: list[dict], *, training_only: bool = False) -> dict:
    required = {
        "sample_id",
        "prompt_group_id",
        "split",
        "prompt_zh",
        "caption_en",
        "seed",
        "image_path",
        "base_model",
    }
    sample_ids: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"row {index} missing fields: {missing}")
        sample_id = str(row["sample_id"])
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"duplicate or empty sample_id: {sample_id}")
        sample_ids.add(sample_id)
        split = str(row["split"])
        if split not in PROMPT_SPLITS:
            raise ValueError(f"invalid split for {sample_id}: {split}")
        if training_only and split != TRAINABLE_SPLIT:
            raise ValueError(f"non-training sample entered training manifest: {sample_id}/{split}")
        group_id = str(row["prompt_group_id"])
        group_splits[group_id].add(split)
        split_counts[split] += 1
        seed = row["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"missing or invalid seed for {sample_id}")
        if not str(row["prompt_zh"]).strip() or not str(row["caption_en"]).strip():
            raise ValueError(f"empty prompt or caption for {sample_id}")
        for field in QUALITY_FIELDS & set(row):
            if row[field] is not None:
                validate_unit_interval(row[field], field)
    leaking = sorted(group for group, splits in group_splits.items() if len(splits) != 1)
    if leaking:
        raise ValueError(f"prompt groups cross splits: {leaking}")
    return {
        "samples": len(rows),
        "groups": len(group_splits),
        "split_samples": dict(split_counts),
    }


def validate_prompt_catalog(
    rows: list[dict],
    *,
    expected_split_counts: dict[str, int],
    forbidden_subject_keys: set[str] | None = None,
) -> dict:
    required = {
        "schema_version",
        "prompt_group_id",
        "split",
        "category",
        "subject_key",
        "subject_zh",
        "prompt_zh",
        "prompt",
    }
    ids: set[str] = set()
    subjects: set[str] = set()
    split_counts: Counter[str] = Counter()
    forbidden = {_normalized(value) for value in (forbidden_subject_keys or set())}
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"prompt row {index} missing fields: {missing}")
        if row["schema_version"] != PROMPT_SCHEMA_VERSION:
            raise ValueError(f"prompt row {index} has unsupported schema_version")
        group_id = str(row["prompt_group_id"])
        if group_id in ids:
            raise ValueError(f"duplicate prompt_group_id: {group_id}")
        ids.add(group_id)
        split = str(row["split"])
        if split not in PROMPT_SPLITS:
            raise ValueError(f"invalid split for {group_id}: {split}")
        split_counts[split] += 1
        subject = _normalized(str(row["subject_key"]))
        if not subject or subject in subjects or subject in forbidden:
            raise ValueError(f"duplicate or forbidden semantic subject: {row['subject_key']}")
        subjects.add(subject)
        if str(row["prompt"]) != str(row["prompt_zh"]):
            raise ValueError(f"prompt alias differs from prompt_zh for {group_id}")
        if len(str(row["prompt_zh"]).strip()) < 25:
            raise ValueError(f"prompt_zh is too short for {group_id}")
    if dict(split_counts) != expected_split_counts:
        raise ValueError(
            f"split counts must be {expected_split_counts}, found {dict(split_counts)}"
        )
    return {
        "groups": len(rows),
        "split_groups": dict(split_counts),
        "categories": sorted({str(row["category"]) for row in rows}),
        "semantic_subjects": len(subjects),
    }


def build_optimizer_inputs(groups: list[dict]) -> list[dict]:
    """Map frozen prompt groups to the optimizer's stable input contract."""
    return [
        {
            "prompt_id": str(row["prompt_group_id"]),
            "prompt_group_id": str(row["prompt_group_id"]),
            "split": str(row["split"]),
            "prompt_zh": str(row["prompt_zh"]),
        }
        for row in groups
    ]


def expand_prompt_candidates(
    groups: list[dict], *, base_seed: int, candidates_per_group: int = 4
) -> list[dict]:
    """Expand development prompts into deterministic, group-isolated candidates."""
    if candidates_per_group != 4:
        raise ValueError("the frozen SANA LoRA protocol requires four candidates per group")
    output: list[dict] = []
    for group_index, group in enumerate(groups):
        if group.get("split") == "final_test":
            raise ValueError("final_test prompts cannot enter development candidate generation")
        group_id = str(group["prompt_group_id"])
        for candidate_index in range(candidates_per_group):
            seed = int(base_seed) + group_index * candidates_per_group + candidate_index
            sample_id = f"{group_id}_c{candidate_index}_s{seed}"
            output.append(
                {
                    "schema_version": CANDIDATE_SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "prompt_group_id": group_id,
                    "prompt_id": group_id,
                    "split": str(group["split"]),
                    "category": str(group["category"]),
                    "subject_key": str(group["subject_key"]),
                    "subject_zh": str(group["subject_zh"]),
                    "prompt_zh": str(group["prompt_zh"]),
                    "candidate_index": candidate_index,
                    "seed": seed,
                    "filename": sample_id + ".png",
                    "base_model": "SANA1.5_1.6B_1024px_diffusers",
                }
            )
    if len({row["sample_id"] for row in output}) != len(output):
        raise ValueError("candidate expansion produced duplicate sample IDs")
    return output
