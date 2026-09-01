"""Deterministic prompt-group protocol for Ready3D V3 Top-1 experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path


SCHEMA_VERSION = "r3dguard.ready3d-v3-prompt-group.v1"
CANDIDATE_SCHEMA_VERSION = "r3dguard.ready3d-v3-candidate.v1"
SPLIT_COUNTS = {"train": 56, "validation": 12, "test": 12}
REQUIRED_RISKS = {
    "low_risk",
    "thin_parts",
    "multi_limb",
    "open_structure",
    "asymmetry",
    "reflective",
    "complex_support",
    "repeated_texture",
}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number} must be a JSON object")
        rows.append(row)
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def subject_fingerprint(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def validate_prompt_groups(rows: list[dict]) -> dict:
    if len(rows) != 80:
        raise ValueError(f"expected 80 prompt groups, found {len(rows)}")
    required = {
        "schema_version",
        "group_id",
        "split",
        "category",
        "difficulty",
        "risk_tags",
        "subject_zh",
        "prompt_zh",
    }
    group_ids: set[str] = set()
    fingerprints: dict[str, tuple[str, str]] = {}
    split_counts: Counter[str] = Counter()
    risks: set[str] = set()
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"row {index} missing fields: {missing}")
        if row["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"row {index} has unsupported schema_version")
        group_id = str(row["group_id"])
        if group_id in group_ids:
            raise ValueError(f"duplicate group_id: {group_id}")
        group_ids.add(group_id)
        split = str(row["split"])
        if split not in SPLIT_COUNTS:
            raise ValueError(f"invalid split for {group_id}: {split}")
        split_counts[split] += 1
        if row["difficulty"] not in {"easy", "medium", "hard"}:
            raise ValueError(f"invalid difficulty for {group_id}")
        if not isinstance(row["risk_tags"], list) or not row["risk_tags"]:
            raise ValueError(f"risk_tags must be a non-empty list for {group_id}")
        risks.update(str(value) for value in row["risk_tags"])
        fingerprint = subject_fingerprint(str(row["subject_zh"]))
        if fingerprint in fingerprints:
            previous_group, previous_split = fingerprints[fingerprint]
            raise ValueError(
                "subject leakage: "
                f"{previous_group}/{previous_split} and {group_id}/{split} share a subject"
            )
        fingerprints[fingerprint] = (group_id, split)
        if len(str(row["prompt_zh"]).strip()) < 30:
            raise ValueError(f"prompt_zh is too short for {group_id}")
    if dict(split_counts) != SPLIT_COUNTS:
        raise ValueError(
            f"split counts must be {SPLIT_COUNTS}, found {dict(split_counts)}"
        )
    missing_risks = sorted(REQUIRED_RISKS - risks)
    if missing_risks:
        raise ValueError(f"missing required risk coverage: {missing_risks}")
    return {
        "schema_version": SCHEMA_VERSION,
        "groups": len(rows),
        "split_groups": dict(split_counts),
        "categories": sorted({str(row["category"]) for row in rows}),
        "difficulties": sorted({str(row["difficulty"]) for row in rows}),
        "risk_tags": sorted(risks),
    }


def expand_candidates(rows: list[dict], base_seed: int = 2026082700) -> list[dict]:
    validate_prompt_groups(rows)
    candidates: list[dict] = []
    for group_position, row in enumerate(rows):
        for candidate_index in range(4):
            seed = int(base_seed + group_position * 10 + candidate_index)
            sample_id = f"{row['group_id']}_c{candidate_index}_s{seed}"
            candidates.append(
                {
                    "schema_version": CANDIDATE_SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "group_id": row["group_id"],
                    "candidate_index": candidate_index,
                    "split": row["split"],
                    "category": row["category"],
                    "difficulty": row["difficulty"],
                    "risk_tags": list(row["risk_tags"]),
                    "subject_zh": row["subject_zh"],
                    "prompt_zh": row["prompt_zh"],
                    "seed": seed,
                    "filename": f"{sample_id}.png",
                }
            )
    return candidates


def build_optimizer_inputs(rows: list[dict]) -> list[dict]:
    """Create the stable Qwen optimizer contract without changing group identity."""

    validate_prompt_groups(rows)
    return [{**row, "prompt_id": str(row["group_id"])} for row in rows]
