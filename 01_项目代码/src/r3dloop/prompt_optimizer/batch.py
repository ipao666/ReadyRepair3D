from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .qwen import PromptResult, optimize_prompt


def stable_prompt_id(source_prompt: str) -> str:
    normalized = source_prompt.strip()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"zh_{digest}"


def build_pipeline_row(
    result: PromptResult,
    *,
    prompt_id: str | None = None,
) -> dict[str, Any]:
    """Convert one validated/fallback result into the SANA JSONL contract."""
    return {
        "prompt_id": prompt_id or stable_prompt_id(result.source_prompt),
        **result.to_dict(),
        "validated": not result.used_fallback,
        "pipeline_ready": True,
        "optimizer_model": "Qwen3-8B",
        "optimizer_mode": "non-thinking-greedy",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be a JSON object")
        rows.append(row)
    return rows


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def optimize_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    generator: Callable[[str], str],
    *,
    prompt_field: str = "prompt",
    min_confidence: float = 0.75,
) -> dict[str, int]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    inputs = _read_jsonl(input_path)
    outputs = _read_jsonl(output_path)
    completed = {str(row.get("prompt_id")) for row in outputs if row.get("prompt_id")}
    written = 0
    skipped = 0
    for input_row in inputs:
        source = input_row.get(prompt_field)
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"input row is missing non-empty string field {prompt_field!r}")
        source = source.strip()
        prompt_id = str(input_row.get("prompt_id") or input_row.get("id") or stable_prompt_id(source))
        if prompt_id in completed:
            skipped += 1
            continue
        result = optimize_prompt(source, generator, min_confidence=min_confidence)
        outputs.append({**input_row, **build_pipeline_row(result, prompt_id=prompt_id)})
        completed.add(prompt_id)
        written += 1
        _atomic_write_jsonl(output_path, outputs)
    return {"total": len(inputs), "written": written, "skipped": skipped}
