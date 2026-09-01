#!/usr/bin/env python3
"""Create a backend-neutral Ready3D label manifest before backend selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FAILURE_TYPES = [
    "silhouette_truncation",
    "depth_ambiguity",
    "background_leakage",
    "texture_lighting_conflict",
    "view_conflict",
    "insufficient_detail",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_pending_rows(
    trellis_rows: list[dict], hunyuan_rows: list[dict], hunyuan_render_root: str
) -> list[dict]:
    trellis = {row["sample_id"]: row for row in trellis_rows}
    hunyuan = {row["sample_id"]: row for row in hunyuan_rows}
    if set(trellis) != set(hunyuan):
        raise ValueError("Both backends must contain the same sample IDs")
    rows = []
    for sample_id in sorted(trellis):
        t_row, h_row = trellis[sample_id], hunyuan[sample_id]
        source = t_row.get("source_path")
        if source != h_row.get("source_path"):
            raise ValueError(f"Source mismatch for {sample_id}")
        rows.append(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "group_id": t_row.get("group_id"),
                "domain": "ai_sana",
                "split": "backend_benchmark",
                "source_path": source,
                "selected_backend": None,
                "selected_glb_path": None,
                "selected_render_dir": None,
                "label_status": "pending_backend_selection",
                "backend_artifacts": {
                    "trellis": {
                        "glb_path": t_row.get("glb_path"),
                        "render_dir": t_row.get("render_dir"),
                    },
                    "hunyuan": {
                        "glb_path": h_row.get("paint", {}).get("artifact_path"),
                        "render_dir": str(Path(hunyuan_render_root) / sample_id),
                    },
                },
                "ready_target": {
                    "quality": None,
                    "qualified": None,
                    "failure_types": None,
                    "label_source": None,
                },
            }
        )
    return rows


def write_outputs(output_dir: Path, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "pending_backend_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    schema = {
        "schema_version": 1,
        "failure_types": FAILURE_TYPES,
        "required_fields": [
            "sample_id",
            "domain",
            "split",
            "source_path",
            "selected_backend",
            "label_status",
            "backend_artifacts",
            "ready_target",
        ],
        "quality_range": [0.0, 1.0],
        "selection_rule": "Freeze selected_backend after blinded review; then compute downstream labels.",
    }
    (output_dir / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        """# Ready3D 标签目录

当前清单只登记输入和两个后端的已有产物，不提前选择后端，也不伪造质量标签。

后端盲评完成后，将 `selected_backend`、`selected_glb_path` 和 `selected_render_dir` 冻结，再批量计算下游质量分数、合格标签和六类自动失败标签。训练、验证、测试划分必须按 `group_id` 隔离。
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trellis-status", type=Path, required=True)
    parser.add_argument("--hunyuan-status", type=Path, required=True)
    parser.add_argument("--hunyuan-render-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pending_rows = build_pending_rows(
        read_jsonl(args.trellis_status),
        read_jsonl(args.hunyuan_status),
        args.hunyuan_render_root,
    )
    write_outputs(args.output_dir, pending_rows)
