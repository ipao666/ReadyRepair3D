#!/usr/bin/env python3
"""Read-only Ready3D V3 server preflight and resume planner."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


STAGES = ("generate_2d", "generate_3d", "render_3d", "score_3d", "complete")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _writable_anchor(path: Path) -> Path:
    current = path.resolve()
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _validate_candidates(rows: list[dict], expected_candidates: int) -> None:
    if len(rows) != expected_candidates:
        raise ValueError(
            f"expected {expected_candidates} candidates, found {len(rows)}"
        )
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate sample_id in candidate manifest")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("schema_version") != "r3dguard.ready3d-v3-candidate.v1":
            raise ValueError("unsupported candidate schema_version")
        groups[str(row["group_id"])].append(row)
    for group_id, candidates in groups.items():
        ordered = sorted(candidates, key=lambda row: int(row["candidate_index"]))
        indices = [int(row["candidate_index"]) for row in ordered]
        if indices != [0, 1, 2, 3]:
            raise ValueError(f"{group_id}: expected candidate indices [0, 1, 2, 3]")
        seeds = [int(row["seed"]) for row in ordered]
        if seeds != list(range(seeds[0], seeds[0] + 4)):
            raise ValueError(f"{group_id}: candidates must use consecutive fixed seeds")
        for row in ordered:
            if f"_s{int(row['seed'])}" not in str(row["sample_id"]):
                raise ValueError(f"{row['sample_id']}: sample_id does not encode fixed seed")


def build_preflight(
    candidates: list[dict],
    output_root: Path,
    labels_path: Path | None = None,
    expected_candidates: int = 320,
) -> dict:
    _validate_candidates(candidates, expected_candidates)
    output_root = output_root.resolve()
    anchor = _writable_anchor(output_root)
    writable = anchor.is_dir() and os.access(anchor, os.W_OK)
    label_ids: set[str] = set()
    if labels_path is not None and labels_path.is_file():
        label_rows = read_jsonl(labels_path)
        label_ids = {str(row["sample_id"]) for row in label_rows}
        if len(label_ids) != len(label_rows):
            raise ValueError("duplicate sample_id in quality labels")

    artifacts = []
    counts = Counter({stage: 0 for stage in STAGES})
    for row in sorted(
        candidates, key=lambda item: (str(item["group_id"]), int(item["candidate_index"]))
    ):
        sample_id = str(row["sample_id"])
        image = output_root / "images" / str(row["filename"])
        glb = output_root / "hunyuan" / "glb" / f"{sample_id}.glb"
        render_dir = output_root / "renders" / sample_id
        render_count = len(list(render_dir.glob("*.png"))) if render_dir.is_dir() else 0
        image_ready = image.is_file() and image.stat().st_size > 0
        glb_ready = glb.is_file() and glb.stat().st_size > 0
        label_ready = sample_id in label_ids
        if not image_ready:
            next_stage = "generate_2d"
        elif not glb_ready:
            next_stage = "generate_3d"
        elif render_count < 8:
            next_stage = "render_3d"
        elif not label_ready:
            next_stage = "score_3d"
        else:
            next_stage = "complete"
        counts[next_stage] += 1
        artifacts.append(
            {
                "sample_id": sample_id,
                "group_id": str(row["group_id"]),
                "candidate_index": int(row["candidate_index"]),
                "seed": int(row["seed"]),
                "image_ready": image_ready,
                "glb_ready": glb_ready,
                "render_count": render_count,
                "label_ready": label_ready,
                "next_stage": next_stage,
            }
        )
    resume = [row for row in artifacts if row["next_stage"] != "complete"]
    return {
        "schema_version": "r3dguard.ready3d-v3-server-preflight.v1",
        "ready": writable,
        "read_only": True,
        "output_root": str(output_root),
        "writable_anchor": str(anchor),
        "candidate_count": len(candidates),
        "group_count": len({row["group_id"] for row in candidates}),
        "fixed_seed_contract_valid": True,
        "counts": dict(counts),
        "resume_plan": resume,
        "artifacts": artifacts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--expect", type=int, default=320)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_preflight(
        read_jsonl(args.manifest), args.output_root, args.labels, args.expect
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(args.report.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, args.report)
    print(text, end="")
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
