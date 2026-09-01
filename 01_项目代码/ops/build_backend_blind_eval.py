#!/usr/bin/env python3
"""Build a deterministic, balanced and method-anonymous backend review pack."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps


SCORE_COLUMNS = [
    "evaluation_id",
    "geometry_a",
    "geometry_b",
    "texture_a",
    "texture_b",
    "input_match_a",
    "input_match_b",
    "artifact_free_a",
    "artifact_free_b",
    "overall_preference",
    "comments",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_assignments(sample_ids: list[str], seed: int) -> list[dict]:
    """Assign method A in a deterministic 50/50 split without changing case order."""
    if not sample_ids:
        raise ValueError("No samples were provided")
    order = list(range(len(sample_ids)))
    random.Random(seed).shuffle(order)
    hunyuan_as_a = set(order[: len(order) // 2])
    assignments = []
    for index, sample_id in enumerate(sample_ids):
        method_a = "hunyuan" if index in hunyuan_as_a else "trellis"
        assignments.append(
            {
                "evaluation_id": f"case_{index + 1:02d}",
                "sample_id": sample_id,
                "method_a": method_a,
                "method_b": "trellis" if method_a == "hunyuan" else "hunyuan",
            }
        )
    return assignments


def build_public_manifest(assignments: list[dict]) -> list[dict]:
    return [
        {
            "evaluation_id": row["evaluation_id"],
            "sheet": f"sheets/{row['evaluation_id']}.jpg",
        }
        for row in assignments
    ]


def write_score_template(path: Path, assignments: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        for row in assignments:
            writer.writerow({"evaluation_id": row["evaluation_id"]})


def require_eight_views(paths: Iterable[Path], sample_id: str, method: str) -> list[Path]:
    views = sorted(Path(path) for path in paths)
    if len(views) != 8 or any(not path.is_file() for path in views):
        raise ValueError(
            f"{sample_id}/{method} must contain exactly 8 existing views; found {len(views)}"
        )
    return views


def _fit(image_path: Path, size: tuple[int, int], background=(238, 238, 238)) -> Image.Image:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def build_sheet(
    output: Path,
    evaluation_id: str,
    source_path: Path,
    views_a: list[Path],
    views_b: list[Path],
) -> None:
    width, height = 2560, 640
    header_y, content_y, tile = 64, 88, 256
    sheet = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    title_font, label_font = _font(30), _font(24)
    draw.text((24, 14), evaluation_id, fill=(30, 30, 30), font=title_font)
    draw.text((205, 18), "Source", fill=(70, 70, 70), font=label_font)
    draw.text((945, 18), "Method A", fill=(36, 76, 120), font=label_font)
    draw.text((1965, 18), "Method B", fill=(120, 70, 36), font=label_font)
    sheet.paste(_fit(source_path, (512, 512)), (0, content_y))
    for method_offset, paths in ((512, views_a), (1536, views_b)):
        for index, path in enumerate(paths):
            x = method_offset + (index % 4) * tile
            y = content_y + (index // 4) * tile
            sheet.paste(_fit(path, (tile, tile)), (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(210, 210, 210), width=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="JPEG", quality=94, subsampling=0, optimize=True)


def _status_index(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    index = {row["sample_id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return index


def _trellis_views(row: dict) -> list[Path]:
    return list(Path(row["render_dir"]).glob(f"{row['render_primary_key']}_*.png"))


def _hunyuan_views(render_root: Path, sample_id: str) -> list[Path]:
    return list((render_root / sample_id).glob("shaded_*.png"))


def write_instructions(path: Path) -> None:
    path.write_text(
        """# 3D 后端双盲评分说明

请两位评审分别填写自己的 CSV，不要互相讨论，也不要查看 `private` 目录。

每张图左侧为同一张输入图，中间为 Method A 的 8 个固定视角，右侧为 Method B 的 8 个固定视角。请分别评价：

- `geometry_*`：几何结构完整、轮廓合理、比例正确，1（很差）到 5（很好）。
- `texture_*`：纹理清晰、连贯、无明显拉伸或接缝，1（很差）到 5（很好）。
- `input_match_*`：与输入物体的形状、颜色和语义一致，1（很差）到 5（很好）。
- `artifact_free_*`：浮片、破洞、错误表面越少分越高，1（伪影严重）到 5（基本无伪影）。
- `overall_preference`：只能填写 `A`、`B` 或 `Tie`。
- `comments`：简短记录决定性优点或失败现象。

请先逐项打分，再填写总体偏好；不要根据生成速度、文件大小或对模型的先验印象评分。
""",
        encoding="utf-8",
    )


def build_pack(
    trellis_status: Path,
    hunyuan_status: Path,
    hunyuan_render_root: Path,
    output_root: Path,
    seed: int,
) -> None:
    trellis = _status_index(trellis_status)
    hunyuan = _status_index(hunyuan_status)
    if set(trellis) != set(hunyuan):
        raise ValueError("The two backends do not contain the same sample IDs")
    sample_ids = sorted(trellis)
    assignments = build_assignments(sample_ids, seed)

    review = output_root / "review"
    private = output_root / "private"
    (review / "sheets").mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    for row in assignments:
        sample_id = row["sample_id"]
        trellis_source = Path(trellis[sample_id]["source_path"])
        hunyuan_source = Path(hunyuan[sample_id]["source_path"])
        if trellis_source != hunyuan_source or not trellis_source.is_file():
            raise ValueError(f"Source mismatch or missing source for {sample_id}")
        method_views = {
            "trellis": require_eight_views(_trellis_views(trellis[sample_id]), sample_id, "trellis"),
            "hunyuan": require_eight_views(
                _hunyuan_views(hunyuan_render_root, sample_id), sample_id, "hunyuan"
            ),
        }
        build_sheet(
            review / "sheets" / f"{row['evaluation_id']}.jpg",
            row["evaluation_id"],
            trellis_source,
            method_views[row["method_a"]],
            method_views[row["method_b"]],
        )

    write_jsonl(review / "manifest.jsonl", build_public_manifest(assignments))
    write_score_template(review / "scores_reviewer_1.csv", assignments)
    write_score_template(review / "scores_reviewer_2.csv", assignments)
    write_instructions(review / "instructions.md")
    (private / "answer_key.json").write_text(
        json.dumps({"seed": seed, "assignments": assignments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trellis-status", type=Path, required=True)
    parser.add_argument("--hunyuan-status", type=Path, required=True)
    parser.add_argument("--hunyuan-render-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_pack(
        args.trellis_status,
        args.hunyuan_status,
        args.hunyuan_render_root,
        args.output_root,
        args.seed,
    )
