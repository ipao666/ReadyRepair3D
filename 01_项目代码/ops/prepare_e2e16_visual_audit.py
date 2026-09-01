from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def build_status(runs: Path, output: Path) -> None:
    rows = []
    for sample_dir in sorted(path for path in runs.iterdir() if path.is_dir()):
        final_glb = sample_dir / "final.glb"
        marker = sample_dir / "PIPELINE_COMPLETE"
        if not final_glb.is_file() or final_glb.stat().st_size == 0 or not marker.is_file():
            continue
        rows.append(
            {
                "sample_id": sample_dir.name,
                "source_path": str(final_glb.resolve()),
                "paint": {
                    "status": "success",
                    "artifact_path": str(final_glb.resolve()),
                },
            }
        )
    if len(rows) != 16:
        raise ValueError(f"expected 16 completed final GLBs, found {len(rows)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"status_rows": len(rows), "output": str(output)}))


def build_sheet(renders: Path, output: Path) -> None:
    sample_dirs = sorted(path for path in renders.iterdir() if path.is_dir())
    if len(sample_dirs) != 16:
        raise ValueError(f"expected 16 render directories, found {len(sample_dirs)}")

    tile_size = 256
    label_height = 34
    block_width = tile_size * 2
    block_height = label_height + tile_size * 2
    sheet = Image.new("RGB", (block_width * 4, block_height * 4), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)

    for index, sample_dir in enumerate(sample_dirs):
        block_x = (index % 4) * block_width
        block_y = (index // 4) * block_height
        draw.rectangle(
            (block_x, block_y, block_x + block_width, block_y + label_height),
            fill=(32, 37, 46),
        )
        draw.text(
            (block_x + 8, block_y + 7),
            sample_dir.name,
            fill="white",
            font=font,
        )
        for view_index, render_index in enumerate((0, 2, 4, 6)):
            image_path = sample_dir / f"shaded_{render_index:02d}.png"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            image = Image.open(image_path).convert("RGB")
            image = ImageOps.fit(image, (tile_size, tile_size))
            x = block_x + (view_index % 2) * tile_size
            y = block_y + label_height + (view_index // 2) * tile_size
            sheet.paste(image, (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=95)
    print(
        json.dumps(
            {
                "samples": len(sample_dirs),
                "views": len(sample_dirs) * 4,
                "size": list(sheet.size),
                "output": str(output),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--runs", type=Path, required=True)
    status.add_argument("--output", type=Path, required=True)

    sheet = subparsers.add_parser("sheet")
    sheet.add_argument("--renders", type=Path, required=True)
    sheet.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "status":
        build_status(args.runs, args.output)
    else:
        build_sheet(args.renders, args.output)


if __name__ == "__main__":
    main()
