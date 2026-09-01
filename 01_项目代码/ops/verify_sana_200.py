from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(sys.argv[1])
    manifest_path = root / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 200, f"expected 200 rows, got {len(rows)}"
    groups = Counter(row["group_id"] for row in rows)
    categories = Counter(row["category"] for row in rows)
    filenames = [row["filename"] for row in rows]
    assert len(groups) == 50, f"expected 50 groups, got {len(groups)}"
    assert set(groups.values()) == {4}, f"group sizes are not all four: {groups}"
    assert len(categories) == 10, f"expected 10 categories, got {len(categories)}"
    assert set(categories.values()) == {20}, f"category counts are unbalanced: {categories}"
    assert len(filenames) == len(set(filenames)), "duplicate filenames in manifest"

    hashes = []
    thumbnails = []
    for row in rows:
        path = Path(row["path"])
        assert path.is_file() and path.stat().st_size > 0, f"missing or empty: {path}"
        with Image.open(path) as image:
            image.load()
            assert image.size == (1024, 1024), f"wrong size: {path} {image.size}"
            assert image.mode == "RGB", f"wrong mode: {path} {image.mode}"
            thumb = ImageOps.fit(image, (120, 120), method=Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (128, 144), "white")
            tile.paste(thumb, (4, 4))
            ImageDraw.Draw(tile).text(
                (4, 126),
                f"{row['group_id']} c{row['candidate_index']}",
                fill="black",
            )
            thumbnails.append(tile)
        hashes.append(sha256(path))

    duplicate_hashes = [digest for digest, count in Counter(hashes).items() if count > 1]
    assert not duplicate_hashes, f"found {len(duplicate_hashes)} exact duplicate hashes"

    sheet = Image.new("RGB", (1280, 2880), "#dddddd")
    for index, tile in enumerate(thumbnails):
        sheet.paste(tile, ((index % 10) * 128, (index // 10) * 144))
    sheet_path = root / "contact_sheet.jpg"
    sheet.save(sheet_path, quality=90, optimize=True)

    report = {
        "ok": True,
        "images": len(rows),
        "groups": len(groups),
        "categories": dict(sorted(categories.items())),
        "exact_duplicate_hashes": 0,
        "total_bytes": sum(Path(row["path"]).stat().st_size for row in rows),
        "contact_sheet": str(sheet_path),
    }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
