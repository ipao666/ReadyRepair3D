"""Build and verify a deterministic SHA256 manifest for the submission package."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXCLUDED_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".venv",
    ".ruff_cache",
    "__pycache__",
    "data",
    "logs",
    "models",
    "outputs",
    "tmp",
}
EXCLUDED_FILES = {"SHA256SUMS", ".env"}


def collect_release_files(root: Path) -> list[Path]:
    root = root.resolve()
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
            continue
        if any(part.endswith(".egg-info") for part in relative.parts):
            continue
        if relative.name in EXCLUDED_FILES or relative.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: Path, manifest: Path) -> None:
    root = root.resolve()
    lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in collect_release_files(root)
    ]
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text("".join(lines), encoding="utf-8", newline="\n")
    temporary.replace(manifest)


def verify_manifest(root: Path, manifest: Path) -> dict[str, list[str]]:
    root = root.resolve()
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    current = {
        path.relative_to(root).as_posix(): sha256(path)
        for path in collect_release_files(root)
    }
    return {
        "missing": sorted(set(expected) - set(current)),
        "mismatched": sorted(
            relative
            for relative in set(expected) & set(current)
            if expected[relative] != current[relative]
        ),
        "unexpected": sorted(set(current) - set(expected)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest = args.root.resolve() / "SHA256SUMS"
    if args.write:
        write_manifest(args.root, manifest)
        print(f"wrote {manifest}")
        return
    result = verify_manifest(args.root, manifest)
    print(result)
    if any(result.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
