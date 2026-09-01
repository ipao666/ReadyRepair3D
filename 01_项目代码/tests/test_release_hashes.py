from __future__ import annotations

from pathlib import Path

from ops.build_sha256s import collect_release_files, sha256, verify_manifest


def test_hash_manifest_excludes_caches_and_itself(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "main.pyc").write_bytes(b"cache")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache" / "state").write_text("cache", encoding="utf-8")
    (tmp_path / "SHA256SUMS").write_text("old", encoding="utf-8")

    files = collect_release_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in files] == ["src/main.py"]


def test_verify_manifest_reports_no_changes_for_matching_files(tmp_path: Path) -> None:
    payload = tmp_path / "result.json"
    payload.write_text('{"value": 1}\n', encoding="utf-8")
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(
        f"{sha256(payload)}  result.json\n",
        encoding="utf-8",
    )

    result = verify_manifest(tmp_path, manifest)

    assert result == {"missing": [], "mismatched": [], "unexpected": []}
