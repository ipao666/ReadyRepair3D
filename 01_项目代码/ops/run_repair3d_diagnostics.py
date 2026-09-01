from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from repair3d_mesh import apply_blender_result, diagnose_glb, write_jsonl


ROOT = Path("/root/r3dguard")


def blender_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith("CONDA_") or name in {
            "PYTHONHOME",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
        }:
            environment.pop(name, None)
    environment["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    return environment


def blender_checks(paths: list[Path], blender: str, script: Path) -> dict[str, dict]:
    command = [blender, "--background", "--python", str(script), "--"] + [
        str(path.resolve()) for path in paths
    ]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=blender_environment(),
    )
    results: dict[str, dict] = {}
    for line in (process.stdout + "\n" + process.stderr).splitlines():
        if line.startswith("REPAIR3D_BLENDER="):
            result = json.loads(line.split("=", 1)[1])
            results[str(Path(result["path"]).resolve())] = result
    for path in paths:
        key = str(path.resolve())
        results.setdefault(
            key,
            {
                "path": key,
                "blender_loadable": False,
                "error": f"Blender returned {process.returncode} without a result",
            },
        )
    return results


def discover(
    input_directory: str | Path, backend: str = "hunyuan"
) -> list[tuple[str, Path]]:
    directory = Path(input_directory)
    return [(backend, path) for path in sorted(directory.glob("*.glb"))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Repair3D GLB inputs")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/hunyuan_16_test/glb",
        help="Hunyuan3D GLB directory",
    )
    parser.add_argument(
        "--backend",
        choices=("hunyuan", "trellis"),
        default="hunyuan",
        help="Hunyuan is primary; TRELLIS is comparison-only",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation/mesh_diagnostics.jsonl",
    )
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()

    discovered = discover(args.input, args.backend)
    paths = [path for _, path in discovered]
    records = [diagnose_glb(path, backend) for backend, path in discovered]
    checks = blender_checks(paths, args.blender, Path(__file__).with_name("blender_check_glb.py"))
    for record in records:
        apply_blender_result(record, checks[record["path"]])
    write_jsonl(args.output, records)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(records),
                "trimesh_loadable": sum(r["trimesh_loadable"] for r in records),
                "blender_loadable": sum(bool(r["blender_loadable"]) for r in records),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
