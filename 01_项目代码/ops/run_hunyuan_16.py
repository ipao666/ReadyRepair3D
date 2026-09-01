from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

from hunyuan_batch_common import atomic_write_jsonl, is_safe_peak, select_records


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_group_ids(value: str) -> list[str]:
    groups = [item.strip() for item in value.split(",") if item.strip()]
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("group IDs must be a non-empty comma-separated unique list")
    return groups

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = str(
    Path(
        os.environ.get(
            "HUNYUAN21_MODEL",
            PROJECT_ROOT / "models" / "Hunyuan3D-2.1",
        )
    )
)
SUBFOLDER = "hunyuan3d-dit-v2-1"
REPO = Path(
    os.environ.get(
        "HUNYUAN21_REPO",
        PROJECT_ROOT / "repos" / "Hunyuan3D-2.1",
    )
)
SOURCE_MANIFEST = Path(
    os.environ.get(
        "R3D_HUNYUAN_SOURCE_MANIFEST",
        PROJECT_ROOT / "data" / "ai_sana_200" / "manifest.jsonl",
    )
)
OUTPUT_ROOT = Path(
    os.environ.get(
        "R3D_HUNYUAN_OUTPUT_ROOT",
        PROJECT_ROOT / "data" / "hunyuan_16_test",
    )
)
GROUP_IDS = parse_group_ids(
    os.environ.get("R3D_HUNYUAN_GROUP_IDS", "sana_000,sana_005,sana_022,sana_047")
)
STATUS_PATH = OUTPUT_ROOT / "status.jsonl"
GPU_CSV_PATH = OUTPUT_ROOT / "gpu_memory.csv"
CONTROL_PATH = OUTPUT_ROOT / "gpu_control.json"


def sample_id_from_row(row: dict) -> str:
    return Path(row["filename"]).stem


def needs_stage(
    row: dict,
    status_by_id: dict[str, dict],
    stage: str,
    validator: Callable[[Path], bool],
) -> bool:
    sample_id = sample_id_from_row(row)
    stage_status = status_by_id.get(sample_id, {}).get(stage, {})
    if stage_status.get("status") != "success":
        return True
    artifact_path = stage_status.get("artifact_path")
    if not artifact_path:
        return True
    path = Path(artifact_path)
    try:
        return not path.is_file() or not validator(path)
    except Exception:
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["shape", "paint", "all"], default="all")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--stop-vram-mib", type=int, default=38_000)
    parser.add_argument("--monitor-control", type=Path, default=CONTROL_PATH)
    return parser.parse_args()


def build_stage_command(
    stage: str,
    limit: int,
    stop_vram_mib: int,
    control_path: Path,
) -> list[str]:
    if stage not in {"shape", "paint"}:
        raise ValueError(f"isolated stage must be shape or paint, got {stage}")
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--limit",
        str(limit),
        "--stop-vram-mib",
        str(stop_vram_mib),
        "--monitor-control",
        str(control_path),
    ]


def load_status() -> dict[str, dict]:
    if not STATUS_PATH.exists():
        return {}
    rows = [
        json.loads(line)
        for line in STATUS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {row["sample_id"]: row for row in rows}


def base_status(row: dict) -> dict:
    return {
        "sample_id": sample_id_from_row(row),
        "group_id": row["group_id"],
        "candidate_index": row["candidate_index"],
        "category": row.get("category"),
        "difficulty": row.get("difficulty"),
        "source_path": row["path"],
        "seed": int(row["seed"]),
    }


def persist_status(selected: list[dict], status_by_id: dict[str, dict]) -> None:
    ordered = [
        status_by_id[sample_id_from_row(row)]
        for row in selected
        if sample_id_from_row(row) in status_by_id
    ]
    atomic_write_jsonl(STATUS_PATH, ordered)


def write_control(path: Path, stage: str, sample_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"stage": stage, "sample_id": sample_id}) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def peak_for_sample(stage: str, sample_id: str) -> int | None:
    if not GPU_CSV_PATH.exists():
        return None
    with GPU_CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            int(row["memory_used_mib"])
            for row in csv.DictReader(handle)
            if row.get("stage") == stage and row.get("sample_id") == sample_id
        ]
    return max(rows) if rows else None


def current_gpu_used_mib() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def validate_shape(path: Path) -> bool:
    import numpy as np
    import trimesh

    loaded = trimesh.load(path, force="mesh", process=False)
    return bool(
        len(loaded.vertices)
        and len(loaded.faces)
        and np.isfinite(loaded.vertices).all()
        and np.isfinite(loaded.faces).all()
    )


def inspect_glb(path: Path) -> dict:
    import numpy as np
    import trimesh

    scene = trimesh.load(path, force="scene", process=False)
    geometries = list(scene.geometry.values())
    if not geometries:
        raise ValueError(f"GLB has no geometry: {path}")
    if not all(
        np.isfinite(mesh.vertices).all() and np.isfinite(mesh.faces).all()
        for mesh in geometries
    ):
        raise ValueError(f"GLB has non-finite geometry: {path}")
    textured = sum(
        getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None)
        is not None
        or getattr(getattr(mesh.visual, "material", None), "image", None) is not None
        for mesh in geometries
    )
    if textured == 0:
        raise ValueError(f"GLB has no texture-bearing material: {path}")
    return {
        "glb_bytes": path.stat().st_size,
        "geometries": len(geometries),
        "vertices": sum(len(mesh.vertices) for mesh in geometries),
        "faces": sum(len(mesh.faces) for mesh in geometries),
        "textured_geometries": textured,
    }


def validate_glb(path: Path) -> bool:
    inspect_glb(path)
    return True


def write_safety_stop(stage: str, sample_id: str, peak: int | None, reason: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": stage,
        "sample_id": sample_id,
        "peak_memory_used_mib": peak,
        "reason": reason,
    }
    (OUTPUT_ROOT / "safety_stop.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def enforce_sample_gate(
    stage: str,
    sample_id: str,
    stop_vram_mib: int,
) -> int:
    time.sleep(1.2)
    peak = peak_for_sample(stage, sample_id)
    if peak is None:
        write_safety_stop(stage, sample_id, None, "missing GPU telemetry")
        raise RuntimeError(f"missing GPU telemetry for {stage}/{sample_id}")
    if not is_safe_peak(peak, stop_vram_mib):
        write_safety_stop(stage, sample_id, peak, f"peak reached {stop_vram_mib} MiB limit")
        raise RuntimeError(
            f"{stage}/{sample_id} peak {peak} MiB reached limit {stop_vram_mib} MiB"
        )
    return peak


def is_cuda_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower() and "cuda" in str(error).lower()


def load_shape_components() -> dict[str, object]:
    sys.path.insert(0, str(REPO))
    from hy3dshape import FaceReducer, Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.pipelines import export_to_trimesh
    from hy3dshape.rembg import BackgroundRemover

    return {
        "BackgroundRemover": BackgroundRemover,
        "FaceReducer": FaceReducer,
        "Hunyuan3DDiTFlowMatchingPipeline": Hunyuan3DDiTFlowMatchingPipeline,
        "export_to_trimesh": export_to_trimesh,
    }


def run_shape_stage(
    selected: list[dict],
    status_by_id: dict[str, dict],
    control_path: Path,
    stop_vram_mib: int,
) -> None:
    import torch
    from PIL import Image

    os.chdir(REPO)
    components = load_shape_components()
    BackgroundRemover = components["BackgroundRemover"]
    FaceReducer = components["FaceReducer"]
    Hunyuan3DDiTFlowMatchingPipeline = components[
        "Hunyuan3DDiTFlowMatchingPipeline"
    ]
    export_to_trimesh = components["export_to_trimesh"]

    pending = [
        row for row in selected if needs_stage(row, status_by_id, "shape", validate_shape)
    ]
    if not pending:
        print(json.dumps({"event": "shape_complete", "processed": 0}), flush=True)
        return

    shape_dir = OUTPUT_ROOT / "shape"
    shape_dir.mkdir(parents=True, exist_ok=True)
    write_control(control_path, "shape", "__model_load__")
    load_started = time.perf_counter()
    remover = BackgroundRemover()
    reducer = FaceReducer()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        MODEL,
        subfolder=SUBFOLDER,
        use_safetensors=False,
        device="cuda",
    )
    model_load_seconds = time.perf_counter() - load_started

    try:
        for pending_index, row in enumerate(pending):
            sample_id = sample_id_from_row(row)
            record = status_by_id.setdefault(sample_id, base_status(row))
            artifact_path = shape_dir / f"{sample_id}.obj"
            write_control(control_path, "shape", sample_id)
            started = time.perf_counter()
            try:
                with Image.open(row["path"]) as source:
                    image = source.convert("RGB")
                image = remover(image)
                generator = torch.Generator().manual_seed(int(row["seed"]))
                outputs = pipeline(
                    image=image,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    generator=generator,
                    octree_resolution=256,
                    num_chunks=200_000,
                    output_type="mesh",
                )
                mesh = export_to_trimesh(outputs)[0]
                mesh = reducer(mesh)
                mesh.export(artifact_path)
                if not validate_shape(artifact_path):
                    raise ValueError(f"invalid shape artifact: {artifact_path}")
                record["shape"] = {
                    "status": "success",
                    "artifact_path": str(artifact_path),
                    "model_load_seconds": round(model_load_seconds, 3),
                    "generation_seconds": round(time.perf_counter() - started, 3),
                    "vertices": int(len(mesh.vertices)),
                    "faces": int(len(mesh.faces)),
                }
            except Exception as error:
                record["shape"] = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(limit=8),
                    "total_seconds": round(time.perf_counter() - started, 3),
                }
                persist_status(selected, status_by_id)
                if is_cuda_oom(error):
                    write_safety_stop("shape", sample_id, peak_for_sample("shape", sample_id), "CUDA OOM")
                    raise
            finally:
                for name in ("mesh", "outputs", "image", "generator"):
                    if name in locals():
                        del locals()[name]
                torch.cuda.empty_cache()

            persist_status(selected, status_by_id)
            peak = peak_for_sample("shape", sample_id)
            if record["shape"]["status"] == "success":
                peak = enforce_sample_gate("shape", sample_id, stop_vram_mib)
                record["shape"]["peak_memory_used_mib"] = peak
                persist_status(selected, status_by_id)
            print(
                json.dumps(
                    {
                        "event": "shape_sample_complete",
                        "sample_id": sample_id,
                        "status": record["shape"]["status"],
                        "peak_memory_used_mib": peak,
                    }
                ),
                flush=True,
            )
    finally:
        write_control(control_path, "idle", "")
        del pipeline, remover, reducer
        gc.collect()
        torch.cuda.empty_cache()


def wait_for_shape_release(max_used_mib: int = 2_000, timeout_seconds: int = 60) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        used = current_gpu_used_mib()
        if used < max_used_mib:
            return
        gc.collect()
        time.sleep(2)
    raise RuntimeError(f"shape model did not release below {max_used_mib} MiB")


def run_paint_stage(
    selected: list[dict],
    status_by_id: dict[str, dict],
    control_path: Path,
    stop_vram_mib: int,
) -> None:
    import torch

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from torchvision_fix import apply_fix  # noqa: PLC0415

    apply_fix()
    from hy3dpaint.convert_utils import create_glb_with_pbr_materials  # noqa: PLC0415
    from hy3dpaint.textureGenPipeline import (  # noqa: PLC0415
        Hunyuan3DPaintConfig,
        Hunyuan3DPaintPipeline,
    )

    pending = [
        row
        for row in selected
        if not needs_stage(row, status_by_id, "shape", validate_shape)
        and needs_stage(row, status_by_id, "paint", validate_glb)
    ]
    if not pending:
        print(json.dumps({"event": "paint_complete", "processed": 0}), flush=True)
        return

    glb_dir = OUTPUT_ROOT / "glb"
    paint_obj_dir = OUTPUT_ROOT / "paint_obj"
    glb_dir.mkdir(parents=True, exist_ok=True)
    paint_obj_dir.mkdir(parents=True, exist_ok=True)
    write_control(control_path, "paint", "__model_load__")
    load_started = time.perf_counter()
    config = Hunyuan3DPaintConfig(max_num_view=8, resolution=768)
    config.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    config.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    config.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    config.dino_ckpt_path = os.environ.get(
        "HUNYUAN_DINO_MODEL", str(PROJECT_ROOT / "models" / "dinov2-giant")
    )
    pipeline = Hunyuan3DPaintPipeline(config)
    model_load_seconds = time.perf_counter() - load_started

    try:
        for pending_index, row in enumerate(pending):
            sample_id = sample_id_from_row(row)
            record = status_by_id.setdefault(sample_id, base_status(row))
            shape_path = Path(record["shape"]["artifact_path"])
            output_obj = paint_obj_dir / f"{sample_id}.obj"
            glb_path = glb_dir / f"{sample_id}.glb"
            write_control(control_path, "paint", sample_id)
            started = time.perf_counter()
            try:
                textured_obj = Path(
                    pipeline(
                        mesh_path=str(shape_path),
                        image_path=row["path"],
                        output_mesh_path=str(output_obj),
                        save_glb=False,
                    )
                )
                textures = {
                    "albedo": str(textured_obj).replace(".obj", ".jpg"),
                    "metallic": str(textured_obj).replace(".obj", "_metallic.jpg"),
                    "roughness": str(textured_obj).replace(".obj", "_roughness.jpg"),
                }
                create_glb_with_pbr_materials(str(textured_obj), textures, str(glb_path))
                metrics = inspect_glb(glb_path)
                record["paint"] = {
                    "status": "success",
                    "artifact_path": str(glb_path),
                    "textured_obj_path": str(textured_obj),
                    "model_load_seconds": round(model_load_seconds, 3),
                    "paint_export_seconds": round(time.perf_counter() - started, 3),
                    **metrics,
                }
            except Exception as error:
                record["paint"] = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(limit=8),
                    "total_seconds": round(time.perf_counter() - started, 3),
                }
                persist_status(selected, status_by_id)
                if is_cuda_oom(error):
                    write_safety_stop("paint", sample_id, peak_for_sample("paint", sample_id), "CUDA OOM")
                    raise
            finally:
                torch.cuda.empty_cache()

            persist_status(selected, status_by_id)
            peak = peak_for_sample("paint", sample_id)
            if record["paint"]["status"] == "success":
                peak = enforce_sample_gate("paint", sample_id, stop_vram_mib)
                record["paint"]["peak_memory_used_mib"] = peak
                persist_status(selected, status_by_id)
            print(
                json.dumps(
                    {
                        "event": "paint_sample_complete",
                        "sample_id": sample_id,
                        "status": record["paint"]["status"],
                        "peak_memory_used_mib": peak,
                    }
                ),
                flush=True,
            )
    finally:
        write_control(control_path, "idle", "")
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    maximum = len(GROUP_IDS) * 4
    if not 1 <= args.limit <= maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        subprocess.run(
            build_stage_command(
                "shape", args.limit, args.stop_vram_mib, args.monitor_control
            ),
            check=True,
        )
        wait_for_shape_release()
        subprocess.run(
            build_stage_command(
                "paint", args.limit, args.stop_vram_mib, args.monitor_control
            ),
            check=True,
        )
        print(
            json.dumps(
                {"event": "complete", "stage": "all", "samples": args.limit}
            ),
            flush=True,
        )
        return

    source_rows = [
        json.loads(line)
        for line in SOURCE_MANIFEST.read_text(encoding="utf-8").splitlines()
    ]
    selected = select_records(source_rows, GROUP_IDS)[: args.limit]
    status_by_id = load_status()
    for row in selected:
        status_by_id.setdefault(sample_id_from_row(row), base_status(row))
    persist_status(selected, status_by_id)

    if args.stage == "shape":
        run_shape_stage(selected, status_by_id, args.monitor_control, args.stop_vram_mib)
    if args.stage == "paint":
        run_paint_stage(selected, status_by_id, args.monitor_control, args.stop_vram_mib)
    print(
        json.dumps(
            {
                "event": "complete",
                "stage": args.stage,
                "samples": len(selected),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
