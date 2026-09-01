from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path


def runtime_path(name: str, default: str, environ: dict[str, str] | None = None) -> Path:
    values = os.environ if environ is None else environ
    return Path(values.get(name, default))


ROOT = runtime_path("R3D_HUNYUAN_ROOT", "/root/r3dguard/data/hunyuan_16_test")
TRELLIS_STATUS = runtime_path(
    "R3D_TRELLIS_STATUS", "/root/r3dguard/data/trellis_16_test/status.jsonl"
)


def camera_positions(
    center: tuple[float, float, float],
    distance: float,
    count: int = 8,
    elevation_degrees: float = 20.0,
) -> list[tuple[float, float, float]]:
    elevation = math.radians(elevation_degrees)
    horizontal = distance * math.cos(elevation)
    vertical = distance * math.sin(elevation)
    return [
        (
            center[0] + horizontal * math.cos(2 * math.pi * index / count),
            center[1] + horizontal * math.sin(2 * math.pi * index / count),
            center[2] + vertical,
        )
        for index in range(count)
    ]


def choose_render_engine(available: set[str]) -> str:
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        if candidate in available:
            return candidate
    raise RuntimeError(f"no supported Blender render engine in {sorted(available)}")


def ensure_world(scene: object, worlds: object) -> object:
    if scene.world is None:
        scene.world = worlds.new("BenchmarkWorld")
    scene.world.color = (0.8, 0.8, 0.8)
    return scene.world


def summarize_telemetry(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"GPU telemetry is empty: {path}")

    def peak(stage: str | None = None) -> int | None:
        values = [
            int(row["memory_used_mib"])
            for row in rows
            if stage is None or row["stage"] == stage
        ]
        return max(values) if values else None

    return {
        "gpu_samples": len(rows),
        "peak_memory_used_mib": peak(),
        "shape_peak_memory_used_mib": peak("shape"),
        "paint_peak_memory_used_mib": peak("paint"),
        "max_utilization_gpu_percent": max(
            int(row["utilization_gpu_percent"]) for row in rows
        ),
        "max_temperature_c": max(int(row["temperature_c"]) for row in rows),
    }


def inspect_glb(path: Path) -> dict:
    import numpy as np
    import trimesh

    scene = trimesh.load(path, force="scene", process=False)
    geometries = list(scene.geometry.values())
    if not geometries:
        raise ValueError(f"GLB has no geometry: {path}")
    finite = all(
        np.isfinite(mesh.vertices).all() and np.isfinite(mesh.faces).all()
        for mesh in geometries
    )
    if not finite:
        raise ValueError(f"GLB contains non-finite geometry: {path}")
    textured = sum(
        getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None)
        is not None
        or getattr(getattr(mesh.visual, "material", None), "image", None) is not None
        for mesh in geometries
    )
    if textured == 0:
        raise ValueError(f"GLB contains no texture-bearing material: {path}")

    component_count = 0
    largest_component_faces = 0
    total_faces = sum(len(mesh.faces) for mesh in geometries)
    for mesh in geometries:
        topology = mesh.copy()
        topology.merge_vertices(merge_tex=True, merge_norm=True)
        components = trimesh.graph.connected_components(
            topology.face_adjacency,
            nodes=np.arange(len(topology.faces)),
            min_len=1,
            engine="scipy",
        )
        component_count += len(components)
        largest_component_faces = max(
            largest_component_faces,
            max((len(component) for component in components), default=0),
        )
    return {
        "glb_bytes": path.stat().st_size,
        "geometries": len(geometries),
        "vertices": sum(len(mesh.vertices) for mesh in geometries),
        "faces": total_faces,
        "finite": finite,
        "textured_geometries": textured,
        "watertight_geometries": sum(bool(mesh.is_watertight) for mesh in geometries),
        "connected_components": component_count,
        "largest_component_face_ratio": (
            largest_component_faces / total_faces if total_faces else 0.0
        ),
    }


def render_glb_views(glb_path: Path, output_dir: Path, count: int = 8) -> list[Path]:
    import bpy
    from mathutils import Vector

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / f"shaded_{index:02d}.png" for index in range(count)]
    if all(path.is_file() and path.stat().st_size > 0 for path in existing):
        return existing

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise ValueError(f"Blender imported no mesh objects: {glb_path}")

    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[axis] for point in corners) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in corners) for axis in range(3)))
    center = (minimum + maximum) * 0.5
    extent = max(maximum - minimum)
    distance = max(extent * 2.2, 1.0)

    scene = bpy.context.scene
    available_engines = {
        item.identifier
        for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    }
    scene.render.engine = choose_render_engine(available_engines)
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    ensure_world(scene, bpy.data.worlds)

    camera_data = bpy.data.cameras.new("BenchmarkCamera")
    camera = bpy.data.objects.new("BenchmarkCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.lens = 52

    for name, energy, offset, size in [
        ("Key", 1100.0, (2.5, -2.0, 3.0), 4.0),
        ("Fill", 650.0, (-2.5, -1.0, 1.5), 3.0),
        ("Rim", 850.0, (0.0, 3.0, 2.5), 3.0),
    ]:
        light_data = bpy.data.lights.new(name=name, type="AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = size * max(extent, 1.0)
        light = bpy.data.objects.new(name, light_data)
        light.location = center + Vector(offset) * max(extent, 1.0)
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(light)

    rendered = []
    for index, position in enumerate(
        camera_positions(tuple(center), distance=distance, count=count)
    ):
        camera.location = Vector(position)
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        target = output_dir / f"shaded_{index:02d}.png"
        scene.render.filepath = str(target)
        bpy.ops.render.render(write_still=True)
        rendered.append(target)
    return rendered


def build_contact_sheets(rows: list[dict], preview_by_id: dict[str, Path]) -> tuple[Path, Path | None]:
    from PIL import Image, ImageDraw, ImageOps

    tiles = []
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in preview_by_id:
            with Image.open(preview_by_id[sample_id]) as source:
                image = ImageOps.fit(source.convert("RGB"), (256, 256))
            tile = Image.new("RGB", (256, 288), "white")
            tile.paste(image, (0, 0))
        else:
            tile = Image.new("RGB", (256, 288), "#d88b8b")
        ImageDraw.Draw(tile).text((6, 264), sample_id, fill="black")
        tiles.append(tile)
    sheet = Image.new("RGB", (1024, ((len(tiles) + 3) // 4) * 288), "#dddddd")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 4) * 256, (index // 4) * 288))
    output = ROOT / "preview_contact_sheet.jpg"
    sheet.save(output, quality=90)

    if not TRELLIS_STATUS.exists():
        return output, None
    trellis_rows = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in TRELLIS_STATUS.read_text(encoding="utf-8").splitlines()
        )
    }
    pair_tiles = []
    for row in rows:
        sample_id = row["sample_id"]
        pair = Image.new("RGB", (512, 288), "#dddddd")
        trellis = trellis_rows.get(sample_id)
        trellis_preview = None
        if trellis and trellis.get("status") == "success":
            files = sorted(
                Path(trellis["render_dir"]).glob(
                    f"{trellis['render_primary_key']}_*.png"
                )
            )
            trellis_preview = files[0] if files else None
        for side, preview, label in [
            (0, trellis_preview, "TRELLIS.2"),
            (1, preview_by_id.get(sample_id), "Hunyuan3D-2.1"),
        ]:
            if preview and Path(preview).is_file():
                with Image.open(preview) as source:
                    image = ImageOps.fit(source.convert("RGB"), (256, 256))
                pair.paste(image, (side * 256, 0))
            ImageDraw.Draw(pair).text((side * 256 + 6, 260), label, fill="black")
        ImageDraw.Draw(pair).text((176, 276), sample_id, fill="black")
        pair_tiles.append(pair)
    paired = Image.new("RGB", (1024, ((len(pair_tiles) + 1) // 2) * 288), "#cccccc")
    for index, tile in enumerate(pair_tiles):
        paired.paste(tile, ((index % 2) * 512, (index // 2) * 288))
    paired_output = ROOT / "paired_contact_sheet.jpg"
    paired.save(paired_output, quality=90)
    return output, paired_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", type=int, default=16)
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in (ROOT / "status.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != args.expect:
        raise AssertionError(f"expected {args.expect} rows, got {len(rows)}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise AssertionError("duplicate sample IDs")

    valid = 0
    total_bytes = 0
    paint_times = []
    preview_by_id: dict[str, Path] = {}
    metrics_by_id = {}
    failures = []
    for row in rows:
        paint = row.get("paint", {})
        if paint.get("status") != "success":
            failures.append({"sample_id": row["sample_id"], "paint": paint})
            continue
        try:
            glb_path = Path(paint["artifact_path"])
            metrics = inspect_glb(glb_path)
            render_dir = ROOT / "renders" / row["sample_id"]
            renders = (
                sorted(render_dir.glob("shaded_*.png"))
                if args.skip_render
                else render_glb_views(glb_path, render_dir)
            )
            if len(renders) != 8:
                raise AssertionError(f"expected 8 renders for {row['sample_id']}")
            preview_by_id[row["sample_id"]] = renders[0]
            metrics_by_id[row["sample_id"]] = metrics
            valid += 1
            total_bytes += glb_path.stat().st_size
            if paint.get("paint_export_seconds") is not None:
                paint_times.append(float(paint["paint_export_seconds"]))
        except Exception as error:
            failures.append(
                {
                    "sample_id": row["sample_id"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    contact_sheet, paired_sheet = build_contact_sheets(rows, preview_by_id)
    telemetry = summarize_telemetry(ROOT / "gpu_memory.csv")
    report = {
        "ok": not failures and valid == args.expect,
        "samples": len(rows),
        "successes": valid,
        "failures": len(failures),
        "valid_glbs": valid,
        "median_paint_export_seconds": statistics.median(paint_times) if paint_times else None,
        "total_glb_bytes": total_bytes,
        "preview_contact_sheet": str(contact_sheet),
        "paired_contact_sheet": str(paired_sheet) if paired_sheet else None,
        "gpu": telemetry,
        "metrics_by_id": metrics_by_id,
        "failure_details": failures,
    }
    (ROOT / "verification_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
