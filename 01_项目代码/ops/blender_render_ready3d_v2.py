#!/usr/bin/env python3
"""Blender worker: render eight fixed views from one existing GLB."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


def camera_positions(center, distance: float, count: int = 8, elevation_degrees: float = 20.0):
    elevation = math.radians(elevation_degrees)
    horizontal, vertical = distance * math.cos(elevation), distance * math.sin(elevation)
    return [
        (
            center[0] + horizontal * math.cos(2 * math.pi * index / count),
            center[1] + horizontal * math.sin(2 * math.pi * index / count),
            center[2] + vertical,
        )
        for index in range(count)
    ]


def choose_render_engine(available: set[str], headless: bool) -> str:
    preferred = (
        ("CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH")
        if headless
        else ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES", "BLENDER_WORKBENCH")
    )
    try:
        return next(name for name in preferred if name in available)
    except StopIteration as error:
        raise RuntimeError(f"no supported Blender render engine found: {sorted(available)}") from error


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    return parser.parse_args(values)


def main() -> None:
    import numpy as np

    if "bool" not in np.__dict__:
        np.bool = bool
    import bpy
    from mathutils import Vector

    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.glb))
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise ValueError(f"Blender imported no mesh objects: {args.glb}")
    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[axis] for point in corners) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in corners) for axis in range(3)))
    center = (minimum + maximum) * 0.5
    extent = max(maximum - minimum); distance = max(extent * 2.2, 1.0)

    scene = bpy.context.scene
    available = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    scene.render.engine = choose_render_engine(available, headless=not bool(os.environ.get("DISPLAY")))
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = 8
        scene.cycles.max_bounces = 2
        scene.cycles.diffuse_bounces = 1
        scene.cycles.glossy_bounces = 1
        scene.cycles.transparent_max_bounces = 1
    scene.render.resolution_x = scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100; scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if scene.world is None: scene.world = bpy.data.worlds.new("Ready3DV2World")
    scene.world.color = (0.8, 0.8, 0.8)

    camera_data = bpy.data.cameras.new("Ready3DV2Camera")
    camera = bpy.data.objects.new("Ready3DV2Camera", camera_data)
    scene.collection.objects.link(camera); scene.camera = camera; camera_data.lens = 52
    for name, energy, offset, size in (
        ("Key", 1100.0, (2.5, -2.0, 3.0), 4.0),
        ("Fill", 650.0, (-2.5, -1.0, 1.5), 3.0),
        ("Rim", 850.0, (0.0, 3.0, 2.5), 3.0),
    ):
        light_data = bpy.data.lights.new(name=name, type="AREA")
        light_data.energy = energy; light_data.shape = "DISK"; light_data.size = size * max(extent, 1.0)
        light = bpy.data.objects.new(name, light_data)
        light.location = center + Vector(offset) * max(extent, 1.0)
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(light)

    outputs = []
    for index, position in enumerate(camera_positions(tuple(center), distance)):
        target = args.output / f"shaded_{index:02d}.png"
        if target.is_file() and target.stat().st_size > 0:
            outputs.append(str(target))
            continue
        camera.location = Vector(position)
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(target); bpy.ops.render.render(write_still=True); outputs.append(str(target))
    print("READY3D_V2_RENDER=" + json.dumps({"glb": str(args.glb), "outputs": outputs}), flush=True)


if __name__ == "__main__":
    main()
