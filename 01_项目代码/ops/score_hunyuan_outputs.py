#!/usr/bin/env python3
"""Score Hunyuan3D outputs and emit bootstrap Ready3D pseudo-labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from image_match_metrics import select_best_view
from mesh_quality_metrics import inspect_glb
from ready3d_quality_v2 import (
    fit_high_quality_threshold,
    fit_normalization,
    score_record,
)


SCORING_VERSION = "ready3d-v2"


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collect_samples(status_path: Path, render_root: Path, expect: int | None = None) -> list[dict]:
    rows = read_jsonl(status_path)
    if expect is not None and len(rows) != expect:
        raise ValueError(f"Expected {expect} status rows, found {len(rows)}")
    samples = []
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        sample_id = row["sample_id"]
        paint = row.get("paint", {})
        if paint.get("status") != "success":
            raise ValueError(f"{sample_id} paint status is not success")
        source_path = Path(row["source_path"])
        glb_path = Path(paint["artifact_path"])
        render_paths = sorted((Path(render_root) / sample_id).glob("shaded_*.png"))
        if len(render_paths) != 8:
            raise ValueError(f"{sample_id} must have exactly 8 shaded render views")
        if not source_path.is_file() or not glb_path.is_file() or any(not p.is_file() for p in render_paths):
            raise ValueError(f"{sample_id} has missing source, GLB, or render files")
        samples.append(
            {
                "sample_id": sample_id,
                "source_path": source_path,
                "glb_path": glb_path,
                "render_paths": render_paths,
            }
        )
    if len({row["sample_id"] for row in samples}) != len(samples):
        raise ValueError("Duplicate sample IDs")
    return samples


def _masked_crop(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    mask = np.asarray(mask, dtype=bool)
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return Image.fromarray(rgb)
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0) + 1
    crop = rgb[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    crop[~crop_mask] = 127
    return Image.fromarray(crop)


class VisionModels:
    def __init__(self, dino_path: Path, birefnet_path: Path, device: str = "cuda") -> None:
        import torch
        from torchvision import transforms
        from transformers import AutoImageProcessor, AutoModel, AutoModelForImageSegmentation

        self.torch = torch
        self.device = device
        self.segment_transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.segment_model = AutoModelForImageSegmentation.from_pretrained(
            str(birefnet_path), trust_remote_code=True, local_files_only=True
        ).eval().to(device)
        self.dino_processor = AutoImageProcessor.from_pretrained(
            str(dino_path), local_files_only=True
        )
        self.dino_model = AutoModel.from_pretrained(
            str(dino_path), local_files_only=True
        ).eval().to(device)

    def segment(self, image: Image.Image) -> np.ndarray:
        tensor = self.segment_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            prediction = self.segment_model(tensor)[-1].sigmoid()[0, 0]
        mask_image = Image.fromarray(
            (prediction.float().cpu().numpy() * 255).astype(np.uint8)
        ).resize(image.size, Image.Resampling.BILINEAR)
        return np.asarray(mask_image) >= 128

    def embed(self, images: list[Image.Image]) -> list[np.ndarray]:
        inputs = self.dino_processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.dino_model(**inputs).last_hidden_state[:, 0]
            output = self.torch.nn.functional.normalize(output.float(), dim=-1)
        return [row for row in output.cpu().numpy()]


def measure_sample(sample: dict, vision: VisionModels) -> dict:
    mesh = inspect_glb(sample["glb_path"])
    if not mesh.get("glb_loadable"):
        raise ValueError(f"Cannot load {sample['glb_path']}: {mesh.get('load_error')}")
    paths = [sample["source_path"], *sample["render_paths"]]
    images = [Image.open(path).convert("RGB") for path in paths]
    try:
        masks = [vision.segment(image) for image in images]
        crops = [_masked_crop(image, mask) for image, mask in zip(images, masks)]
        embeddings = vision.embed(crops)
    finally:
        for image in images:
            image.close()
    match = select_best_view(embeddings[0], embeddings[1:], masks[0], masks[1:])
    raw = {
        "sample_id": sample["sample_id"],
        "source_path": str(sample["source_path"]),
        "glb_path": str(sample["glb_path"]),
        "render_count": len(sample["render_paths"]),
        **mesh,
        **match,
    }
    return raw


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = [
        "sample_id", "quality_score_v2", "technically_valid", "relative_top30", "high_quality",
        "input_match", "geometry", "texture_available", "best_view_index",
        "largest_component_area_ratio", "debris_area_ratio", "component_count_raw",
        "effective_component_count", "non_manifold_edge_length_ratio",
        "boundary_edge_length_ratio", "textured_surface_ratio", "technical_failure_reasons",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row[key], ensure_ascii=False)
                if key == "technical_failure_reasons" else row.get(key)
                for key in columns
            })


def assign_split(rows: list[dict], split: str) -> list[dict]:
    return [{**row, "split": split} for row in rows]


def finalize_scored(rows: list[dict], threshold: float, calibration_source: str) -> list[dict]:
    bootstrap = calibration_source.startswith("bootstrap")
    finalized = []
    for source_row in rows:
        row = dict(source_row)
        technically_valid = bool(row["technically_valid"])
        relative_top30 = bool(
            technically_valid and row["quality_score_v2"] >= threshold
        )
        row.update(
            {
                "technically_valid": technically_valid,
                "relative_top30": relative_top30,
                "high_quality": None if bootstrap else relative_top30,
                "threshold": threshold,
                "calibration_source": calibration_source,
            }
        )
        finalized.append(row)
    return finalized


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_metrics.jsonl"
    samples = collect_samples(args.status, args.render_root, args.expect)
    if raw_path.is_file() and not args.refresh:
        raw_rows = read_jsonl(raw_path)
        if [row["sample_id"] for row in raw_rows] != [row["sample_id"] for row in samples]:
            raise ValueError("Raw metric cache does not match current sample manifest")
    else:
        vision = VisionModels(args.dino_model, args.birefnet_model, args.device)
        raw_rows = []
        for index, sample in enumerate(samples, 1):
            raw_rows.append(measure_sample(sample, vision))
            write_jsonl(raw_path, raw_rows)
            print(f"measured {index}/{len(samples)} {sample['sample_id']}", flush=True)

    if args.calibration_in:
        calibration_report = json.loads(args.calibration_in.read_text(encoding="utf-8"))
        calibration = calibration_report["metrics"]
        threshold = float(calibration_report["quality_threshold"])
        calibration_source = calibration_report["source"]
    else:
        normalization = fit_normalization(
            assign_split(raw_rows, "validation"),
            required_split="validation",
        )
        calibration = normalization["metrics"]
        calibration_source = args.fit_source
    scored = []
    for raw in raw_rows:
        result = score_record(raw, calibration)
        scored.append({
            "sample_id": raw["sample_id"],
            "scoring_version": SCORING_VERSION,
            **raw,
            **result,
        })
    if not args.calibration_in:
        threshold_report = fit_high_quality_threshold(
            assign_split(scored, "validation")
        )
        threshold = float(threshold_report["threshold"])
        calibration_report = {
            "scoring_version": SCORING_VERSION,
            "source": calibration_source,
            "quality_threshold_percentile": int(
                threshold_report["threshold_percentile"]
            ),
            "quality_threshold": threshold,
            "metrics": calibration,
            "normalization": normalization,
            "test_used_for_fitting": False,
        }
    scored = finalize_scored(scored, threshold, calibration_source)

    write_jsonl(args.output_dir / "quality_labels.jsonl", scored)
    write_csv(args.output_dir / "quality_labels.csv", scored)
    (args.output_dir / "calibration.json").write_text(
        json.dumps(calibration_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "scoring_version": SCORING_VERSION,
        "samples": len(scored),
        "technically_valid": sum(row["technically_valid"] for row in scored),
        "relative_top30": sum(row["relative_top30"] for row in scored),
        "high_quality": (
            None if calibration_source.startswith("bootstrap")
            else sum(row["high_quality"] for row in scored)
        ),
        "quality_threshold": threshold,
        "calibration_source": calibration_source,
        "median_quality_score_v2": float(
            np.median([row["quality_score_v2"] for row in scored])
        ),
        "median_largest_component_area_ratio": float(
            np.median([row["largest_component_area_ratio"] for row in scored])
        ),
        "median_component_count_raw": float(
            np.median([row["component_count_raw"] for row in scored])
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dino-model", type=Path, required=True)
    parser.add_argument("--birefnet-model", type=Path, required=True)
    parser.add_argument("--expect", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--calibration-in", type=Path)
    parser.add_argument("--fit-source", default="bootstrap_16")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
