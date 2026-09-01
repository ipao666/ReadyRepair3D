#!/usr/bin/env python3
"""Extract reproducible Ready3D image features from local pretrained models."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel, AutoModelForDepthEstimation


FEATURE_VERSION = 1
SAFE_SAMPLE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def gpu_preflight() -> dict:
    process = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True, check=False,
    )
    if process.returncode != 0 or not process.stdout.strip():
        detail = process.stderr.strip() or "nvidia-smi returned no GPU rows"
        raise RuntimeError(f"GPU preflight failed: {detail}")
    rows = []
    for line in process.stdout.splitlines():
        name, total, used = (value.strip() for value in line.rsplit(",", 2))
        rows.append({"name": name, "memory_total_mib": int(total), "memory_used_mib": int(used)})
    return {"command": "nvidia-smi", "gpus": rows}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_manifest_rows(rows: list[dict]) -> list[dict]:
    """Accept both research manifests and raw SANA generation manifests."""
    normalized = []
    seen: set[str] = set()
    for original in rows:
        row = dict(original)
        source_path = row.get("source_path") or row.get("path")
        filename = row.get("filename") or (Path(source_path).name if source_path else None)
        sample_id = row.get("sample_id") or (Path(filename).stem if filename else None)
        if not sample_id or not source_path:
            raise ValueError("manifest row requires sample_id/source_path or filename/path")
        group_id = row.get("group_id")
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(f"manifest row requires a non-empty group_id: {sample_id}")
        if not SAFE_SAMPLE_ID.fullmatch(str(sample_id)):
            raise ValueError(f"unsafe sample_id: {sample_id}")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(str(sample_id))
        row["sample_id"] = str(sample_id)
        row["source_path"] = str(source_path)
        row.setdefault("domain", "ai_sana")
        row.setdefault("split", "unassigned")
        normalized.append(row)
    return normalized


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros_like(depth)
    low, high = np.percentile(depth[finite], [2, 98])
    if high <= low + 1e-8:
        return np.zeros_like(depth)
    return np.clip((depth - low) / (high - low), 0.0, 1.0)


def depth_normal_features(depth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    normalized = normalize_depth(depth)
    valid = np.asarray(mask, dtype=bool)
    if not valid.any():
        valid = np.ones_like(normalized, dtype=bool)
    gy, gx = np.gradient(normalized)
    nz = np.ones_like(normalized)
    length = np.sqrt(gx * gx + gy * gy + nz * nz)
    nx, ny, nz = -gx / length, -gy / length, nz / length
    gradient = np.hypot(gx, gy)
    values = normalized[valid]
    return {
        "depth_mean": float(values.mean()),
        "depth_std": float(values.std()),
        "depth_q10": float(np.quantile(values, 0.10)),
        "depth_q50": float(np.quantile(values, 0.50)),
        "depth_q90": float(np.quantile(values, 0.90)),
        "depth_gradient_mean": float(gradient[valid].mean()),
        "depth_discontinuity_ratio": float((gradient[valid] > 0.08).mean()),
        "normal_x_mean": float(nx[valid].mean()),
        "normal_y_mean": float(ny[valid].mean()),
        "normal_z_mean": float(nz[valid].mean()),
        "normal_z_std": float(nz[valid].std()),
    }


def mask_occlusion_features(mask_probability: np.ndarray) -> dict[str, float]:
    probability = np.asarray(mask_probability, dtype=np.float32)
    mask = probability >= 0.5
    height, width = mask.shape
    if mask.any():
        ys, xs = np.nonzero(mask)
        margin = min(xs.min(), width - 1 - xs.max(), ys.min(), height - 1 - ys.max())
        bbox_margin = margin / max(1, min(height, width))
        hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        convex_fill = float(mask.sum() / hull_area)
    else:
        bbox_margin, convex_fill = 0.0, 0.0
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    labels, component_count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:] if component_count else np.array([])
    largest_ratio = float(sizes.max() / max(mask.sum(), 1)) if sizes.size else 0.0
    uncertainty = probability[(probability > 0.1) & (probability < 0.9)]
    return {
        "mask_area_ratio": float(mask.mean()),
        "mask_probability_mean": float(probability.mean()),
        "mask_uncertain_ratio": float(uncertainty.size / probability.size),
        "mask_component_count": float(component_count),
        "mask_largest_component_ratio": largest_ratio,
        "mask_convex_fill_ratio": convex_fill,
        "occlusion_border_contact_ratio": float(border.mean()),
        "occlusion_bbox_margin_ratio": float(bbox_margin),
    }


def image_features(image: Image.Image) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    histogram /= histogram.sum()
    nonzero = histogram[histogram > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    return {
        "image_brightness_mean": float(gray.mean() / 255.0),
        "image_brightness_std": float(gray.std() / 255.0),
        "image_saturation_mean": float(hsv[..., 1].mean() / 255.0),
        "image_entropy": entropy / 8.0,
        "image_dark_clip_ratio": float((gray <= 5).mean()),
        "image_light_clip_ratio": float((gray >= 250).mean()),
        "image_sharpness": float(math.log1p(cv2.Laplacian(gray, cv2.CV_64F).var())),
    }


def cache_paths(cache_dir: Path, sample_id: str) -> tuple[Path, Path]:
    if not SAFE_SAMPLE_ID.fullmatch(sample_id):
        raise ValueError(f"unsafe sample_id for feature cache: {sample_id}")
    return cache_dir / f"{sample_id}.npy", cache_dir / f"{sample_id}.json"


def load_feature_cache(cache_dir: Path, sample_id: str) -> tuple[np.ndarray, dict] | None:
    embedding_path, scalar_path = cache_paths(cache_dir, sample_id)
    if not embedding_path.is_file() or not scalar_path.is_file():
        return None
    try:
        embedding = np.load(embedding_path)
        payload = json.loads(scalar_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("feature_version") != FEATURE_VERSION:
        return None
    scalars = payload.get("scalars", {})
    if embedding.shape != (1024,) or not np.isfinite(embedding).all():
        return None
    if not scalars or not all(np.isfinite(float(value)) for value in scalars.values()):
        return None
    return embedding.astype(np.float32), scalars


def save_feature_cache(cache_dir: Path, sample_id: str, embedding: np.ndarray, scalars: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedding_path, scalar_path = cache_paths(cache_dir, sample_id)
    embedding_temporary = embedding_path.with_suffix(".npy.tmp")
    scalar_temporary = scalar_path.with_suffix(".json.tmp")
    with embedding_temporary.open("wb") as handle:
        np.save(handle, np.asarray(embedding, dtype=np.float32))
        handle.flush(); os.fsync(handle.fileno())
    scalar_temporary.write_text(
        json.dumps({"feature_version": FEATURE_VERSION, "scalars": scalars}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(embedding_temporary, embedding_path)
    os.replace(scalar_temporary, scalar_path)


class FeatureModels:
    def __init__(self, dino: Path, depth: Path, birefnet: Path, device: str):
        from transformers import AutoModelForImageSegmentation

        self.device = torch.device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.dino_processor = AutoImageProcessor.from_pretrained(dino, local_files_only=True)
        self.dino = AutoModel.from_pretrained(dino, local_files_only=True).to(self.device, dtype=dtype).eval()
        self.depth_processor = AutoImageProcessor.from_pretrained(depth, local_files_only=True)
        self.depth = AutoModelForDepthEstimation.from_pretrained(depth, local_files_only=True).to(self.device, dtype=dtype).eval()
        self.biref = AutoModelForImageSegmentation.from_pretrained(
            birefnet, trust_remote_code=True, local_files_only=True
        ).to(self.device, dtype=dtype).eval()
        self.mask_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.dtype = dtype

    @torch.inference_mode()
    def extract(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dino_inputs = self.dino_processor(images=image, return_tensors="pt")
        dino_inputs = {key: value.to(self.device, dtype=self.dtype) for key, value in dino_inputs.items()}
        embedding = self.dino(**dino_inputs).last_hidden_state[:, 0].float().cpu().numpy()[0]

        depth_inputs = self.depth_processor(images=image, return_tensors="pt")
        depth_inputs = {key: value.to(self.device, dtype=self.dtype) for key, value in depth_inputs.items()}
        predicted_depth = self.depth(**depth_inputs).predicted_depth[:, None]
        predicted_depth = torch.nn.functional.interpolate(
            predicted_depth, size=(image.height, image.width), mode="bicubic", align_corners=False
        )[0, 0].float().cpu().numpy()

        mask_input = self.mask_transform(image).unsqueeze(0).to(self.device, dtype=self.dtype)
        mask = self.biref(mask_input)[-1].sigmoid()
        mask = torch.nn.functional.interpolate(
            mask, size=(image.height, image.width), mode="bilinear", align_corners=False
        )[0, 0].float().cpu().numpy()
        return embedding.astype(np.float32), predicted_depth.astype(np.float32), mask.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dino", type=Path, default=Path("/root/r3dguard/models/dinov2-large"))
    parser.add_argument("--depth", type=Path, default=Path("/root/r3dguard/models/Depth-Anything-V2-Large-hf"))
    parser.add_argument("--birefnet", type=Path, default=Path("/root/r3dguard/models/BiRefNet"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = normalize_manifest_rows(read_jsonl(args.manifest))
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    pending = [row for row in rows if load_feature_cache(cache_dir, row["sample_id"]) is None]
    preflight = None
    if pending and args.device == "cuda":
        preflight = gpu_preflight()
        print(json.dumps({"event": "gpu_preflight", **preflight}), flush=True)
    elif args.device == "cuda":
        existing_schema = args.output_dir / "feature_schema.json"
        if existing_schema.is_file():
            try:
                preflight = json.loads(existing_schema.read_text(encoding="utf-8")).get("gpu_preflight")
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                preflight = None
    models = FeatureModels(args.dino, args.depth, args.birefnet, args.device) if pending else None
    embeddings, records = [], []
    for index, row in enumerate(rows):
        cached = load_feature_cache(cache_dir, row["sample_id"])
        if cached is None:
            with Image.open(row["source_path"]) as source:
                image = source.convert("RGB")
            if models is None:
                raise RuntimeError("feature models were not initialized")
            embedding, depth, mask_probability = models.extract(image)
            scalars = image_features(image)
            scalars.update(mask_occlusion_features(mask_probability))
            scalars.update(depth_normal_features(depth, mask_probability >= 0.5))
            save_feature_cache(cache_dir, row["sample_id"], embedding, scalars)
            cache_hit = False
        else:
            embedding, scalars = cached
            cache_hit = True
        records.append({
            "feature_version": FEATURE_VERSION,
            "feature_index": index,
            "sample_id": row["sample_id"],
            "group_id": row["group_id"],
            "domain": row["domain"],
            "split": row["split"],
            "source_path": row["source_path"],
            **scalars,
        })
        embeddings.append(embedding)
        print(json.dumps({"event": "feature_complete", "index": index, "sample_id": row["sample_id"], "cache_hit": cache_hit}), flush=True)

    np.save(args.output_dir / "dino_embeddings.npy", np.stack(embeddings))
    with (args.output_dir / "features.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    schema = {
        "feature_version": FEATURE_VERSION,
        "device": args.device,
        "gpu_preflight": preflight,
        "samples": len(records),
        "dino_dimensions": int(embeddings[0].shape[0]),
        "dino_model": str(args.dino),
        "depth_model": str(args.depth),
        "mask_model": str(args.birefnet),
        "normal_source": "finite_differences_of_depth_anything_v2_relative_depth",
        "occlusion_source": "birefnet_mask_border_contact_components_and_bbox_margin",
        "scalar_features": sorted(key for key in records[0] if key.startswith(("image_", "mask_", "depth_", "normal_", "occlusion_"))),
    }
    (args.output_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
