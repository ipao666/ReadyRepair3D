"""Pure mask and embedding metrics used by the Hunyuan quality scorer."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt


def align_mask(mask: np.ndarray, size: int = 256, margin: int = 16) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return np.zeros((size, size), dtype=bool)
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0) + 1
    crop = mask[y0:y1, x0:x1]
    available = size - 2 * margin
    scale = min(available / crop.shape[1], available / crop.shape[0])
    width = max(1, int(round(crop.shape[1] * scale)))
    height = max(1, int(round(crop.shape[0] * scale)))
    resized = np.asarray(
        Image.fromarray(crop.astype(np.uint8) * 255).resize(
            (width, height), Image.Resampling.NEAREST
        )
    ) > 0
    output = np.zeros((size, size), dtype=bool)
    x = (size - width) // 2
    y = (size - height) // 2
    output[y : y + height, x : x + width] = resized
    return output


def silhouette_iou(first: np.ndarray, second: np.ndarray) -> float:
    first, second = align_mask(first), align_mask(second)
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(first, second).sum() / union)


def _boundary(mask: np.ndarray) -> np.ndarray:
    return np.logical_xor(mask, binary_erosion(mask))


def contour_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first, second = align_mask(first), align_mask(second)
    edge_a, edge_b = _boundary(first), _boundary(second)
    if not edge_a.any() or not edge_b.any():
        return 0.0
    distance_to_b = distance_transform_edt(~edge_b)
    distance_to_a = distance_transform_edt(~edge_a)
    symmetric_distance = 0.5 * (
        float(distance_to_b[edge_a].mean()) + float(distance_to_a[edge_b].mean())
    )
    diagonal = np.hypot(*first.shape)
    return float(np.clip(1.0 - 4.0 * symmetric_distance / diagonal, 0.0, 1.0))


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).ravel()
    second = np.asarray(second, dtype=np.float64).ravel()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        raise ValueError("DINO embeddings must have non-zero norm")
    return float(np.dot(first, second) / denominator)


def select_best_view(
    source_embedding: np.ndarray,
    view_embeddings: list[np.ndarray],
    source_mask: np.ndarray,
    view_masks: list[np.ndarray],
) -> dict:
    if not view_embeddings or len(view_embeddings) != len(view_masks):
        raise ValueError("View embeddings and masks must be non-empty and aligned")
    similarities = [_cosine(source_embedding, embedding) for embedding in view_embeddings]
    best_index = int(np.argmax(similarities))
    best_mask = view_masks[best_index]
    return {
        "best_view_index": best_index,
        "dino_similarity": similarities[best_index],
        "silhouette_iou": silhouette_iou(source_mask, best_mask),
        "contour_similarity": contour_similarity(source_mask, best_mask),
        "view_dino_similarities": similarities,
    }
