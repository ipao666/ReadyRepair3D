import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from image_match_metrics import (  # noqa: E402
    align_mask,
    contour_similarity,
    select_best_view,
    silhouette_iou,
)


def rectangle(shape, y0, y1, x0, x1):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_align_mask_removes_translation_and_scale_without_changing_aspect():
    first = rectangle((80, 100), 10, 50, 20, 40)
    second = rectangle((160, 200), 50, 130, 120, 160)
    assert np.array_equal(align_mask(first), align_mask(second))


def test_identical_aligned_masks_have_perfect_scores():
    mask = rectangle((64, 64), 8, 56, 20, 44)
    assert silhouette_iou(mask, mask) == 1.0
    assert contour_similarity(mask, mask) == 1.0


def test_best_dino_view_controls_both_silhouette_metrics():
    source_embedding = np.array([1.0, 0.0])
    view_embeddings = [np.array([0.5, 0.5]), np.array([1.0, 0.0])]
    source_mask = rectangle((64, 64), 8, 56, 20, 44)
    perfect_mask = source_mask.copy()
    different_mask = rectangle((64, 64), 20, 44, 8, 56)

    result = select_best_view(
        source_embedding,
        view_embeddings,
        source_mask,
        [perfect_mask, different_mask],
    )

    assert result["best_view_index"] == 1
    assert result["dino_similarity"] == 1.0
    assert result["silhouette_iou"] < 1.0
    assert result["contour_similarity"] < 1.0
