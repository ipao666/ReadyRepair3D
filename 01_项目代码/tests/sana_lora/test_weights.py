from __future__ import annotations

import pytest

from r3dloop.sana_lora.weights import compute_quality_weight, select_3d_quality


def test_fixed_quality_weight_formula_at_boundaries():
    assert compute_quality_weight(0.0, 0.0) == (0.0, 0.25)
    assert compute_quality_weight(1.0, 1.0) == (1.0, 1.5)
    combined, weight = compute_quality_weight(0.8, 0.5)
    assert combined == pytest.approx(0.71)
    assert weight == pytest.approx(1.1375)


def test_missing_adherence_uses_only_3d_quality():
    combined, weight = compute_quality_weight(0.6, None)
    assert combined == 0.6
    assert weight == 1.0


def test_out_of_range_scores_are_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        compute_quality_weight(-0.01, 0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        compute_quality_weight(0.5, 1.01)


def test_frozen_hunyuan_quality_has_priority():
    quality, source = select_3d_quality(
        {"hunyuan_quality": 0.8, "ready3d_calibrated_quality": 0.1}
    )
    assert quality == 0.8
    assert source == "hunyuan_frozen_q"


def test_requires_real_or_calibrated_3d_label():
    with pytest.raises(ValueError, match="neither"):
        select_3d_quality({})
