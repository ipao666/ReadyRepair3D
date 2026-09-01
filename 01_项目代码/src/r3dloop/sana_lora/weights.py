"""Quality weighting for SANA LoRA samples."""

from __future__ import annotations

from .data_protocol import validate_unit_interval


def select_3d_quality(label: dict) -> tuple[float, str]:
    """Prefer frozen Hunyuan quality, otherwise use calibrated Ready3D V3."""
    if label.get("hunyuan_quality") is not None:
        return validate_unit_interval(label["hunyuan_quality"], "hunyuan_quality"), "hunyuan_frozen_q"
    if label.get("quality") is not None and label.get("label_source") == "hunyuan_frozen_q":
        return validate_unit_interval(label["quality"], "quality"), "hunyuan_frozen_q"
    if label.get("ready3d_calibrated_quality") is not None:
        return (
            validate_unit_interval(
                label["ready3d_calibrated_quality"], "ready3d_calibrated_quality"
            ),
            "ready3d_v3_calibrated",
        )
    if label.get("quality") is not None and label.get("label_source") == "ready3d_v3_calibrated":
        return validate_unit_interval(label["quality"], "quality"), "ready3d_v3_calibrated"
    raise ValueError("label has neither frozen Hunyuan nor calibrated Ready3D quality")


def compute_quality_weight(
    calibrated_3d_quality: float,
    prompt_adherence: float | None,
) -> tuple[float, float]:
    quality = validate_unit_interval(calibrated_3d_quality, "calibrated_3d_quality")
    if prompt_adherence is None:
        combined = quality
    else:
        adherence = validate_unit_interval(prompt_adherence, "prompt_adherence")
        combined = 0.70 * quality + 0.30 * adherence
    weight = min(1.50, max(0.25, 0.25 + 1.25 * combined))
    return combined, weight


def build_weighted_record(candidate: dict, label: dict, adherence: dict | None) -> dict:
    if candidate.get("split") != "train":
        raise ValueError(
            f"non-training sample entered training manifest: {candidate.get('sample_id')}/{candidate.get('split')}"
        )
    quality, label_source = select_3d_quality(label)
    adherence_missing = adherence is None or bool(adherence.get("adherence_missing", False))
    adherence_value = None if adherence_missing else adherence.get("prompt_adherence")
    combined, weight = compute_quality_weight(quality, adherence_value)
    return {
        **candidate,
        "calibrated_3d_quality": quality,
        "label_source": label_source,
        "prompt_adherence": adherence_value,
        "adherence_missing": adherence_missing,
        "combined_quality": combined,
        "sample_weight": weight,
    }
