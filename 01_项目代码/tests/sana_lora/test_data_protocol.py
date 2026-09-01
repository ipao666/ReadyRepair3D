from __future__ import annotations

import pytest

from r3dloop.sana_lora.data_protocol import (
    expand_prompt_candidates,
    validate_lora_samples,
)


def sample(**updates):
    row = {
        "sample_id": "train_0001_c0",
        "prompt_group_id": "train_0001",
        "split": "train",
        "prompt_zh": "一只蓝白陶瓷茶壶，完整壶嘴和把手，单物体，纯色背景",
        "caption_en": "a complete blue and white ceramic teapot, isolated single object",
        "seed": 2026082801,
        "image_path": "images/train_0001_c0.png",
        "base_model": "SANA1.5_1.6B_1024px_diffusers",
    }
    row.update(updates)
    return row


def test_rejects_duplicate_sample_id():
    with pytest.raises(ValueError, match="duplicate"):
        validate_lora_samples([sample(), sample()])


def test_rejects_invalid_split_and_missing_seed():
    with pytest.raises(ValueError, match="invalid split"):
        validate_lora_samples([sample(split="holdout")])
    row = sample()
    row.pop("seed")
    with pytest.raises(ValueError, match="missing fields"):
        validate_lora_samples([row])


def test_rejects_prompt_group_crossing_splits():
    rows = [sample(), sample(sample_id="train_0001_c1", split="validation")]
    with pytest.raises(ValueError, match="cross splits"):
        validate_lora_samples(rows)


def test_rejects_quality_outside_unit_interval():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_lora_samples([sample(prompt_adherence=1.01)])


def test_training_manifest_rejects_test_samples():
    with pytest.raises(ValueError, match="non-training"):
        validate_lora_samples([sample(split="final_test")], training_only=True)


def test_development_candidate_expansion_is_deterministic():
    groups = [
        {
            "prompt_group_id": "lora_train_000",
            "split": "train",
            "category": "tools",
            "subject_key": "caliper",
            "subject_zh": "blue caliper",
            "prompt_zh": "single blue caliper centered on a plain background with all parts visible",
        },
        {
            "prompt_group_id": "lora_validation_001",
            "split": "validation",
            "category": "decor",
            "subject_key": "sundial",
            "subject_zh": "brass sundial",
            "prompt_zh": "single brass sundial centered on a plain background with all parts visible",
        },
    ]
    rows = expand_prompt_candidates(groups, base_seed=100)
    assert len(rows) == 8
    assert [row["seed"] for row in rows] == list(range(100, 108))
    assert {row["split"] for row in rows[:4]} == {"train"}
    assert len({row["sample_id"] for row in rows}) == 8


def test_final_test_cannot_enter_development_candidate_expansion():
    group = {
        "prompt_group_id": "lora_final_000",
        "split": "final_test",
        "category": "science",
        "subject_key": "pendulum",
        "subject_zh": "pendulum",
        "prompt_zh": "single pendulum centered on a plain background with all parts visible",
    }
    with pytest.raises(ValueError, match="final_test"):
        expand_prompt_candidates([group], base_seed=100)
