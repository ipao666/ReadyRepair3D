from __future__ import annotations

from pathlib import Path

import pytest

from ops.sana_lora.train_quality_lora import load_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "sana_lora" / "train_rank16.yaml"


def test_frozen_rank16_config_is_valid():
    config = load_config(CONFIG)
    assert config["rank"] == 16
    assert config["max_steps"] == 1500
    assert config["trainable_modules"] == ["to_q", "to_k", "to_v"]


def test_rejects_changes_to_frozen_training_contract(tmp_path):
    text = CONFIG.read_text(encoding="utf-8").replace("rank: 16", "rank: 8")
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="frozen experiment"):
        load_config(path)
