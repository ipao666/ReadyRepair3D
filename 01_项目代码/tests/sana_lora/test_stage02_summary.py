from __future__ import annotations

from PIL import Image

from ops.sana_lora.summarize_stage02 import summarize


def test_stage02_summary_records_complete_960_set(tmp_path):
    image = tmp_path / "shared.png"
    Image.new("RGB", (1024, 1024), "white").save(image)
    manifest = []
    adherence = []
    for index in range(960):
        split = "train" if index < 720 else "validation" if index < 840 else "dev_test"
        sample_id = f"sample_{index:04d}"
        manifest.append(
            {
                "sample_id": sample_id,
                "split": split,
                "seed": 1000 + index,
                "source_path": str(image),
            }
        )
        adherence.append(
            {
                "sample_id": sample_id,
                "adherence_missing": index == 0,
                "prompt_adherence": None if index == 0 else 0.9,
            }
        )
    summary = summarize(manifest, adherence)
    assert summary["candidates_success"] == 960
    assert summary["unique_seeds"] == 960
    assert summary["adherence_scored"] == 959
    assert summary["final_test_used"] is False
