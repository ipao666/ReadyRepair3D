from __future__ import annotations

from ops.sana_lora.build_hunyuan_queue import build_queue


def test_queue_has_exact_train_validation_counts_and_no_test_labels():
    candidates = []
    predictions = []
    features = []
    for index in range(960):
        split = "train" if index < 720 else "validation" if index < 840 else "dev_test"
        sample_id = f"sample_{index:04d}"
        candidate = {
            "sample_id": sample_id,
            "group_id": f"group_{index // 4:03d}",
            "prompt_group_id": f"group_{index // 4:03d}",
            "candidate_index": index % 4,
            "split": split,
            "category": f"category_{index % 15}",
            "source_path": f"/tmp/{sample_id}.png",
        }
        candidates.append(candidate)
        if split == "train":
            predictions.append(
                {
                    **candidate,
                    "predicted_quality": index / 719,
                    "structural_input_pass": index % 13 != 0,
                }
            )
        features.append(
            {
                "sample_id": sample_id,
                "mask_convex_fill_ratio": 0.7 if index % 7 == 0 else 0.9,
                "depth_discontinuity_ratio": 0.04 if index % 11 == 0 else 0.01,
            }
        )
    queue, coverage = build_queue(candidates, predictions, features)
    assert len(queue) == 300
    assert coverage["train_hunyuan"] == 180
    assert coverage["validation_hunyuan"] == 120
    assert coverage["final_test_labels"] == 0
    assert {row["split"] for row in queue} == {"train", "validation"}
    assert set(coverage["quality_quantiles"]) == {"0", "1", "2", "3", "4"}
