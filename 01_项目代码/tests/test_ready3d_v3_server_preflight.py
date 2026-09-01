import json

import pytest

from ops.run_ready3d_v3_server_preflight import build_preflight


def _candidates(groups=2, base_seed=1000):
    return [
        {
            "schema_version": "r3dguard.ready3d-v3-candidate.v1",
            "group_id": f"v3_{group:03d}",
            "sample_id": f"v3_{group:03d}_c{candidate}_s{base_seed + group * 10 + candidate}",
            "candidate_index": candidate,
            "seed": base_seed + group * 10 + candidate,
            "filename": f"v3_{group:03d}_c{candidate}.png",
            "split": "test",
        }
        for group in range(groups)
        for candidate in range(4)
    ]


def test_preflight_builds_resume_plan_without_mutating_output(tmp_path):
    output = tmp_path / "new-output"

    report = build_preflight(_candidates(), output, expected_candidates=8)

    assert report["ready"] is True
    assert report["counts"]["generate_2d"] == 8
    assert report["counts"]["complete"] == 0
    assert len(report["resume_plan"]) == 8
    assert not output.exists()


def test_preflight_rejects_duplicate_ids_and_changed_seed_contract(tmp_path):
    duplicate = _candidates()
    duplicate[-1]["sample_id"] = duplicate[0]["sample_id"]
    with pytest.raises(ValueError, match="duplicate sample_id"):
        build_preflight(duplicate, tmp_path, expected_candidates=8)

    changed = _candidates()
    changed[3]["seed"] += 10
    with pytest.raises(ValueError, match="consecutive fixed seeds"):
        build_preflight(changed, tmp_path, expected_candidates=8)


def test_preflight_detects_partial_artifacts_and_completed_labels(tmp_path):
    candidates = _candidates()
    output = tmp_path / "run"
    images = output / "images"
    glbs = output / "hunyuan" / "glb"
    renders = output / "renders"
    images.mkdir(parents=True)
    glbs.mkdir(parents=True)
    sample0 = candidates[0]["sample_id"]
    sample1 = candidates[1]["sample_id"]
    (images / candidates[0]["filename"]).write_bytes(b"image")
    (images / candidates[1]["filename"]).write_bytes(b"image")
    (glbs / f"{sample1}.glb").write_bytes(b"glb")
    (renders / sample1).mkdir(parents=True)
    for view in range(8):
        (renders / sample1 / f"rgb_{view:02d}.png").write_bytes(b"png")
    labels = output / "quality_labels.jsonl"
    labels.write_text(json.dumps({"sample_id": sample1}) + "\n", encoding="utf-8")

    report = build_preflight(
        candidates, output, labels_path=labels, expected_candidates=8
    )

    status = {row["sample_id"]: row["next_stage"] for row in report["artifacts"]}
    assert status[sample0] == "generate_3d"
    assert status[sample1] == "complete"
    assert report["counts"]["complete"] == 1
