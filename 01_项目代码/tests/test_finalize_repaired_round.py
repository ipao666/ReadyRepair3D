import importlib.util
import json
import hashlib
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "ops" / "finalize_repaired_round.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "finalize_repaired_round", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_labels(score=0.75, valid=True):
    return [
        {
            "sample_id": "winner",
            "quality_score_v2": score,
            "technically_valid": valid,
            "glb_path": "/assets/original.glb",
        },
        {
            "sample_id": "other",
            "quality_score_v2": 0.50,
            "technically_valid": True,
            "glb_path": "/assets/other.glb",
        },
    ]


def selection_report():
    return {"winner_sample_id": "winner"}


def repair_report(accepted):
    return {
        "accepted": accepted,
        "status": "repair_rollback" if accepted else "accept_original",
        "candidate_ready_for_rescore": accepted,
        "recommended_action": "remove_floaters",
        "rollback_reason": None if accepted else "no_change",
    }


def repaired_label(score=0.80, valid=True):
    return [
        {
            "sample_id": "winner_postrepair",
            "quality_score_v2": score,
            "technically_valid": valid,
            "glb_path": "/assets/repaired.glb",
        }
    ]


def test_rejected_repair_reuses_original_score():
    module = load_module()

    report = module.finalize_round(
        original_labels(0.82),
        selection_report(),
        repair_report(False),
    )

    assert report["delivered_variant"] == "original"
    assert report["final_quality_score_v2"] == 0.82
    assert report["v2_repair_accepted"] is False
    assert report["v2_rollback_reason"] == "no_change"
    assert report["meets_quality_threshold"] is True


def test_accepted_repair_with_higher_v2_wins():
    module = load_module()

    report = module.finalize_round(
        original_labels(0.75),
        selection_report(),
        repair_report(True),
        repaired_label(0.84),
    )

    assert report["delivered_variant"] == "repaired"
    assert report["final_quality_score_v2"] == 0.84
    assert report["v2_repair_accepted"] is True
    assert report["v2_rollback_reason"] is None
    assert report["repair3d_status"] == "repair_accepted"


def test_accepted_repair_with_lower_v2_rolls_back():
    module = load_module()

    report = module.finalize_round(
        original_labels(0.80),
        selection_report(),
        repair_report(True),
        repaired_label(0.79),
    )

    assert report["delivered_variant"] == "original"
    assert report["final_quality_score_v2"] == 0.80
    assert report["v2_repair_accepted"] is False
    assert report["v2_rollback_reason"] == "post_repair_gain_below_0.005"
    assert report["repair3d_status"] == "repair_rollback"


def test_repair_gain_below_point_zero_zero_five_rolls_back():
    module = load_module()

    report = module.finalize_round(
        original_labels(0.8000),
        selection_report(),
        repair_report(True),
        repaired_label(0.8049),
    )

    assert report["delivered_variant"] == "original"
    assert report["repair3d_status"] == "repair_rollback"
    assert report["v2_rollback_reason"] == "post_repair_gain_below_0.005"


def test_invalid_repair_rolls_back_even_with_higher_numeric_score():
    module = load_module()

    report = module.finalize_round(
        original_labels(0.70, valid=True),
        selection_report(),
        repair_report(True),
        repaired_label(0.99, valid=False),
    )

    assert report["delivered_variant"] == "original"
    assert report["final_technically_valid"] is True
    assert report["v2_rollback_reason"] == "post_repair_technically_invalid"


def test_invalid_original_sets_low_confidence_and_fails_threshold():
    module = load_module()

    report = module.finalize_round(
        original_labels(0.99, valid=False),
        selection_report(),
        repair_report(False),
    )

    assert report["final_technically_valid"] is False
    assert report["meets_quality_threshold"] is False
    assert report["low_confidence"] is True
    assert report["quality_threshold"] == 0.806912747446761


def test_finalize_artifacts_rollback_is_sha_identical(tmp_path):
    module = load_module()
    original = tmp_path / "original.glb"
    candidate = tmp_path / "candidate.glb"
    final = tmp_path / "final.glb"
    original.write_bytes(b"original-bytes")
    candidate.write_bytes(b"candidate-bytes")
    report_path = tmp_path / "repair_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "repair_rollback",
                "original_path": str(original),
                "candidate_path": str(candidate),
                "final_path": str(final),
            }
        ),
        encoding="utf-8",
    )
    round_report = {
        "v2_repair_accepted": False,
        "v2_rollback_reason": "post_repair_gain_below_0.005",
        "original_quality_score_v2": 0.80,
        "post_repair_quality_score_v2": 0.804,
    }

    module.finalize_repair_artifacts(round_report, report_path)

    updated = json.loads(report_path.read_text(encoding="utf-8"))
    assert final.read_bytes() == original.read_bytes()
    assert candidate.read_bytes() == b"candidate-bytes"
    assert updated["status"] == "repair_rollback"
    assert updated["sha256"]["original"] == updated["sha256"]["final"]


def test_v2_report_rejects_labels_not_bound_to_candidate(tmp_path):
    module = load_module()
    candidate = tmp_path / "candidate.glb"
    candidate.write_bytes(b"current-candidate")
    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    report = {
        "schema_version": "r3dguard.repair3d-auto.v2",
        "status": "repair_rollback",
        "candidate_ready_for_rescore": True,
        "candidate_path": str(candidate),
        "sha256": {"candidate": candidate_sha},
    }
    wrong = repaired_label(0.90)
    wrong[0]["sample_id"] = "winner_postrepair_wrongsha"
    wrong[0]["glb_path"] = str(candidate)

    with pytest.raises(ValueError, match="candidate"):
        module.finalize_round(
            original_labels(0.70), selection_report(), report, wrong
        )


def test_v2_report_rejects_candidate_changed_after_scoring(tmp_path):
    module = load_module()
    candidate = tmp_path / "candidate.glb"
    candidate.write_bytes(b"scored-candidate")
    scored_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    report = {
        "schema_version": "r3dguard.repair3d-auto.v2",
        "status": "repair_rollback",
        "candidate_ready_for_rescore": True,
        "candidate_path": str(candidate),
        "sha256": {"candidate": scored_sha},
    }
    labels = repaired_label(0.90)
    labels[0]["sample_id"] = f"winner_postrepair_{scored_sha[:12]}"
    labels[0]["glb_path"] = str(candidate)
    candidate.write_bytes(b"changed-after-scoring")

    with pytest.raises(ValueError, match="candidate"):
        module.finalize_round(
            original_labels(0.70), selection_report(), report, labels
        )


def test_finalize_preserves_accept_original_status(tmp_path):
    module = load_module()
    original = tmp_path / "original.glb"
    candidate = tmp_path / "candidate.glb"
    original.write_bytes(b"same")
    candidate.write_bytes(b"same")
    report_path = tmp_path / "repair_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "accept_original",
                "candidate_ready_for_rescore": False,
            }
        ),
        encoding="utf-8",
    )
    module.finalize_repair_artifacts(
        {
            "v2_repair_accepted": False,
            "v2_rollback_reason": None,
            "original_quality_score_v2": 0.90,
            "post_repair_quality_score_v2": None,
        },
        report_path,
    )

    updated = json.loads(report_path.read_text(encoding="utf-8"))
    assert updated["status"] == "accept_original"
