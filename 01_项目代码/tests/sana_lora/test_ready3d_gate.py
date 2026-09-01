from __future__ import annotations

from ops.sana_lora.audit_ready3d_v3_gate import audit_gate


def payloads(v3_quality=0.7, random_quality=0.6, validation_top1=0.4):
    model = {
        "model_selection": {"split": "validation", "test_used_for_selection": False},
        "validation": {"top1_accuracy": validation_top1, "quality_capture": 0.9},
    }
    strategies = {
        "test_used_for_model_or_threshold_selection": False,
        "strategies": {
            "ready_v3_top1": {"mean_quality": v3_quality},
            "random_top1": {"mean_quality": random_quality},
        },
    }
    checkpoint = {"schema_version": "r3dguard.ready3d-v3-top1-checkpoint.v1"}
    return model, strategies, checkpoint


def test_gate_allows_pseudo_labels_only_when_all_checks_pass():
    report = audit_gate(*payloads())
    assert report["allow_ready3d_v3_pseudo_labels"] is True
    assert all(report["checks"].values())


def test_gate_rejects_v3_not_above_random_or_weak_validation():
    assert audit_gate(*payloads(v3_quality=0.5))["allow_ready3d_v3_pseudo_labels"] is False
    assert audit_gate(*payloads(validation_top1=0.25))[
        "allow_ready3d_v3_pseudo_labels"
    ] is False
