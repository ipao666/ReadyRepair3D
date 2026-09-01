import json
from pathlib import Path


from scripts.aggregate_final_results import summarize_repairs


def _write_report(path: Path, *, accepted: bool, before: float, after: float) -> None:
    path.write_text(
        json.dumps(
            {
                "accepted": accepted,
                "executed_action": "cleanup" if accepted else "rollback_original",
                "quality_before": {"quality_score": before},
                "quality_after": {
                    "quality_score": after,
                    "trimesh_loadable": True,
                    "blender_loadable": True,
                    "finite": True,
                },
            }
        ),
        encoding="utf-8",
    )


def test_summarize_repairs_separates_acceptance_from_safe_rollback(tmp_path: Path) -> None:
    _write_report(tmp_path / "a.json", accepted=True, before=0.60, after=0.70)
    _write_report(tmp_path / "b.json", accepted=False, before=0.80, after=0.80)
    _write_report(tmp_path / "c.json", accepted=False, before=0.75, after=0.74)

    summary = summarize_repairs(sorted(tmp_path.glob("*.json")))

    assert summary["report_count"] == 3
    assert summary["accepted_count"] == 1
    assert summary["rollback_count"] == 2
    assert summary["no_harm_count"] == 2
    assert summary["loadable_after_count"] == 3


def test_summarize_repairs_does_not_call_passthrough_a_repair(tmp_path: Path) -> None:
    path = tmp_path / "accepted.json"
    path.write_text(
        json.dumps(
            {
                "accepted": True,
                "executed_action": "accept",
                "quality_before": {"quality_score": 0.8},
                "quality_after": {
                    "quality_score": 0.8,
                    "trimesh_loadable": True,
                    "blender_loadable": True,
                    "finite": True,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_repairs([path])

    assert summary["passthrough_count"] == 1
    assert summary["modified_count"] == 0
    assert summary["modified_and_accepted_count"] == 0
