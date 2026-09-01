import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from build_ready3d_splits import build_group_splits  # noqa: E402


def rows():
    output = []
    difficulties = ["easy", "medium", "hard"]
    for group_index in range(50):
        for candidate_index in range(4):
            output.append(
                {
                    "group_id": f"g{group_index:02d}",
                    "candidate_index": candidate_index,
                    "category": f"c{group_index % 10}",
                    "difficulty": difficulties[group_index % 3],
                }
            )
    return output


def test_splits_are_deterministic_disjoint_and_use_all_remaining_groups():
    excluded = {"g00", "g01", "g02", "g03"}
    validation = {f"g{i:02d}" for i in range(4, 20)}
    first = build_group_splits(rows(), excluded, validation, seed=20260716)
    second = build_group_splits(rows(), excluded, validation, seed=20260716)
    assert first == second
    assert len(first["train"]) == 24
    assert len(first["test"]) == 6
    assert not set(first["train"]) & set(first["test"])
    assert not (set(first["train"]) | set(first["test"])) & (excluded | validation)
    assert len(set(first["test_categories"])) >= 6
    assert set(first["test_difficulties"]) == {"easy", "medium", "hard"}
