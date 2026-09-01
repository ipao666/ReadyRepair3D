import sys
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from build_validation64_manifest import select_validation_groups  # noqa: E402


def synthetic_rows():
    rows = []
    difficulties = ["easy", "medium", "hard", "medium", "hard"]
    for category_index in range(10):
        for group_index in range(5):
            group_id = f"g_{category_index}_{group_index}"
            for candidate_index in range(4):
                rows.append(
                    {
                        "group_id": group_id,
                        "candidate_index": candidate_index,
                        "category": f"category_{category_index}",
                        "difficulty": difficulties[group_index],
                    }
                )
    return rows


def test_selection_is_deterministic_balanced_and_group_isolated():
    rows = synthetic_rows()
    excluded = {"g_0_0", "g_1_0", "g_2_0", "g_3_0"}
    first = select_validation_groups(rows, excluded, seed=20260716)
    second = select_validation_groups(rows, excluded, seed=20260716)
    assert first == second
    assert len(first) == 16
    assert not excluded.intersection(first)

    group_meta = {row["group_id"]: row for row in rows}
    categories = Counter(group_meta[group]["category"] for group in first)
    difficulties = Counter(group_meta[group]["difficulty"] for group in first)
    assert len(categories) == 10
    assert max(categories.values()) <= 2
    assert all(difficulties[name] >= 4 for name in ("easy", "medium", "hard"))
