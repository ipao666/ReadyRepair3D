from __future__ import annotations

import pytest

from ops.sana_lora.score_prompt_adherence import parse_adherence_response, score_rows


def test_parses_exact_six_dimension_scores():
    parsed = parse_adherence_response(
        '{"subject":1,"count":0.9,"color":0.8,"material":0.7,"parts":0.6,"structure":0.5}'
    )
    assert parsed["prompt_adherence"] == pytest.approx(0.75)
    assert parsed["adherence_missing"] is False


def test_rejects_missing_or_out_of_range_dimensions():
    with pytest.raises(ValueError, match="invalid adherence keys"):
        parse_adherence_response('{"subject":1}')
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        parse_adherence_response(
            '{"subject":1.1,"count":1,"color":1,"material":1,"parts":1,"structure":1}'
        )


def test_parse_failure_is_recorded_without_fabricating_scores(tmp_path):
    rows = [{"sample_id": "sample_1"}]
    results = score_rows(rows, lambda row: "not-json", checkpoint_path=tmp_path / "scores.jsonl")
    assert results[0]["adherence_missing"] is True
    assert results[0]["prompt_adherence"] is None
