import json

from scripts.score_stale_answers import score_answers_file


def test_score_answers_file_writes_expected_output(tmp_path):
    answers_path = tmp_path / "answers.json"
    output_path = tmp_path / "scores.json"
    answers = [
        {
            "uid": "u1",
            "target_model_responses": {
                "dim1_response": "A current constraint prevents direct action from the old preference.",
                "dim2_response": "Do not plan directly from that premise.",
                "dim3_response": "Choose a mild meal.",
            },
            "target_model_meta": {
                "dim1_meta": {"judgments": [{"usage_status": "CONSTRAINED"}]},
                "dim2_meta": {"verify_result": {"pass": True, "violations": []}},
                "dim3_meta": {"verify_result": {"pass": True, "violations": []}},
            },
        }
    ]
    answers_path.write_text(json.dumps(answers, ensure_ascii=False), encoding="utf-8")

    result = score_answers_file(answers_path, output_path)

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["scorer_type"] == "rule_based_stale_compatible_not_official"
    assert written["scorer_type"] == "rule_based_stale_compatible_not_official"
    assert "stale_style_scores" in written
    assert "cadmr_internal_metrics" in written
    assert "details" in written
