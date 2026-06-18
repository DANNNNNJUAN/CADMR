from cadmr.metrics import (
    CADMRMetricScorer,
    StaleDimensionScorer,
    summarize_cadmr_metrics,
    summarize_stale_scores,
)


def meta_with_status(status: str) -> dict:
    return {
        "judgments": [{"usage_status": status}],
        "verify_result": {"pass": True, "violations": []},
    }


def test_dim1_sr_uses_structured_meta_not_response_keywords():
    scorer = StaleDimensionScorer()

    passed, _, signals = scorer.score_dim1_sr("A current constraint prevents direct use of the old preference.", {})
    assert passed is False
    assert signals["has_constrained_judgment"] is False

    passed, _, signals = scorer.score_dim1_sr("It can be used as background.", meta_with_status("CONSTRAINED"))
    assert passed is True
    assert signals["has_constrained_judgment"] is True


def test_dim2_pr_uses_structured_meta_not_response_keywords():
    passed, _, signals = StaleDimensionScorer().score_dim2_pr("Do not plan directly from that premise.", {})

    assert passed is False
    assert signals["has_constrained_judgment"] is False
    assert signals["has_stale_judgment"] is False


def test_dim1_sr_does_not_count_verify_only_as_state_resolution():
    meta = {
        "signals": [{"signal_type": "question_premise"}],
        "write_decisions": [{"decision": "VERIFY_ONLY"}],
        "query_info": {"query_intent": "question_with_premise"},
    }

    passed, reason, signals = StaleDimensionScorer().score_dim1_sr("The old state needs verification.", meta)

    assert passed is False
    assert signals["has_premise_verification_signal"] is True
    assert reason == "No explicit stale/constrained state-resolution signal found."


def test_dim2_pr_does_not_count_suspended_as_premise_resistance():
    meta = {"judgments": [{"usage_status": "SUSPENDED"}]}

    passed, reason, signals = StaleDimensionScorer().score_dim2_pr("This premise needs confirmation first.", meta)

    assert passed is False
    assert signals["has_suspended_judgment"] is True
    assert reason == "No premise-resistance signal found in CADMR meta."


def test_dim2_pr_does_not_keyword_scan_for_spicy_violation():
    passed, reason, signals = StaleDimensionScorer().score_dim2_pr("I recommend very spicy hotpot.", {})

    assert passed is False
    assert "possible_violation" not in signals
    assert reason == "No premise-resistance signal found in CADMR meta."


def test_dim3_ipa_does_not_count_verifier_pass_without_current_state_signal():
    passed, reason, signals = StaleDimensionScorer().score_dim3_ipa(
        "Adapt to the current constraint.",
        {"verify_result": {"pass": True}},
    )

    assert passed is False
    assert signals["verify_pass"] is True
    assert reason == "No current-state action-adaptation signal found in CADMR meta."


def test_dim3_ipa_passes_with_usable_current_evidence_and_verifier_pass():
    meta = {
        "judgments": [{"usage_status": "USABLE"}],
        "verify_result": {"pass": True, "violations": []},
    }

    passed, reason, signals = StaleDimensionScorer().score_dim3_ipa("Answer from the current memory.", meta)

    assert passed is True
    assert signals["has_usable_judgment"] is True
    assert reason == "CADMR action output is grounded in current-state judgments and passes verification."


def test_dim3_ipa_does_not_count_usable_evidence_without_verifier_pass():
    meta = {"judgments": [{"usage_status": "USABLE"}]}

    passed, reason, signals = StaleDimensionScorer().score_dim3_ipa("Answer from the current memory.", meta)

    assert passed is False
    assert signals["has_usable_judgment"] is True
    assert reason == "CADMR has current-state judgments but verifier does not confirm the final action."


def test_dim3_ipa_fails_when_only_noise_and_verifier_fails():
    meta = {
        "judgments": [{"usage_status": "NOISE"}],
        "verify_result": {"pass": False, "violations": ["unsupported"]},
    }

    passed, reason, signals = StaleDimensionScorer().score_dim3_ipa("Use cautiously.", meta)

    assert passed is False
    assert signals["has_noise_judgment"] is True
    assert reason == "No current-state action-adaptation signal found in CADMR meta."


def test_scorer_reads_structured_output_when_top_level_meta_is_compact():
    meta = {
        "structured_output": {
            "judgments": [{"usage_status": "STALE"}],
            "signals": [{"signal_type": "question_premise"}],
            "write_decisions": [{"decision": "VERIFY_ONLY"}],
            "query_info": {"query_intent": "question_with_premise"},
        }
    }

    passed, _, signals = StaleDimensionScorer().score_dim1_sr("The old state has been updated.", meta)

    assert passed is True
    assert signals["has_stale_judgment"] is True
    assert signals["has_premise_verification_signal"] is True


def test_summarize_stale_scores_counts_dimensions_and_overall():
    scored_items = [
        {
            "uid": "u1",
            "dim1": {"metric": "SR", "passed": True, "signals": {}},
            "dim2": {"metric": "PR", "passed": False, "signals": {}},
            "dim3": {"metric": "IPA", "passed": True, "signals": {}},
        }
    ]

    summary = summarize_stale_scores(scored_items)

    assert summary["SR"]["score"] == 1.0
    assert summary["PR"]["failed"] == 1
    assert summary["IPA"]["passed"] == 1
    assert summary["overall"]["total"] == 3
    assert summary["overall"]["passed"] == 2


def test_cadmr_metric_scorer_detects_internal_signals():
    item = {
        "uid": "u1",
        "target_model_meta": {
            "dim1_meta": {
                "judgments": [{"usage_status": "CONSTRAINED"}],
                "goal_plan": {"needs_goal_reconstruction": True},
                "verify_result": {"pass": True, "violations": ["bad"]},
            },
            "dim2_meta": {"judgments": [{"usage_status": "NOISE"}]},
            "dim3_meta": {},
        },
    }

    scored = CADMRMetricScorer().score_answer_item(item)

    assert scored["dim1"]["constrained_hit"] is True
    assert scored["dim2"]["noise_filter_hit"] is True
    assert scored["dim1"]["goal_reconstruction_hit"] is True
    assert scored["dim1"]["verifier_pass"] is False
    assert scored["dim1"]["constraint_violation"] is True


def test_summarize_cadmr_metrics_rates():
    scored_items = [
        {
            "uid": "u1",
            "dim1": {
                "constrained_hit": True,
                "stale_hit": False,
                "noise_filter_hit": False,
                "goal_reconstruction_hit": True,
                "verifier_pass": True,
                "constraint_violation": False,
            },
            "dim2": {
                "constrained_hit": False,
                "stale_hit": False,
                "noise_filter_hit": True,
                "goal_reconstruction_hit": False,
                "verifier_pass": False,
                "constraint_violation": True,
            },
            "dim3": {
                "constrained_hit": False,
                "stale_hit": True,
                "noise_filter_hit": False,
                "goal_reconstruction_hit": False,
                "verifier_pass": False,
                "constraint_violation": False,
            },
        }
    ]

    summary = summarize_cadmr_metrics(scored_items)

    assert summary["total_dims"] == 3
    assert summary["constrained_hit_rate"] == 1 / 3
    assert summary["noise_filter_rate"] == 1 / 3
    assert summary["goal_reconstruction_rate"] == 1 / 3
    assert summary["verifier_pass_rate"] == 1 / 3
    assert summary["constraint_violation_rate"] == 1 / 3
