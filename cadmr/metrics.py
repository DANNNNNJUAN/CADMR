"""Structure-based evaluation metrics for CADMR outputs."""

from dataclasses import asdict, dataclass, field


DIM_TO_METRIC = {"dim1": "SR", "dim2": "PR", "dim3": "IPA"}


@dataclass
class MetricResult:
    name: str
    score: float
    total: int
    passed: int
    failed: int
    details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class StaleDimensionScorer:
    """Scores STALE-style dimensions from CADMR structured metadata only."""

    def score_answer_item(self, item: dict) -> dict:
        uid = item.get("uid", "")
        responses = item.get("target_model_responses", {}) or {}
        meta = item.get("target_model_meta", {}) or {}
        dim1 = self.score_dim1_sr(
            responses.get("dim1_response", ""),
            meta.get("dim1_meta", {}) or {},
        )
        dim2 = self.score_dim2_pr(
            responses.get("dim2_response", ""),
            meta.get("dim2_meta", {}) or {},
        )
        dim3 = self.score_dim3_ipa(
            responses.get("dim3_response", ""),
            meta.get("dim3_meta", {}) or {},
        )
        return {
            "uid": uid,
            "dim1": self._dim_result("SR", dim1),
            "dim2": self._dim_result("PR", dim2),
            "dim3": self._dim_result("IPA", dim3),
        }

    def score_dim1_sr(self, response: str, meta: dict) -> tuple[bool, str, dict]:
        signals = self._meta_signals(meta)
        if not response.strip():
            return False, "Empty response.", signals
        if signals["has_stale_judgment"]:
            return True, "CADMR marks M_old as no longer currently valid.", signals
        if signals["has_constrained_judgment"]:
            return True, "CADMR marks M_old as unsafe to use directly under current conditions.", signals
        return False, "No explicit stale/constrained state-resolution signal found.", signals

    def score_dim2_pr(self, response: str, meta: dict) -> tuple[bool, str, dict]:
        signals = self._meta_signals(meta)
        if not response.strip():
            return False, "Empty response.", signals
        if signals["has_stale_judgment"] or signals["has_constrained_judgment"]:
            return True, "CADMR detects the false premise or updates it with current state.", signals
        return False, "No premise-resistance signal found in CADMR meta.", signals

    def score_dim3_ipa(self, response: str, meta: dict) -> tuple[bool, str, dict]:
        signals = self._meta_signals(meta)
        if not response.strip():
            return False, "Empty response.", signals
        has_current_state_signal = bool(
            signals["has_usable_judgment"]
            or signals["has_constrained_judgment"]
            or signals["has_stale_judgment"]
            or signals["has_goal_reconstruction"]
        )
        if has_current_state_signal and signals["verify_pass"]:
            return True, "CADMR action output is grounded in current-state judgments and passes verification.", signals
        if signals["has_goal_reconstruction"]:
            return False, "CADMR reconstructs a goal but verifier does not confirm a safe current-state action.", signals
        if has_current_state_signal:
            return False, "CADMR has current-state judgments but verifier does not confirm the final action.", signals
        return False, "No current-state action-adaptation signal found in CADMR meta.", signals

    def _dim_result(self, metric: str, scored: tuple[bool, str, dict]) -> dict:
        passed, reason, signals = scored
        return {"metric": metric, "passed": passed, "reason": reason, "signals": signals}

    def _judgment_signals(self, meta: dict) -> dict:
        statuses = self._judgment_statuses(meta)
        return {
            "has_usable_judgment": "USABLE" in statuses,
            "has_constrained_judgment": "CONSTRAINED" in statuses,
            "has_stale_judgment": "STALE" in statuses,
            "has_suspended_judgment": "SUSPENDED" in statuses,
            "has_noise_judgment": "NOISE" in statuses,
            "judgment_statuses": statuses,
        }

    def _has_any_usability_signal(self, signals: dict) -> bool:
        return bool(
            signals.get("has_usable_judgment")
            or signals.get("has_constrained_judgment")
            or signals.get("has_stale_judgment")
            or signals.get("has_suspended_judgment")
            or signals.get("has_noise_judgment")
        )

    def _meta_signals(self, meta: dict) -> dict:
        signals = self._judgment_signals(meta)
        goal_plan = self._first_mapping(meta, "goal_plan")
        query_info = self._first_mapping(meta, "query_info")
        extracted_signals = self._first_list(meta, "signals")
        write_decisions = self._first_list(meta, "write_decisions")
        retrieved_memories = self._first_list(meta, "retrieved_memories")
        retrieved_constraints = self._first_list(meta, "retrieved_constraints")

        signal_types = [
            signal.get("signal_type")
            for signal in extracted_signals
            if isinstance(signal, dict)
        ]
        decisions = [
            decision.get("decision")
            for decision in write_decisions
            if isinstance(decision, dict)
        ]
        query_intent = query_info.get("query_intent") if isinstance(query_info, dict) else None
        has_question_premise = "question_premise" in signal_types
        has_verify_only = "VERIFY_ONLY" in decisions
        signals.update(
            {
                "verify_pass": self._verify_pass(meta),
                "has_goal_reconstruction": bool(goal_plan.get("needs_goal_reconstruction")),
                "has_question_premise_signal": has_question_premise,
                "has_query_intent_signal": "query_intent" in signal_types,
                "has_verify_only_decision": has_verify_only,
                "has_premise_verification_signal": bool(
                    (has_question_premise or query_intent == "question_with_premise")
                    and has_verify_only
                ),
                "has_retrieved_evidence": bool(retrieved_memories or retrieved_constraints),
                "signal_types": signal_types,
                "write_decisions": decisions,
                "query_intent": query_intent,
            }
        )
        return signals

    def _judgment_statuses(self, meta: dict) -> list[str]:
        judgments = self._first_list(meta, "judgments")
        return [
            judgment.get("usage_status")
            for judgment in judgments
            if isinstance(judgment, dict)
        ]

    def _first_mapping(self, meta: dict, key: str) -> dict:
        value = meta.get(key)
        if isinstance(value, dict):
            return value
        structured_output = meta.get("structured_output", {}) or {}
        value = structured_output.get(key) if isinstance(structured_output, dict) else None
        return value if isinstance(value, dict) else {}

    def _first_list(self, meta: dict, key: str) -> list:
        value = meta.get(key)
        if isinstance(value, list):
            return value
        structured_output = meta.get("structured_output", {}) or {}
        value = structured_output.get(key) if isinstance(structured_output, dict) else None
        return value if isinstance(value, list) else []

    def _verify_pass(self, meta: dict) -> bool:
        verify_result = self._first_mapping(meta, "verify_result")
        return bool(verify_result.get("pass")) and not verify_result.get("violations")


def summarize_stale_scores(scored_items: list[dict]) -> dict:
    summary = {
        "SR": {"score": 0.0, "passed": 0, "failed": 0, "total": 0},
        "PR": {"score": 0.0, "passed": 0, "failed": 0, "total": 0},
        "IPA": {"score": 0.0, "passed": 0, "failed": 0, "total": 0},
        "overall": {"score": 0.0, "passed": 0, "failed": 0, "total": 0},
    }

    for item in scored_items:
        for dim, metric in DIM_TO_METRIC.items():
            dim_result = item.get(dim, {}) or {}
            if dim_result.get("signals", {}).get("skipped"):
                continue
            bucket = summary[metric]
            bucket["total"] += 1
            if dim_result.get("passed"):
                bucket["passed"] += 1
            else:
                bucket["failed"] += 1

    for metric in ["SR", "PR", "IPA"]:
        _finalize_score(summary[metric])
        summary["overall"]["passed"] += summary[metric]["passed"]
        summary["overall"]["failed"] += summary[metric]["failed"]
        summary["overall"]["total"] += summary[metric]["total"]
    _finalize_score(summary["overall"])
    return summary


class CADMRMetricScorer:
    """Scores CADMR internal meta signals."""

    def score_answer_item(self, item: dict) -> dict:
        meta = item.get("target_model_meta", {}) or {}
        return {
            "uid": item.get("uid", ""),
            "dim1": self._score_dim(meta.get("dim1_meta", {}) or {}),
            "dim2": self._score_dim(meta.get("dim2_meta", {}) or {}),
            "dim3": self._score_dim(meta.get("dim3_meta", {}) or {}),
        }

    def _score_dim(self, meta: dict) -> dict:
        judgments = meta.get("judgments", []) or []
        statuses = [
            judgment.get("usage_status")
            for judgment in judgments
            if isinstance(judgment, dict)
        ]
        goal_plan = meta.get("goal_plan", {}) or {}
        verify_result = meta.get("verify_result", {}) or {}
        return {
            "constrained_hit": "CONSTRAINED" in statuses,
            "stale_hit": "STALE" in statuses,
            "noise_filter_hit": "NOISE" in statuses,
            "goal_reconstruction_hit": bool(goal_plan.get("needs_goal_reconstruction")),
            "verifier_pass": bool(verify_result.get("pass")) and not verify_result.get("violations"),
            "constraint_violation": bool(verify_result.get("violations")),
        }


def summarize_cadmr_metrics(cadmr_scored_items: list[dict]) -> dict:
    metric_keys = [
        "constrained_hit",
        "stale_hit",
        "noise_filter_hit",
        "goal_reconstruction_hit",
        "verifier_pass",
        "constraint_violation",
    ]
    counts = {key: 0 for key in metric_keys}
    total_dims = 0
    for item in cadmr_scored_items:
        for dim in ["dim1", "dim2", "dim3"]:
            dim_result = item.get(dim, {}) or {}
            if dim_result.get("skipped"):
                continue
            total_dims += 1
            for key in metric_keys:
                if dim_result.get(key):
                    counts[key] += 1

    return {
        "constrained_hit_rate": _rate(counts["constrained_hit"], total_dims),
        "stale_hit_rate": _rate(counts["stale_hit"], total_dims),
        "noise_filter_rate": _rate(counts["noise_filter_hit"], total_dims),
        "goal_reconstruction_rate": _rate(counts["goal_reconstruction_hit"], total_dims),
        "verifier_pass_rate": _rate(counts["verifier_pass"], total_dims),
        "constraint_violation_rate": _rate(counts["constraint_violation"], total_dims),
        "total_dims": total_dims,
    }


def _finalize_score(bucket: dict) -> None:
    bucket["score"] = _rate(bucket["passed"], bucket["total"])


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
