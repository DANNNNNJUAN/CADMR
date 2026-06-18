"""Answer verification implementations for CADMR."""

import json

from cadmr.schemas import ActiveConstraint, MemoryJudgment


class AnswerVerifier:
    """No-op verifier used when no LLM verifier is configured."""

    def verify(
        self,
        answer: str,
        judgments: list[MemoryJudgment],
        constraints: list[ActiveConstraint],
        goal_plan: dict | None = None,
        structured_output: dict | None = None,
    ) -> dict:
        return {
            "pass": True,
            "violations": [],
            "missing_components": [],
            "needs_revision": False,
            "verifier_type": "noop",
        }


class LLMAnswerVerifier:
    """LLM verifier that checks answers against CADMR structured reasoning."""

    ALLOWED_VIOLATION_TYPES = {
        "constraint_violation",
        "stale_memory_use",
        "noise_use",
        "missing_goal_component",
        "unsupported_claim",
        "other",
    }

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def verify(
        self,
        answer: str,
        judgments: list[MemoryJudgment],
        constraints: list[ActiveConstraint],
        goal_plan: dict | None = None,
        structured_output: dict | None = None,
    ) -> dict:
        payload = structured_output or {
            "answer": answer,
            "judgments": self._jsonable(judgments),
            "retrieved_constraints": self._jsonable(constraints),
            "goal_plan": self._jsonable(goal_plan),
        }
        prompt = self._build_prompt(answer=answer, structured_output=payload)
        try:
            raw_result = self.llm_client.complete_json(prompt)
        except Exception as error:
            return self._fallback_result(f"LLM verifier failed: {error}")
        return self._validate_result(raw_result, payload)

    def _build_prompt(self, answer: str, structured_output: dict) -> str:
        structured_json = json.dumps(structured_output, ensure_ascii=False, indent=2)
        return f"""
You are a CADMR answer verifier.

Your job is to check whether the answer obeys the supplied CADMR structured reasoning.

Rules:
- Do not generate a new answer.
- Do not redo memory extraction, retrieval, or usability judgment.
- Do not override CADMR judgments.
- Only verify consistency between the answer and the supplied structured_output.
- Treat the supplied judgments as authoritative. If a memory is marked USABLE in
  structured_output.judgments, you must not report using that memory as noise_use
  or stale_memory_use.
- Report noise_use only when the answer relies on content from a memory whose
  supplied usage_status is NOISE.
- Report stale_memory_use only when the answer relies on content from a memory
  whose supplied usage_status is STALE as a current fact. It is allowed to say
  an old premise should not be assumed current.
- Do not flag summary or safety language as memory use unless the answer repeats
  concrete content from a disallowed memory.
- If a judgment is CONSTRAINED, the answer must not use that memory as direct action advice.
- If a judgment is STALE, the answer must not rely on stale memory as a current fact.
- If a judgment is NOISE, the answer must not use that memory as evidence.
- If active constraints exist, the answer must not recommend actions forbidden by those constraints.
- If goal_plan.required_plan_components exist, report missing components.
- If goal_plan.forbidden_actions exist, the answer must not recommend them.

Return strict JSON with this shape:
{{
  "pass": true,
  "violations": [
    {{
      "type": "constraint_violation | stale_memory_use | noise_use | missing_goal_component | unsupported_claim | other",
      "evidence": "short evidence from the answer",
      "related_id": "memory_id, constraint_id, or empty string"
    }}
  ],
  "missing_components": ["..."],
  "needs_revision": false,
  "reason": "short explanation"
}}

CADMR structured_output:
{structured_json}

Answer:
{answer}
""".strip()

    def _validate_result(self, raw_result: dict, structured_output: dict | None = None) -> dict:
        if not isinstance(raw_result, dict):
            return self._fallback_result("LLM verifier did not return a JSON object.")

        violations = raw_result.get("violations", [])
        if not isinstance(violations, list):
            violations = []
        violations = [self._normalize_violation(item) for item in violations]
        raw_violation_count = len(violations)
        violations = self._cleanup_violations(violations, structured_output or {})
        removed_violation_count = raw_violation_count - len(violations)

        missing_components = raw_result.get("missing_components", [])
        if not isinstance(missing_components, list):
            missing_components = []
        missing_components = [str(item) for item in missing_components if str(item).strip()]

        passed = not violations and not missing_components
        needs_revision = not passed
        reason = raw_result.get("reason")
        if removed_violation_count and passed:
            reason = (
                "Verifier-reported violations were removed because they contradicted "
                "CADMR's supplied judgments."
            )
        elif removed_violation_count:
            reason = (
                f"{removed_violation_count} verifier-reported violation(s) were removed "
                "because they contradicted CADMR's supplied judgments; remaining issues "
                "still require revision."
            )
        elif not isinstance(reason, str) or not reason.strip():
            reason = "LLM verifier completed."

        return {
            "pass": passed,
            "violations": violations,
            "missing_components": missing_components,
            "needs_revision": needs_revision,
            "reason": reason,
            "verifier_type": "llm",
        }

    def _cleanup_violations(
        self,
        violations: list[dict],
        structured_output: dict,
    ) -> list[dict]:
        status_by_id = self._judgment_status_by_id(structured_output)
        cleaned = []
        for violation in violations:
            violation_type = violation.get("type")
            related_id = violation.get("related_id", "")
            related_status = status_by_id.get(related_id)
            evidence = violation.get("evidence", "")

            if related_status == "USABLE" and violation_type in {
                "noise_use",
                "stale_memory_use",
                "constraint_violation",
            }:
                continue
            if violation_type == "stale_memory_use" and self._is_stale_rejection(evidence):
                continue
            cleaned.append(violation)
        return cleaned

    def _judgment_status_by_id(self, structured_output: dict) -> dict[str, str]:
        judgments = structured_output.get("judgments", [])
        if not isinstance(judgments, list):
            judgments = []
        status_by_id = {}
        for judgment in judgments:
            if not isinstance(judgment, dict):
                continue
            memory_id = judgment.get("memory_id")
            usage_status = judgment.get("usage_status")
            if isinstance(memory_id, str) and isinstance(usage_status, str):
                status_by_id[memory_id] = usage_status
        return status_by_id

    def _is_stale_rejection(self, evidence: str) -> bool:
        lowered = str(evidence).lower()
        return any(
            phrase in lowered
            for phrase in [
                "would not assume",
                "not assume",
                "older premise",
                "not current",
                "no longer current",
                "outdated premise",
            ]
        )

    def _normalize_violation(self, item) -> dict:
        if isinstance(item, str):
            return {"type": "other", "evidence": item, "related_id": ""}
        if not isinstance(item, dict):
            return {"type": "other", "evidence": str(item), "related_id": ""}

        violation_type = item.get("type", "other")
        if violation_type not in self.ALLOWED_VIOLATION_TYPES:
            violation_type = "other"
        evidence = item.get("evidence", "")
        related_id = item.get("related_id", "")
        return {
            "type": violation_type,
            "evidence": str(evidence),
            "related_id": str(related_id),
        }

    def _fallback_result(self, reason: str) -> dict:
        return {
            "pass": False,
            "violations": [{"type": "other", "evidence": reason, "related_id": ""}],
            "missing_components": [],
            "needs_revision": True,
            "reason": reason,
            "verifier_type": "llm",
        }

    def _jsonable(self, value):
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        if isinstance(value, dict):
            return {key: self._jsonable(item) for key, item in value.items()}
        return value
