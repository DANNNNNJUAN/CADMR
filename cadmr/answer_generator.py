"""Constrained answer generation without domain keyword rules."""

import json
import re

from cadmr.schemas import ActiveConstraint, MemoryJudgment, OrdinaryMemory, QueryInfo


_CJK_RE = re.compile(r"[\u3400-\u9fff]")


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _english_user_visible_text(text: str, fallback: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if _contains_cjk(stripped):
        return fallback
    return stripped


class ConstrainedAnswerGenerator:
    """Generates user-facing answers from structured memory judgments."""

    def generate(
        self,
        query_info: QueryInfo | None,
        judgments: list[MemoryJudgment],
        constraints: list[ActiveConstraint],
        memories: list[OrdinaryMemory] | None = None,
        revision_context: dict | None = None,
    ) -> str:
        if query_info is None:
            return ""

        structured_answer = self.build_structured_answer(
            query_info,
            judgments,
            constraints,
            memories or [],
        )
        return self._render_structured_answer(structured_answer)

    def build_structured_answer(
        self,
        query_info: QueryInfo,
        judgments: list[MemoryJudgment],
        constraints: list[ActiveConstraint],
        memories: list[OrdinaryMemory] | None = None,
    ) -> dict:
        if not judgments:
            return {
                "answer": (
                    "I do not have enough relevant stored evidence to answer from memory."
                ),
                "evidence": [],
                "constraints": [],
                "recommendation": (
                    "Answer from the current question only and avoid assuming extra long-term background."
                ),
            }

        memories = memories or []
        memory_by_id = {memory.memory_id: memory for memory in memories}
        constrained_judgments = [
            judgment for judgment in judgments if judgment.usage_status == "CONSTRAINED"
        ]
        stale_judgments = [
            judgment for judgment in judgments if judgment.usage_status == "STALE"
        ]
        usable_judgments = [
            judgment for judgment in judgments if judgment.usage_status == "USABLE"
        ]
        suspended_judgments = [
            judgment for judgment in judgments if judgment.usage_status == "SUSPENDED"
        ]
        noise_judgments = [
            judgment for judgment in judgments if judgment.usage_status == "NOISE"
        ]

        if constrained_judgments:
            evidence = self._memory_texts(usable_judgments, memory_by_id)
            constraint_texts = self._blocking_constraint_texts(
                constrained_judgments,
                constraints,
            )
            return {
                "answer": (
                    "The relevant older fact or preference can be acknowledged only as background, "
                    "not as direct action guidance."
                ),
                "evidence": evidence,
                "constraints": constraint_texts,
                "recommendation": (
                    "Adapt the plan around the current constraints before giving any concrete action."
                ),
            }

        if stale_judgments:
            current_evidence = self._memory_texts(usable_judgments, memory_by_id)
            return {
                "answer": "I would not assume the older premise is still current.",
                "evidence": current_evidence,
                "constraints": [],
                "recommendation": (
                    "Base the answer on the newer or currently usable evidence instead of the older premise."
                ),
            }

        if usable_judgments:
            evidence = self._memory_texts(usable_judgments, memory_by_id)
            return {
                "answer": "The available relevant evidence can be used for this question.",
                "evidence": evidence,
                "constraints": [],
                "recommendation": "Give a direct answer grounded in the relevant evidence.",
            }

        if suspended_judgments:
            return {
                "answer": "The relevant evidence is uncertain and should be confirmed before relying on it.",
                "evidence": [],
                "constraints": [],
                "recommendation": "Ask for confirmation or answer cautiously from the current question only.",
            }

        if noise_judgments:
            return {
                "answer": "I do not have relevant stored evidence for this question.",
                "evidence": [],
                "constraints": [],
                "recommendation": (
                    "Answer from the current question only; do not assume extra background from storage."
                ),
            }

        return {
            "answer": "The available evidence is mixed and does not provide a reliable basis by itself.",
            "evidence": [],
            "constraints": [],
            "recommendation": "Answer cautiously from the current question and ask for clarification if needed.",
        }

    def _render_structured_answer(self, structured_answer: dict) -> str:
        parts = [f"Answer: {structured_answer['answer']}"]

        evidence = structured_answer.get("evidence") or []
        if evidence:
            parts.append(
                "Relevant evidence:\n"
                + "\n".join(f"- {item}" for item in evidence[:2])
            )

        constraints = structured_answer.get("constraints") or []
        if constraints:
            parts.append(
                "Current constraints:\n"
                + "\n".join(f"- {item}" for item in constraints[:3])
            )

        recommendation = structured_answer.get("recommendation")
        if recommendation:
            parts.append(f"Recommendation: {recommendation}")

        return "\n\n".join(parts)

    def _blocking_constraint_texts(
        self,
        constrained_judgments: list[MemoryJudgment],
        constraints: list[ActiveConstraint],
    ) -> list[str]:
        constraints_by_id = {
            constraint.constraint_id: constraint
            for constraint in constraints
            if constraint.status == "active"
        }
        texts: list[str] = []
        for judgment in constrained_judgments:
            for constraint_id in judgment.blocked_by:
                constraint = constraints_by_id.get(constraint_id)
                if constraint is not None:
                    self._append_unique(
                        texts,
                        _english_user_visible_text(
                            constraint.content,
                            "A retrieved active constraint applies to this query.",
                        ),
                    )

        if not texts:
            for constraint in constraints:
                if constraint.status == "active":
                    self._append_unique(
                        texts,
                        _english_user_visible_text(
                            constraint.content,
                            "A retrieved active constraint applies to this query.",
                        ),
                    )

        if not texts:
            return ["A current constraint exists, but no concrete constraint text was retrieved."]
        return texts

    def _memory_texts(
        self,
        judgments: list[MemoryJudgment],
        memory_by_id: dict[str, OrdinaryMemory],
    ) -> list[str]:
        texts: list[str] = []
        for judgment in judgments:
            memory = memory_by_id.get(judgment.memory_id)
            if memory is not None and memory.content.strip():
                content = _english_user_visible_text(
                    memory.content,
                    "A retrieved memory applies to this query.",
                )
                if content:
                    self._append_unique(texts, content)
        return texts

    def _append_unique(self, values: list[str], value: str) -> None:
        stripped = value.strip()
        if stripped and stripped not in values:
            values.append(stripped)


class LLMConstrainedAnswerGenerator:
    """LLM answer generator grounded in CADMR structured reasoning."""

    ANSWER_CONTENT_VISIBLE_STATUSES = {"USABLE"}
    ANSWER_CONTROL_STATUSES = {"CONSTRAINED", "STALE", "SUSPENDED"}
    ANSWER_HIDDEN_STATUSES = {"NOISE"}

    def __init__(
        self,
        llm_client,
        fallback_generator: ConstrainedAnswerGenerator | None = None,
    ):
        self.llm_client = llm_client
        self.fallback_generator = fallback_generator or ConstrainedAnswerGenerator()

    def generate(
        self,
        query_info: QueryInfo | None,
        judgments: list[MemoryJudgment],
        constraints: list[ActiveConstraint],
        memories: list[OrdinaryMemory] | None = None,
        revision_context: dict | None = None,
    ) -> str:
        if query_info is None:
            return ""

        prompt = self._build_prompt(
            query_info=query_info,
            judgments=judgments,
            constraints=constraints,
            memories=memories or [],
            revision_context=revision_context,
        )
        try:
            raw_result = self.llm_client.complete_json(prompt)
        except Exception:
            return self._generate_english_fallback(query_info, judgments, constraints, memories or [])

        answer = raw_result.get("answer") if isinstance(raw_result, dict) else None
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
        return self._generate_english_fallback(query_info, judgments, constraints, memories or [])

    def _build_prompt(
        self,
        query_info: QueryInfo,
        judgments: list[MemoryJudgment],
        constraints: list[ActiveConstraint],
        memories: list[OrdinaryMemory],
        revision_context: dict | None = None,
    ) -> str:
        payload = self._build_answer_payload(
            query_info,
            judgments,
            constraints,
            memories,
            revision_context=revision_context,
        )
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        return f"""
You are the CADMR final answer generator.

Generate a concise, user-facing answer in English.
Use English as the default output language; do not draft in another language and then translate.
Do not mention CADMR, memory stores, schemas, judgments, memory IDs, or internal status labels.
Answer the user's actual question directly.
Do not invent facts beyond the supplied structured reasoning.
Only answer_visible_memories may be used as positive evidence or as a basis for
action recommendations. Memory content from CONSTRAINED, STALE, SUSPENDED, and
NOISE judgments is intentionally hidden from this prompt. Do not infer facts
from hidden memories.

Use the structured reasoning as follows:
- If answer_control_judgments contains STALE, clearly state that the old premise should not be assumed as current. Use only answer_visible_memories as newer/current evidence.
- If answer_control_judgments contains CONSTRAINED, treat the old memory as blocked for direct action. Follow answer_visible_constraints, but do not use hidden constrained-memory content as a recommendation source.
- If a memory is USABLE, use it as current grounding.
- If filtered_out_judgment_counts reports NOISE memories, treat them as unavailable and irrelevant.
- If filtered_out_judgment_counts reports SUSPENDED memories, do not rely on them; mention uncertainty only if needed.
- If evidence is insufficient, say what can and cannot be inferred, but still answer cautiously when possible.
- For a false-premise question, explicitly resist the premise before giving advice.
- For an implicit task, give a concrete action, plan, or recommendation that fits
  the current state, but base concrete actions only on the user query,
  answer_visible_memories, and answer_visible_constraints. Avoid using blocked,
  stale, suspended, or noise memory content as action material.
- If revision_context is present, revise the prior answer specifically to fix
  verifier violations. Remove any action that triggered stale_memory_use or
  constraint_violation, and produce a corrected answer that should pass verification.

Return strict JSON:
{{
  "answer": "English answer for the user"
}}

Structured reasoning:
{payload_json}
""".strip()

    def _build_answer_payload(
        self,
        query_info: QueryInfo,
        judgments: list[MemoryJudgment],
        constraints: list[ActiveConstraint],
        memories: list[OrdinaryMemory],
        revision_context: dict | None = None,
    ) -> dict:
        memory_by_id = {memory.memory_id: memory for memory in memories}
        constraint_by_id = {constraint.constraint_id: constraint for constraint in constraints}
        status_by_memory_id = {
            judgment.memory_id: judgment.usage_status
            for judgment in judgments
        }

        visible_memory_ids: set[str] = set()
        visible_constraint_ids: set[str] = set()
        hidden_counts = {
            "CONSTRAINED": 0,
            "STALE": 0,
            "SUSPENDED": 0,
            "NOISE": 0,
        }
        control_judgments: list[dict] = []

        for judgment in judgments:
            if judgment.usage_status in self.ANSWER_CONTENT_VISIBLE_STATUSES:
                visible_memory_ids.add(judgment.memory_id)
                for replacement_id in judgment.replaced_by:
                    if (
                        replacement_id in memory_by_id
                        and status_by_memory_id.get(replacement_id) in self.ANSWER_CONTENT_VISIBLE_STATUSES
                    ):
                        visible_memory_ids.add(replacement_id)
                    if replacement_id in constraint_by_id:
                        visible_constraint_ids.add(replacement_id)
            elif judgment.usage_status in self.ANSWER_CONTROL_STATUSES:
                hidden_counts[judgment.usage_status] += 1
                visible_constraint_ids.update(judgment.blocked_by)
                for replacement_id in judgment.replaced_by:
                    if (
                        replacement_id in memory_by_id
                        and status_by_memory_id.get(replacement_id) in self.ANSWER_CONTENT_VISIBLE_STATUSES
                    ):
                        visible_memory_ids.add(replacement_id)
                    if replacement_id in constraint_by_id:
                        visible_constraint_ids.add(replacement_id)
                control_judgments.append(self._sanitize_control_judgment(judgment))
            elif judgment.usage_status in hidden_counts:
                hidden_counts[judgment.usage_status] += 1

        visible_memories = [
            memory
            for memory in memories
            if memory.memory_id in visible_memory_ids
        ]
        visible_constraints = [
            constraint
            for constraint in constraints
            if constraint.constraint_id in visible_constraint_ids
        ]
        visible_judgments = [
            judgment
            for judgment in judgments
            if judgment.usage_status in self.ANSWER_CONTENT_VISIBLE_STATUSES
            or judgment.memory_id in visible_memory_ids
        ]

        payload = {
            "query_info": query_info.model_dump(),
            "answer_visible_memories": [memory.model_dump() for memory in visible_memories],
            "answer_visible_constraints": [
                constraint.model_dump() for constraint in visible_constraints
            ],
            "answer_visible_judgments": [
                judgment.model_dump() for judgment in visible_judgments
            ],
            "answer_control_judgments": control_judgments,
            "filtered_out_judgment_counts": hidden_counts,
        }
        if revision_context:
            payload["revision_context"] = self._sanitize_revision_context(revision_context)
        return payload

    def _sanitize_revision_context(self, revision_context: dict) -> dict:
        violation_types = {"stale_memory_use", "constraint_violation"}
        violations = revision_context.get("violations", [])
        if not isinstance(violations, list):
            violations = []
        sanitized_violations = []
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            violation_type = str(violation.get("type", ""))
            if violation_type not in violation_types:
                continue
            sanitized_violations.append(
                {
                    "type": violation_type,
                    "evidence": str(violation.get("evidence", ""))[:500],
                    "related_id": str(violation.get("related_id", "")),
                }
            )
        return {
            "prior_answer": str(revision_context.get("prior_answer", ""))[:2000],
            "reason": str(revision_context.get("reason", ""))[:1000],
            "violations": sanitized_violations,
            "instruction": (
                "Rewrite the answer to remove stale-memory use and constraint violations. "
                "Use only answer_visible_memories, answer_visible_constraints, and the current query for concrete advice."
            ),
        }

    def _sanitize_control_judgment(self, judgment: MemoryJudgment) -> dict:
        return {
            "memory_id": judgment.memory_id,
            "usage_status": judgment.usage_status,
            "truth_status": judgment.truth_status,
            "actionability": judgment.actionability,
            "blocked_by": judgment.blocked_by,
            "replaced_by": judgment.replaced_by,
            "allowed_use": judgment.allowed_use,
            "forbidden_use": judgment.forbidden_use,
            "reason": judgment.reason,
        }

    def _generate_english_fallback(
        self,
        query_info: QueryInfo,
        judgments: list[MemoryJudgment],
        constraints: list[ActiveConstraint],
        memories: list[OrdinaryMemory],
    ) -> str:
        if not judgments:
            return (
                "I do not have enough relevant long-term memory or active constraints "
                "to ground this answer. I should answer from the current question only "
                "and avoid assuming extra background."
            )

        memory_by_id = {memory.memory_id: memory for memory in memories}
        stale = [judgment for judgment in judgments if judgment.usage_status == "STALE"]
        constrained = [judgment for judgment in judgments if judgment.usage_status == "CONSTRAINED"]
        usable = [judgment for judgment in judgments if judgment.usage_status == "USABLE"]

        if stale:
            current_evidence = self._memory_lines(usable, memory_by_id)
            answer = "I would not assume the older premise is still current."
            if current_evidence:
                answer += " The usable current evidence is: " + "; ".join(current_evidence[:3]) + "."
            answer += " Any recommendation should be based on the newer/current state rather than the outdated memory."
            return answer

        if constrained:
            visible_constraints = self._constraints_for_answer(constrained, constraints)
            constraint_lines = [
                _english_user_visible_text(
                    constraint.content,
                    "A retrieved active constraint applies to this query",
                )
                for constraint in visible_constraints
                if constraint.status == "active" and constraint.content.strip()
            ]
            answer = "The older memory can only be used as background, not as direct action guidance."
            if constraint_lines:
                answer += " Current constraint: " + "; ".join(constraint_lines[:3]) + "."
            answer += " The safest answer is to adapt the plan around that constraint."
            return answer

        if usable:
            usable_lines = self._memory_lines(usable, memory_by_id)
            if usable_lines:
                return "The recalled memory appears usable for this question: " + "; ".join(usable_lines[:3]) + "."
            return "The recalled memory appears usable for this question."

        return "The recalled memories do not provide a reliable basis for answering this question."

    def _constraints_for_answer(
        self,
        judgments: list[MemoryJudgment],
        constraints: list[ActiveConstraint],
    ) -> list[ActiveConstraint]:
        constraint_by_id = {constraint.constraint_id: constraint for constraint in constraints}
        visible_ids: set[str] = set()
        for judgment in judgments:
            visible_ids.update(judgment.blocked_by)
            for replacement_id in judgment.replaced_by:
                if replacement_id in constraint_by_id:
                    visible_ids.add(replacement_id)
        return [
            constraint
            for constraint in constraints
            if constraint.constraint_id in visible_ids
        ]

    def _memory_lines(
        self,
        judgments: list[MemoryJudgment],
        memory_by_id: dict[str, OrdinaryMemory],
    ) -> list[str]:
        lines = []
        for judgment in judgments:
            memory = memory_by_id.get(judgment.memory_id)
            if memory is not None and memory.content.strip():
                content = _english_user_visible_text(
                    memory.content,
                    "a retrieved memory applies to this query",
                )
                if content:
                    lines.append(content)
        return lines
