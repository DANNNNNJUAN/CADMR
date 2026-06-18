"""Memory signal extraction interfaces and lightweight implementations."""

import uuid

from cadmr.scope import canonicalize_scopes
from cadmr.schemas import MemorySignal


ALLOWED_SIGNAL_TYPES = {
    "ordinary_memory",
    "active_constraint",
    "query_intent",
    "hypothetical",
    "question_premise",
    "uncertain_intention",
}

DEFAULT_CONFIDENCE = {
    "ordinary_memory": 0.8,
    "active_constraint": 0.9,
    "query_intent": 0.8,
    "hypothetical": 0.75,
    "question_premise": 0.75,
    "uncertain_intention": 0.65,
}


def infer_scope(text: str) -> list[str]:
    """Compatibility hook; scope inference is owned by LLM extraction."""
    return ["general"]


class MemorySignalExtractor:
    """Base interface for extracting memory-related signals from text."""

    def extract(self, text: str) -> list[MemorySignal]:
        raise NotImplementedError


class RuleBasedMemorySignalExtractor(MemorySignalExtractor):
    """No-op fallback extractor.

    CADMR's evaluation path should use LLMMemorySignalExtractor. This class is
    kept only as a safe import/default that does not inject keyword rules.
    """

    def extract(self, text: str) -> list[MemorySignal]:
        return []


class LLMClient:
    """Abstract client for structured LLM JSON completion."""

    def complete_json(self, prompt: str) -> dict:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic LLM client used by tests and local development."""

    def __init__(self, response: dict):
        self.response = response

    def complete_json(self, prompt: str) -> dict:
        return self.response


class RuleValidator:
    """Validate and repair LLM-extracted memory signals using safety rules."""

    def validate(self, raw_items: list[dict], original_text: str) -> list[MemorySignal]:
        signals: list[MemorySignal] = []
        seen: set[tuple[str, str]] = set()

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            signal_type = item.get("signal_type")
            if signal_type not in ALLOWED_SIGNAL_TYPES:
                continue

            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            content = content.strip()

            subject = item.get("subject")
            if not isinstance(subject, str) or not subject.strip():
                subject = "user"

            scope = item.get("scope")
            if not self._is_string_list(scope):
                scope = ["general"]
            scope = canonicalize_scopes(scope or ["general"])

            confidence = item.get("confidence", DEFAULT_CONFIDENCE[signal_type])
            if not isinstance(confidence, int | float):
                confidence = DEFAULT_CONFIDENCE[signal_type]
            confidence = max(0.0, min(1.0, float(confidence)))

            evidence_text = item.get("evidence_text")
            if not isinstance(evidence_text, str) or not evidence_text.strip():
                evidence_text = original_text

            dedupe_key = (signal_type, content)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            signals.append(
                MemorySignal(
                    signal_id=str(uuid.uuid4()),
                    signal_type=signal_type,
                    content=content,
                    subject=subject.strip(),
                    scope=scope,
                    confidence=confidence,
                    evidence_text=evidence_text,
                )
            )

        return signals

    def _is_string_list(self, value) -> bool:
        return isinstance(value, list) and all(isinstance(item, str) for item in value)

class LLMMemorySignalExtractor(MemorySignalExtractor):
    """LLM extractor with rule validation and pollution prevention."""

    def __init__(self, llm_client: LLMClient, validator: RuleValidator | None = None):
        self.llm_client = llm_client
        self.validator = validator or RuleValidator()

    def extract(self, text: str) -> list[MemorySignal]:
        prompt = self._build_prompt(text)
        try:
            raw_output = self.llm_client.complete_json(prompt)
        except Exception:
            return []
        raw_items = raw_output.get("signals", [])
        if not isinstance(raw_items, list):
            raw_items = []
        return self.validator.validate(raw_items, original_text=text)

    def _build_prompt(self, text: str) -> str:
        return f"""
Extract CADMR memory signals from the user input and return strict JSON only.

Allowed signal_type values:
ordinary_memory, active_constraint, query_intent, hypothetical,
question_premise, uncertain_intention.

Definitions:
- ordinary_memory: stable facts, preferences, habits, or long-term background.
- active_constraint: an explicit current condition that limits answers, plans, or action advice.
- query_intent: the user's current question, request, or task intent.
- hypothetical: a hypothetical or counterfactual premise.
- question_premise: an old premise embedded in a question that should be verified, not written as memory.
- uncertain_intention: an uncertain, tentative, or exploratory intention.

Rules:
- Do not turn hypotheses into facts.
- Do not turn old premises inside a question into active memory.
- Do not turn uncertain intentions into stable preferences.
- query_intent is for answering only and is not long-term memory.
- A request such as "help me arrange", "can I", "should I", "I am considering", or "I want to know" is query_intent, not active_constraint.
- active_constraint requires an explicit limiting condition, requirement, prohibition, deadline, safety/privacy/health/resource restriction, or other current constraint.
- Do not create active_constraint merely because the user asks for a recommendation, plan, or possibility.
- If one input contains both stable background and a current constraint, split them into separate signals.
- One input may produce multiple signals.

Return shape:
{{
  "signals": [
    {{
      "signal_type": "ordinary_memory | active_constraint | query_intent | hypothetical | question_premise | uncertain_intention",
      "content": "...",
      "subject": "user",
      "scope": ["..."],
      "confidence": 0.0,
      "evidence_text": "..."
    }}
  ]
}}

User input: {text}
""".strip()
