"""Referent and topic resolver interface placeholder."""

from cadmr.schemas import ActiveConstraint, OrdinaryMemory, RawInteraction


class ReferentTopicResolver:
    """No-op resolver used unless an LLM/pluggable resolver is injected."""

    def resolve(
        self,
        query: str,
        recent_interactions: list[RawInteraction],
        memories: list[OrdinaryMemory],
        constraints: list[ActiveConstraint],
    ) -> dict:
        return {
            "resolved_subject": "user",
            "current_topic": "general",
            "topic_status": "active",
            "referents": {},
            "discarded_contexts": [],
            "reason": "No referent/topic resolver is configured.",
        }
