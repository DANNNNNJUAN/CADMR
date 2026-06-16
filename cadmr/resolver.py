"""Rule-based referent and topic resolution."""

from cadmr.schemas import ActiveConstraint, OrdinaryMemory, RawInteraction


CAT_CONTEXT_KEYWORDS = ["猫", "宠物", "兽医", "医院", "肾功能", "饮食小心"]
CAT_QUERY_KEYWORDS = ["它", "猫", "宠物"]
FATHER_KEYWORDS = ["我爸", "父亲"]
MOTHER_KEYWORDS = ["我妈", "母亲"]
TOPIC_RETURN_KEYWORDS = ["刚才那个", "之前那个", "前面那个", "继续刚才", "回到刚才", "那个流程", "那个方案"]
DEMO_CONTEXT_KEYWORDS = ["demo", "汇报", "PPT", "长期记忆系统", "流程", "方案"]


class ReferentTopicResolver:
    """Resolves the current subject and topic from recent context."""

    def resolve(
        self,
        query: str,
        recent_interactions: list[RawInteraction],
        memories: list[OrdinaryMemory],
        constraints: list[ActiveConstraint],
    ) -> dict:
        context_text = self._join_context(recent_interactions, memories, constraints)

        if any(keyword in query for keyword in FATHER_KEYWORDS):
            return self._result("father", "family", "active", {}, [], "Father referent detected.")
        if any(keyword in query for keyword in MOTHER_KEYWORDS):
            return self._result("mother", "family", "active", {}, [], "Mother referent detected.")

        if any(keyword in query for keyword in CAT_QUERY_KEYWORDS) and any(
            keyword in context_text for keyword in CAT_CONTEXT_KEYWORDS
        ):
            return self._result(
                "cat",
                "cat_diet",
                "active",
                {"它": "cat"},
                ["user_health_context"],
                "Pet/cat context makes the pronoun refer to the cat.",
            )

        if any(keyword in query for keyword in TOPIC_RETURN_KEYWORDS) and any(
            keyword in context_text for keyword in DEMO_CONTEXT_KEYWORDS
        ):
            topic = "long_term_memory_demo" if "长期记忆系统" in context_text else "previous_task"
            return self._result(
                "user",
                topic,
                "reentered",
                {},
                [],
                "Query returns to a recent work/demo topic.",
            )

        return self._result("user", "general", "active", {}, [], "Default user subject.")

    def _join_context(
        self,
        recent_interactions: list[RawInteraction],
        memories: list[OrdinaryMemory],
        constraints: list[ActiveConstraint],
    ) -> str:
        parts = [interaction.text for interaction in recent_interactions]
        parts.extend(memory.content for memory in memories)
        parts.extend(constraint.content for constraint in constraints)
        return "\n".join(parts)

    def _result(
        self,
        resolved_subject: str,
        current_topic: str,
        topic_status: str,
        referents: dict,
        discarded_contexts: list[str],
        reason: str,
    ) -> dict:
        return {
            "resolved_subject": resolved_subject,
            "current_topic": current_topic,
            "topic_status": topic_status,
            "referents": referents,
            "discarded_contexts": discarded_contexts,
            "reason": reason,
        }
