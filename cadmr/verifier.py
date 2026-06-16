"""Rule-based answer verification."""

from cadmr.schemas import ActiveConstraint, MemoryJudgment


SPICY_ACTION_KEYWORDS = ["重辣", "麻辣", "火锅"]
SAFE_NEGATION_KEYWORDS = ["不建议", "不适合", "避免", "不能"]


class AnswerVerifier:
    """Checks generated answers against judgments, constraints, and goal plans."""

    def verify(
        self,
        answer: str,
        judgments: list[MemoryJudgment],
        constraints: list[ActiveConstraint],
        goal_plan: dict | None = None,
    ) -> dict:
        violations: list[str] = []
        missing_components: list[str] = []

        if any(judgment.usage_status == "CONSTRAINED" for judgment in judgments):
            constraint_text = "\n".join(constraint.content for constraint in constraints)
            if "不能吃辣" in constraint_text and any(
                keyword in answer for keyword in SPICY_ACTION_KEYWORDS
            ) and not any(keyword in answer for keyword in SAFE_NEGATION_KEYWORDS):
                violations.append("Answer appears to recommend spicy food despite an active constraint.")

        if goal_plan:
            for component in goal_plan.get("required_plan_components", []):
                if component not in answer:
                    missing_components.append(component)

        return {
            "pass": not violations and not missing_components,
            "violations": violations,
            "missing_components": missing_components,
            "needs_revision": bool(violations or missing_components),
        }
