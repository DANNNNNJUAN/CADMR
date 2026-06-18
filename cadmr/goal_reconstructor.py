"""Goal reconstruction interface placeholder."""

from cadmr.schemas import ActiveConstraint, OrdinaryMemory, QueryInfo


class GoalReconstructor:
    """No-op goal reconstructor used unless an LLM/pluggable reconstructor is injected."""

    def reconstruct(
        self,
        query_info: QueryInfo,
        memories: list[OrdinaryMemory],
        constraints: list[ActiveConstraint],
    ) -> dict:
        return {
            "needs_goal_reconstruction": False,
            "current_goal": "",
            "retained_parts": [],
            "changed_parts": [],
            "required_plan_components": [],
            "forbidden_actions": [],
            "reason": "No goal reconstructor is configured.",
        }
