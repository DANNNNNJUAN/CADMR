"""Rule-based goal reconstruction."""

from cadmr.schemas import ActiveConstraint, OrdinaryMemory, QueryInfo


GOAL_KEYWORDS = ["方案", "计划", "demo", "演示", "汇报", "PPT", "流程", "安排"]
EXTERNAL_PRIVACY_KEYWORDS = ["不能展示真实数据", "隐私", "脱敏", "敏感", "合作方", "管理老师", "外部"]


class GoalReconstructor:
    """Reconstructs current goals under active constraints."""

    def reconstruct(
        self,
        query_info: QueryInfo,
        memories: list[OrdinaryMemory],
        constraints: list[ActiveConstraint],
    ) -> dict:
        if not any(keyword in query_info.query for keyword in GOAL_KEYWORDS):
            return {
                "needs_goal_reconstruction": False,
                "current_goal": "",
                "retained_parts": [],
                "changed_parts": [],
                "required_plan_components": [],
                "forbidden_actions": [],
                "reason": "Query does not request a plan or demo arrangement.",
            }

        constraint_text = "\n".join(constraint.content for constraint in constraints)
        if any(keyword in constraint_text for keyword in EXTERNAL_PRIVACY_KEYWORDS):
            return {
                "needs_goal_reconstruction": True,
                "current_goal": "面向外部受众的稳定、脱敏、非技术化演示",
                "retained_parts": [memory.content for memory in memories],
                "changed_parts": ["受众与数据展示边界已经变化"],
                "required_plan_components": ["时间分配", "脱敏案例", "非技术化说明", "风险控制", "备用方案"],
                "forbidden_actions": ["展示真实用户数据", "暴露敏感 memory trace", "只按内部技术 demo 方式演示"],
                "reason": "External/privacy constraints require reconstructing the demo goal.",
            }

        return {
            "needs_goal_reconstruction": True,
            "current_goal": "制定当前问题的可执行计划",
            "retained_parts": [memory.content for memory in memories],
            "changed_parts": [],
            "required_plan_components": ["步骤", "注意事项", "风险点"],
            "forbidden_actions": [],
            "reason": "Query asks for a plan or arrangement.",
        }
