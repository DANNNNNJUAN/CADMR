"""Rule-based constrained answer generation."""

from cadmr.schemas import ActiveConstraint, MemoryJudgment, QueryInfo


FOOD_KEYWORDS = ["火锅", "吃", "辣", "川菜", "餐厅"]


def _contains_food_keyword(query: str) -> bool:
    return any(keyword in query for keyword in FOOD_KEYWORDS)


class ConstrainedAnswerGenerator:
    """Generates answers using judged memories and active constraints."""

    def generate(
        self,
        query_info: QueryInfo | None,
        judgments: list[MemoryJudgment],
        constraints: list[ActiveConstraint],
    ) -> str:
        if query_info is None:
            return ""

        if "演示" in query_info.query or "方案" in query_info.query or "安排" in query_info.query:
            return (
                "建议按稳妥方案组织：时间分配、脱敏案例、非技术化说明、风险控制、备用方案。"
                "不要展示真实用户数据，也不要暴露敏感 memory trace。"
            )

        if not judgments:
            return (
                "当前没有找到可用的历史记忆或限制条件。可以基于当前问题直接回答，"
                "但不应假设额外长期记忆。"
            )

        constrained_judgments = [
            judgment
            for judgment in judgments
            if judgment.usage_status == "CONSTRAINED"
        ]
        if constrained_judgments:
            if _contains_food_keyword(query_info.query):
                return self._generate_constrained_food_answer(constraints)
            return self._generate_constrained_answer(constraints)

        stale_judgments = [
            judgment for judgment in judgments if judgment.usage_status == "STALE"
        ]
        if stale_judgments:
            usable_lines = [
                f"- {judgment.memory_id}"
                for judgment in judgments
                if judgment.usage_status == "USABLE"
            ]
            answer = "召回到的部分历史记忆已经被更新状态替代，因此不能作为当前回答依据。请优先使用最新记忆。"
            if usable_lines:
                answer += "\n\n可用记忆：\n" + "\n".join(usable_lines)
            return answer

        if all(judgment.usage_status == "NOISE" for judgment in judgments):
            return "召回到的历史记忆与当前问题主体或范围不匹配，因此不应作为当前回答依据。"

        if all(judgment.usage_status == "USABLE" for judgment in judgments):
            usable_lines = [
                f"- {judgment.memory_id}"
                for judgment in judgments
                if judgment.usage_status == "USABLE"
            ]
            if usable_lines:
                return "当前召回的记忆没有被约束阻断，可以作为回答依据。\n\n可用记忆：\n" + "\n".join(usable_lines)
            return "当前召回的记忆没有被约束阻断，可以作为回答依据。"

        return "当前召回的记忆需要谨慎使用，应优先遵守当前限制条件。"

    def _generate_constrained_food_answer(
        self,
        constraints: list[ActiveConstraint],
    ) -> str:
        constraint_lines = self._format_constraints(constraints)
        return (
            "你确实有相关饮食偏好，但当前存在饮食或健康限制，所以不适合直接按这个偏好执行。\n\n"
            f"当前限制：\n{constraint_lines}\n\n"
            "因此，今晚不建议选择刺激性饮食。可以考虑清淡替代方案，例如清汤锅、"
            "番茄锅、菌汤锅，或少油少辣的菜品。"
        )

    def _generate_constrained_answer(
        self,
        constraints: list[ActiveConstraint],
    ) -> str:
        constraint_lines = self._format_constraints(constraints)
        return (
            "你之前的相关偏好或记忆仍然可以作为背景参考，但当前存在更高优先级的限制条件，"
            "因此不能直接按旧记忆执行。\n\n"
            f"当前限制：\n{constraint_lines}\n\n"
            "建议：\n在满足当前限制的前提下调整方案，避免使用被限制的旧偏好作为直接行动依据。"
        )

    def _format_constraints(self, constraints: list[ActiveConstraint]) -> str:
        if not constraints:
            return "- 当前存在限制条件，但未召回具体限制内容。"
        return "\n".join(f"- {constraint.content}" for constraint in constraints)
