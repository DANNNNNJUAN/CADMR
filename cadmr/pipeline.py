"""Top-level CADMR pipeline orchestration."""

from datetime import UTC, datetime
import uuid

from cadmr.answer_generator import ConstrainedAnswerGenerator
from cadmr.extractor import MemorySignalExtractor, RuleBasedMemorySignalExtractor
from cadmr.goal_reconstructor import GoalReconstructor
from cadmr.retrieval import MemoryRetriever
from cadmr.resolver import ReferentTopicResolver
from cadmr.schemas import (
    ActiveConstraint,
    MemorySignal,
    OrdinaryMemory,
    PipelineResult,
    QueryInfo,
    RawInteraction,
)
from cadmr.stores import ActiveConstraintStore, OrdinaryMemoryStore, RawInteractionLog
from cadmr.usability_judge import MemoryUsabilityJudge
from cadmr.verifier import AnswerVerifier
from cadmr.write_gate import MemoryWriteGate


LOCATION_KEYWORDS = [
    "北京",
    "上海",
    "成都",
    "杭州",
    "深圳",
    "广州",
    "南京",
    "武汉",
    "西安",
    "美国",
    "中国",
    "纽约",
    "洛杉矶",
]
LOCATION_REPLACEMENT_KEYWORDS = [
    "现在",
    "已经",
    "搬到",
    "搬去",
    "住在",
    "在上海",
    "在北京",
    "来到",
    "出差到",
]
PROJECT_KEYWORDS = ["demo", "汇报", "PPT", "项目", "论文", "研究"]
PROJECT_REPLACEMENT_KEYWORDS = ["改成", "换成", "现在", "这次", "新目标", "不再"]


class CADMRPipeline:
    """Minimal write-side pipeline for CADMR."""

    def __init__(
        self,
        raw_log: RawInteractionLog,
        ordinary_store: OrdinaryMemoryStore,
        constraint_store: ActiveConstraintStore,
        extractor: MemorySignalExtractor | None = None,
        write_gate: MemoryWriteGate | None = None,
        retriever: MemoryRetriever | None = None,
        usability_judge: MemoryUsabilityJudge | None = None,
        answer_generator: ConstrainedAnswerGenerator | None = None,
        resolver: ReferentTopicResolver | None = None,
        goal_reconstructor: GoalReconstructor | None = None,
        answer_verifier: AnswerVerifier | None = None,
    ):
        self.raw_log = raw_log
        self.ordinary_store = ordinary_store
        self.constraint_store = constraint_store
        self.extractor = extractor or RuleBasedMemorySignalExtractor()
        self.write_gate = write_gate or MemoryWriteGate()
        self.retriever = retriever or MemoryRetriever(ordinary_store, constraint_store)
        self.usability_judge = usability_judge or MemoryUsabilityJudge()
        self.answer_generator = answer_generator or ConstrainedAnswerGenerator()
        self.resolver = resolver or ReferentTopicResolver()
        self.goal_reconstructor = goal_reconstructor or GoalReconstructor()
        self.answer_verifier = answer_verifier or AnswerVerifier()

    def run(self, user_input: str) -> PipelineResult:
        interaction = self._make_raw_interaction(user_input)
        self.raw_log.append(interaction)

        signals = self.extractor.extract(user_input)
        write_decisions = self.write_gate.decide_many(signals)

        for signal, decision in zip(signals, write_decisions, strict=True):
            if decision.decision == "WRITE_TO_ORDINARY_MEMORY":
                memory = self._signal_to_ordinary_memory(signal, interaction)
                self._mark_superseded_memories(memory)
                self.ordinary_store.add(memory)
            elif decision.decision == "WRITE_TO_ACTIVE_CONSTRAINT":
                self.constraint_store.add(
                    self._signal_to_active_constraint(signal, interaction)
                )

        recent_interactions = self.raw_log.list_all()
        all_memories = self.ordinary_store.list_all()
        all_constraints = self.constraint_store.list_all()
        query_info = self._build_query_info(
            user_input,
            signals,
            recent_interactions,
            all_memories,
            all_constraints,
        )
        if query_info is None:
            judgments = []
            answer = None
            goal_plan = None
            verify_result = None
        else:
            memories, constraints = self.retriever.retrieve(query_info)
            judgments = self.usability_judge.judge(query_info, memories, constraints)
            goal_plan = self.goal_reconstructor.reconstruct(query_info, memories, constraints)
            answer = self.answer_generator.generate(query_info, judgments, constraints)
            answer = self._append_goal_plan_summary(answer, goal_plan)
            verify_result = self.answer_verifier.verify(answer, judgments, constraints, goal_plan)

        return PipelineResult(
            user_input=user_input,
            signals=signals,
            write_decisions=write_decisions,
            query_info=query_info,
            judgments=judgments,
            answer=answer,
            goal_plan=goal_plan,
            verify_result=verify_result,
        )

    def _make_raw_interaction(self, user_input: str) -> RawInteraction:
        return RawInteraction(
            interaction_id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC).isoformat(),
            speaker="user",
            text=user_input,
        )

    def _signal_to_ordinary_memory(
        self,
        signal: MemorySignal,
        interaction: RawInteraction,
    ) -> OrdinaryMemory:
        return OrdinaryMemory(
            memory_id=str(uuid.uuid4()),
            content=signal.content,
            subject=self._resolve_signal_subject(signal.content, signal.subject),
            scope=signal.scope,
            stability="long_term",
            status="active",
            confidence=signal.confidence,
            evidence_ids=[interaction.interaction_id],
            created_at=interaction.timestamp,
            updated_at=interaction.timestamp,
        )

    def _signal_to_active_constraint(
        self,
        signal: MemorySignal,
        interaction: RawInteraction,
    ) -> ActiveConstraint:
        return ActiveConstraint(
            constraint_id=str(uuid.uuid4()),
            content=signal.content,
            subject=self._resolve_signal_subject(signal.content, signal.subject),
            scope=signal.scope,
            priority="high",
            strength="hard",
            valid_time={},
            status="active",
            source="user_input",
            confidence=signal.confidence,
            evidence_ids=[interaction.interaction_id],
            created_at=interaction.timestamp,
            updated_at=interaction.timestamp,
        )

    def _build_query_info(
        self,
        user_input: str,
        signals: list[MemorySignal],
        recent_interactions: list[RawInteraction],
        memories: list[OrdinaryMemory],
        constraints: list[ActiveConstraint],
    ) -> QueryInfo | None:
        if not any(signal.signal_type == "query_intent" for signal in signals):
            return None

        resolution = self.resolver.resolve(
            user_input,
            recent_interactions,
            memories,
            constraints,
        )
        query_scope = self._merge_scopes(signals)
        if resolution.get("topic_status") == "reentered":
            for scope in ["work", "project"]:
                if scope not in query_scope:
                    query_scope.append(scope)

        return QueryInfo(
            query=user_input,
            query_intent="unknown",
            query_scope=query_scope,
            resolved_subject=resolution.get("resolved_subject", "user"),
            requires_action=True,
            requires_plan=False,
            possible_old_premises=[],
        )

    def _merge_scopes(self, signals: list[MemorySignal]) -> list[str]:
        scopes: list[str] = []
        for signal in signals:
            for scope in signal.scope:
                if scope not in scopes:
                    scopes.append(scope)
        return scopes or ["general"]

    def _mark_superseded_memories(self, new_memory: OrdinaryMemory) -> None:
        for old_memory in self.ordinary_store.get_active():
            if old_memory.subject != new_memory.subject:
                continue
            if not set(old_memory.scope).intersection(new_memory.scope):
                continue
            if self._looks_like_replacement(old_memory.content, new_memory.content):
                self.ordinary_store.mark_stale(old_memory.memory_id)

    def _looks_like_replacement(self, old_content: str, new_content: str) -> bool:
        if self._looks_like_location_replacement(old_content, new_content):
            return True
        if self._looks_like_project_replacement(old_content, new_content):
            return True
        return False

    def _looks_like_location_replacement(
        self,
        old_content: str,
        new_content: str,
    ) -> bool:
        old_locations = self._matched_keywords(old_content, LOCATION_KEYWORDS)
        new_locations = self._matched_keywords(new_content, LOCATION_KEYWORDS)
        if not old_locations or not new_locations:
            return False
        if not set(old_locations).symmetric_difference(new_locations):
            return False
        return any(keyword in new_content for keyword in LOCATION_REPLACEMENT_KEYWORDS)

    def _looks_like_project_replacement(
        self,
        old_content: str,
        new_content: str,
    ) -> bool:
        has_old_project = any(keyword in old_content for keyword in PROJECT_KEYWORDS)
        has_new_project = any(keyword in new_content for keyword in PROJECT_KEYWORDS)
        has_replacement = any(keyword in new_content for keyword in PROJECT_REPLACEMENT_KEYWORDS)
        return has_old_project and has_new_project and has_replacement

    def _matched_keywords(self, text: str, keywords: list[str]) -> list[str]:
        return [keyword for keyword in keywords if keyword in text]

    def _resolve_signal_subject(self, content: str, fallback: str) -> str:
        if any(keyword in content for keyword in ["猫", "宠物", "兽医"]):
            return "cat"
        if any(keyword in content for keyword in ["我爸", "父亲"]):
            return "father"
        if any(keyword in content for keyword in ["我妈", "母亲"]):
            return "mother"
        return fallback

    def _append_goal_plan_summary(self, answer: str, goal_plan: dict | None) -> str:
        if not goal_plan or not goal_plan.get("needs_goal_reconstruction"):
            return answer
        components = goal_plan.get("required_plan_components", [])
        forbidden_actions = goal_plan.get("forbidden_actions", [])
        parts = [answer]
        if components and not all(component in answer for component in components):
            parts.append("计划组件：" + "、".join(components))
        if forbidden_actions and not all(action in answer for action in forbidden_actions):
            parts.append("禁止动作：" + "、".join(forbidden_actions))
        return "\n\n".join(part for part in parts if part)
