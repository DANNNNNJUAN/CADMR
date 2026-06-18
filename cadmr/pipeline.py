"""Top-level CADMR pipeline orchestration."""

from datetime import datetime, timezone
import uuid

from cadmr.answer_generator import ConstrainedAnswerGenerator
from cadmr.extractor import MemorySignalExtractor, RuleBasedMemorySignalExtractor
from cadmr.goal_reconstructor import GoalReconstructor
from cadmr.retrieval import MemoryRetriever
from cadmr.resolver import ReferentTopicResolver
from cadmr.scope import canonicalize_scopes
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


ANSWER_TRIGGER_SIGNAL_TYPES = {
    "query_intent",
    "question_premise",
    "hypothetical",
    "uncertain_intention",
}


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
        retrieved_memories = []
        retrieved_constraints = []
        if query_info is None:
            judgments = []
            answer = None
            goal_plan = None
            verify_result = None
            judge_diagnostics = None
        else:
            retrieved_memories, retrieved_constraints = self.retriever.retrieve(query_info)
            judgments = self.usability_judge.judge(
                query_info,
                retrieved_memories,
                retrieved_constraints,
            )
            judge_diagnostics = self._get_judge_diagnostics()
            goal_plan = self.goal_reconstructor.reconstruct(
                query_info,
                retrieved_memories,
                retrieved_constraints,
            )
            answer = self.answer_generator.generate(
                query_info,
                judgments,
                retrieved_constraints,
                retrieved_memories,
            )
            pre_verify_structured_output = self._build_structured_output(
                user_input=user_input,
                signals=signals,
                write_decisions=write_decisions,
                query_info=query_info,
                retrieved_memories=retrieved_memories,
                retrieved_constraints=retrieved_constraints,
                judgments=judgments,
                judge_diagnostics=judge_diagnostics,
                goal_plan=goal_plan,
                verify_result=None,
                answer=answer,
            )
            verify_result = self.answer_verifier.verify(
                answer=answer,
                judgments=judgments,
                constraints=retrieved_constraints,
                goal_plan=goal_plan,
                structured_output=pre_verify_structured_output,
            )

        structured_output = self._build_structured_output(
            user_input=user_input,
            signals=signals,
            write_decisions=write_decisions,
            query_info=query_info,
            retrieved_memories=retrieved_memories,
            retrieved_constraints=retrieved_constraints,
            judgments=judgments,
            judge_diagnostics=judge_diagnostics,
            goal_plan=goal_plan,
            verify_result=verify_result,
            answer=answer,
        )

        return PipelineResult(
            user_input=user_input,
            signals=signals,
            write_decisions=write_decisions,
            query_info=query_info,
            judgments=judgments,
            answer=answer,
            goal_plan=goal_plan,
            verify_result=verify_result,
            structured_output=structured_output,
        )

    def _make_raw_interaction(self, user_input: str) -> RawInteraction:
        return RawInteraction(
            interaction_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
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
            subject=signal.subject,
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
            subject=signal.subject,
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
        if not self._should_answer(user_input, signals):
            return None

        resolution = self.resolver.resolve(
            user_input,
            recent_interactions,
            memories,
            constraints,
        )
        query_scope = self._merge_scopes(signals)

        return QueryInfo(
            query=user_input,
            query_intent=self._infer_query_intent(signals),
            query_scope=canonicalize_scopes(query_scope),
            resolved_subject=resolution.get("resolved_subject", "user"),
            requires_action=True,
            requires_plan=False,
            possible_old_premises=[],
        )

    def _should_answer(self, user_input: str, signals: list[MemorySignal]) -> bool:
        return any(signal.signal_type in ANSWER_TRIGGER_SIGNAL_TYPES for signal in signals)

    def _infer_query_intent(self, signals: list[MemorySignal]) -> str:
        if any(signal.signal_type == "query_intent" for signal in signals):
            return "explicit_query_intent"
        if any(signal.signal_type == "question_premise" for signal in signals):
            return "question_with_premise"
        if any(signal.signal_type == "hypothetical" for signal in signals):
            return "hypothetical_query"
        if any(signal.signal_type == "uncertain_intention" for signal in signals):
            return "uncertain_intention_query"
        return "implicit_query"

    def _merge_scopes(self, signals: list[MemorySignal]) -> list[str]:
        scopes: list[str] = []
        for signal in signals:
            for scope in signal.scope:
                if scope not in scopes:
                    scopes.append(scope)
        return canonicalize_scopes(scopes or ["general"])

    def _build_structured_output(
        self,
        user_input: str,
        signals: list[MemorySignal],
        write_decisions: list,
        query_info: QueryInfo | None,
        retrieved_memories: list[OrdinaryMemory],
        retrieved_constraints: list[ActiveConstraint],
        judgments: list,
        judge_diagnostics: dict | None,
        goal_plan: dict | None,
        verify_result: dict | None,
        answer: str | None,
    ) -> dict:
        return {
            "version": 1,
            "user_input": user_input,
            "query_info": self._jsonable(query_info),
            "signals": self._jsonable(signals),
            "write_decisions": self._jsonable(write_decisions),
            "retrieved_memories": self._jsonable(retrieved_memories),
            "retrieved_constraints": self._jsonable(retrieved_constraints),
            "judgments": self._jsonable(judgments),
            "status_summary": self._judgment_status_summary(judgments),
            "judge_diagnostics": self._jsonable(judge_diagnostics),
            "goal_plan": self._jsonable(goal_plan),
            "verify_result": self._jsonable(verify_result),
            "answer": answer,
        }

    def _get_judge_diagnostics(self) -> dict | None:
        diagnostics = getattr(self.usability_judge, "last_diagnostics", None)
        if diagnostics is None:
            return None
        return diagnostics

    def _jsonable(self, value):
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        if isinstance(value, dict):
            return {key: self._jsonable(item) for key, item in value.items()}
        return value

    def _judgment_status_summary(self, judgments: list) -> dict:
        summary = {
            "USABLE": 0,
            "CONSTRAINED": 0,
            "STALE": 0,
            "SUSPENDED": 0,
            "NOISE": 0,
        }
        for judgment in judgments:
            status = getattr(judgment, "usage_status", None)
            if status in summary:
                summary[status] += 1
        return summary
