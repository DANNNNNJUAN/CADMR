def test_modules_can_be_imported():
    import cadmr.answer_generator
    import cadmr.extractor
    import cadmr.pipeline
    import cadmr.retrieval
    import cadmr.openrouter_client
    import cadmr.resolver
    import cadmr.schemas
    import cadmr.stores
    import cadmr.usability_judge
    import cadmr.verifier
    import cadmr.write_gate
    import cadmr.goal_reconstructor


def test_classes_can_be_imported():
    from cadmr.answer_generator import ConstrainedAnswerGenerator
    from cadmr.extractor import MemorySignalExtractor, RuleBasedMemorySignalExtractor
    from cadmr.pipeline import CADMRPipeline
    from cadmr.openrouter_client import OpenRouterClient
    from cadmr.retrieval import MemoryRetriever
    from cadmr.resolver import ReferentTopicResolver
    from cadmr.stores import ActiveConstraintStore, OrdinaryMemoryStore, RawInteractionLog
    from cadmr.usability_judge import MemoryUsabilityJudge
    from cadmr.verifier import AnswerVerifier
    from cadmr.write_gate import MemoryWriteGate
    from cadmr.goal_reconstructor import GoalReconstructor

    assert ConstrainedAnswerGenerator
    assert MemorySignalExtractor
    assert RuleBasedMemorySignalExtractor
    assert CADMRPipeline
    assert OpenRouterClient
    assert MemoryRetriever
    assert ReferentTopicResolver
    assert ActiveConstraintStore
    assert OrdinaryMemoryStore
    assert RawInteractionLog
    assert MemoryUsabilityJudge
    assert AnswerVerifier
    assert MemoryWriteGate
    assert GoalReconstructor
