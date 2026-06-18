"""Run CADMR hard-case evaluation with OpenRouter-backed extraction."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cadmr.extractor import LLMMemorySignalExtractor
from cadmr.openrouter_client import OpenRouterClient
from cadmr.pipeline import CADMRPipeline
from cadmr.stores import ActiveConstraintStore, OrdinaryMemoryStore, RawInteractionLog
from cadmr.config import get_env, load_dotenv

SAFE_NEGATION_KEYWORDS = [
    "not recommend",
    "not suitable",
    "avoid",
    "cannot",
    "can't",
    "should not",
    "do not",
]


def make_pipeline(tmp_dir: Path) -> CADMRPipeline:
    extractor = LLMMemorySignalExtractor(OpenRouterClient())
    return CADMRPipeline(
        raw_log=RawInteractionLog(tmp_dir / "raw_interaction_log.jsonl"),
        ordinary_store=OrdinaryMemoryStore(tmp_dir / "ordinary_memory.json"),
        constraint_store=ActiveConstraintStore(tmp_dir / "active_constraints.json"),
        extractor=extractor,
    )


def to_jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


def case_passed(result, expected: dict) -> bool:
    answer = result.answer or ""
    if expected.get("must_have_judgment") and not any(
        judgment.usage_status == expected["must_have_judgment"]
        for judgment in result.judgments
    ):
        return False

    if expected.get("resolved_subject"):
        if result.query_info is None or result.query_info.resolved_subject != expected["resolved_subject"]:
            return False

    include_any = expected.get("answer_must_include_any") or []
    if include_any and not any(item in answer for item in include_any):
        return False

    not_include_any = expected.get("answer_must_not_include_any") or []
    for item in not_include_any:
        if item in answer and not any(keyword in answer for keyword in SAFE_NEGATION_KEYWORDS):
            return False

    required_components = expected.get("goal_required_components") or []
    plan_components = []
    if result.goal_plan:
        plan_components = result.goal_plan.get("required_plan_components", [])
    for component in required_components:
        if component not in plan_components:
            return False

    return True


def run_case(case: dict, tmp_root: Path) -> dict:
    pipeline = make_pipeline(tmp_root / case["case_id"])
    result = None
    for turn in case["turns"]:
        result = pipeline.run(turn)
    assert result is not None

    simple_pass = case_passed(result, case.get("expected", {}))
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "query_info": to_jsonable(result.query_info),
        "judgments": to_jsonable(result.judgments),
        "answer": result.answer,
        "goal_plan": to_jsonable(result.goal_plan),
        "verify_result": to_jsonable(result.verify_result),
        "simple_pass": simple_pass,
    }


def main() -> int:
    load_dotenv()
    cases_path = PROJECT_ROOT / (get_env("CADMR_EVAL_CASES_PATH", "evals/hard_cases.json") or "evals/hard_cases.json")
    results_path = PROJECT_ROOT / (get_env("CADMR_EVAL_RESULTS_PATH", "evals/results_latest.json") or "evals/results_latest.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    try:
        with tempfile.TemporaryDirectory() as tmp:
            results = [run_case(case, Path(tmp)) for case in cases]
    except ValueError as error:
        if "OPENROUTER_API_KEY" in str(error):
            print("OPENROUTER_API_KEY is not set. Set it before running eval.")
            return 0
        raise

    passed = sum(1 for result in results if result["simple_pass"])
    for result in results:
        status = "PASS" if result["simple_pass"] else "FAIL"
        print(f"{status} {result['case_id']} ({result['category']})")
        print("Answer:", result["answer"])
        print("Judgments:", json.dumps(result["judgments"], ensure_ascii=False, indent=2))
        print()

    print(f"Passed {passed} / Total {len(results)}")
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
