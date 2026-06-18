"""Run CADMR demo cases with OpenRouter-backed extraction."""

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
from cadmr.config import load_dotenv


DEMO_CASES = [
    {
        "name": "state-level food case",
        "turns": [
            "I like very spicy food, especially Sichuan food and hotpot.",
            "My doctor said I cannot eat spicy food during the next four weeks of gastritis recovery. Can I still have hotpot tonight?",
        ],
        "print_fields": ["signals", "write_decisions", "query_info", "judgments", "answer", "verify_result"],
    },
    {
        "name": "discourse-level cat case",
        "turns": [
            "I only have mild lactose intolerance.",
            "My cat just came back from the animal hospital, and the vet said its diet needs to be handled carefully.",
            "Since I only have mild lactose intolerance, it should be fine if it drinks a little milk occasionally, right?",
        ],
        "print_fields": ["query_info", "judgments", "answer"],
    },
    {
        "name": "goal-level external demo case",
        "turns": [
            "This long-term memory system demo was first intended for internal technical teammates, so a prototype run-through was enough.",
            "Tomorrow I need to demo it to partners and faculty supervisors. I cannot show real user data, and examples must be anonymized.",
            "Help me arrange the safest demo plan.",
        ],
        "print_fields": ["goal_plan", "answer", "verify_result"],
    },
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


def print_field(name: str, value) -> None:
    print(f"{name}:")
    print(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2))


def main() -> int:
    load_dotenv()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for case in DEMO_CASES:
                print("=" * 80)
                print(f"Case: {case['name']}")
                pipeline = make_pipeline(Path(tmp) / case["name"].replace(" ", "_"))
                result = None
                for turn in case["turns"]:
                    result = pipeline.run(turn)
                assert result is not None
                print(f"Final query: {case['turns'][-1]}")
                for field in case["print_fields"]:
                    print_field(field, getattr(result, field))
        return 0
    except ValueError as error:
        if "OPENROUTER_API_KEY" in str(error):
            print("OPENROUTER_API_KEY is not set. Set it before running the demo.")
            return 0
        raise


if __name__ == "__main__":
    raise SystemExit(main())
