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


DEMO_CASES = [
    {
        "name": "state-level food case",
        "turns": [
            "我喜欢重辣口味，尤其喜欢川菜和火锅。",
            "医生说我接下来四周胃炎恢复期不能吃辣，那今晚还能吃火锅吗？",
        ],
        "print_fields": ["signals", "write_decisions", "query_info", "judgments", "answer", "verify_result"],
    },
    {
        "name": "discourse-level cat case",
        "turns": [
            "我只是轻度乳糖不耐受。",
            "我的猫刚从医院回来，兽医说饮食需要小心。",
            "既然我只是轻度乳糖不耐受，那它偶尔喝点牛奶应该没事吧？",
        ],
        "print_fields": ["query_info", "judgments", "answer"],
    },
    {
        "name": "goal-level external demo case",
        "turns": [
            "这个长期记忆系统 demo 先给内部技术同事看，只要跑通原型就行。",
            "明天要给合作方和管理老师演示，不能展示真实用户数据，需要脱敏。",
            "帮我安排一个最稳妥的演示方案。",
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
