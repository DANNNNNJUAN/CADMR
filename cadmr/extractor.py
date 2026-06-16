"""Memory signal extraction interfaces and lightweight implementations."""

import uuid

from cadmr.schemas import MemorySignal


ALLOWED_SIGNAL_TYPES = {
    "ordinary_memory",
    "active_constraint",
    "query_intent",
    "hypothetical",
    "question_premise",
    "uncertain_intention",
}

DEFAULT_CONFIDENCE = {
    "ordinary_memory": 0.8,
    "active_constraint": 0.9,
    "query_intent": 0.8,
    "hypothetical": 0.75,
    "question_premise": 0.75,
    "uncertain_intention": 0.65,
}

ORDINARY_MEMORY_KEYWORDS = [
    "我喜欢",
    "我偏好",
    "我习惯",
    "我一般",
    "我经常",
    "我正在做",
    "我在准备",
    "我正在准备",
    "我的研究方向",
    "我现在住在",
    "我住在",
    "我现在在",
    "我已经搬到",
    "我搬到",
    "我搬去",
    "我只是",
    "我的猫",
]
ACTIVE_CONSTRAINT_KEYWORDS = [
    "医生说",
    "不能",
    "不要",
    "避免",
    "禁止",
    "必须",
    "需要",
    "四周",
    "三个月内",
    "胃炎",
    "受伤",
    "过敏",
    "首付",
    "隐私",
    "脱敏",
    "真实数据",
    "敏感",
    "兽医说",
    "饮食需要小心",
]
QUERY_INTENT_KEYWORDS = ["能不能", "还能", "帮我", "推荐", "安排", "怎么", "如何", "吗", "？", "?"]
HYPOTHETICAL_KEYWORDS = ["如果", "假如", "假设", "以后如果", "万一"]
QUESTION_PREMISE_KEYWORDS = ["既然我", "既然之前", "既然已经"]
UNCERTAIN_INTENTION_KEYWORDS = ["可能想", "也许", "不确定", "我在考虑", "我有点想"]


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def infer_scope(text: str) -> list[str]:
    scope_rules = [
        (["辣", "火锅", "川菜", "饮食", "吃", "餐厅", "牛奶", "乳糖"], ["diet", "preference"]),
        (["医生", "胃炎", "过敏", "受伤", "不能吃辣", "不耐受", "兽医", "医院", "肾功能", "饮食需要小心"], ["health", "diet"]),
        (["跑步", "运动", "健身"], ["exercise", "habit"]),
        (["投资", "风险", "资产", "首付", "资金", "预算"], ["finance"]),
        (["隐私", "真实数据", "脱敏", "敏感"], ["privacy", "safety"]),
        (["demo", "汇报", "PPT", "研究", "论文", "项目", "演示"], ["work", "project"]),
        (["通勤", "骑车", "地铁", "上海", "成都", "北京", "附近"], ["location", "transport"]),
    ]

    scopes: list[str] = []
    for keywords, inferred_scopes in scope_rules:
        if contains_any(text, keywords):
            for scope in inferred_scopes:
                if scope not in scopes:
                    scopes.append(scope)

    return scopes or ["general"]


class MemorySignalExtractor:
    """Base interface for extracting memory-related signals from text."""

    def extract(self, text: str) -> list[MemorySignal]:
        raise NotImplementedError


class RuleBasedMemorySignalExtractor(MemorySignalExtractor):
    """Rule-based extractor for a minimal runnable CADMR version."""

    def extract(self, text: str) -> list[MemorySignal]:
        rules = [
            ("ordinary_memory", ORDINARY_MEMORY_KEYWORDS, 0.85),
            ("active_constraint", ACTIVE_CONSTRAINT_KEYWORDS, 0.9),
            ("query_intent", QUERY_INTENT_KEYWORDS, 0.8),
            ("hypothetical", HYPOTHETICAL_KEYWORDS, 0.8),
            ("question_premise", QUESTION_PREMISE_KEYWORDS, 0.75),
            ("uncertain_intention", UNCERTAIN_INTENTION_KEYWORDS, 0.65),
        ]

        signals: list[MemorySignal] = []
        for signal_type, keywords, confidence in rules:
            if contains_any(text, keywords):
                signals.append(
                    MemorySignal(
                        signal_id=str(uuid.uuid4()),
                        signal_type=signal_type,
                        content=text,
                        subject="user",
                        scope=infer_scope(text),
                        confidence=confidence,
                        evidence_text=text,
                    )
                )

        return signals


class LLMClient:
    """Abstract client for structured LLM JSON completion."""

    def complete_json(self, prompt: str) -> dict:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic LLM client used by tests and local development."""

    def __init__(self, response: dict):
        self.response = response

    def complete_json(self, prompt: str) -> dict:
        return self.response


class RuleValidator:
    """Validate and repair LLM-extracted memory signals using safety rules."""

    def validate(self, raw_items: list[dict], original_text: str) -> list[MemorySignal]:
        signals: list[MemorySignal] = []
        seen: set[tuple[str, str]] = set()

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            signal_type = item.get("signal_type")
            if signal_type not in ALLOWED_SIGNAL_TYPES:
                continue

            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            content = content.strip()

            subject = item.get("subject")
            if not isinstance(subject, str) or not subject.strip():
                subject = "user"

            scope = item.get("scope")
            if not self._is_string_list(scope):
                scope = infer_scope(content)
            scope = scope or ["general"]

            signal_type = self._repair_signal_type(signal_type, content, original_text)
            signal_type = self._downgrade_unsupported_constraint(
                signal_type,
                content,
                original_text,
                scope,
            )

            confidence = item.get("confidence", DEFAULT_CONFIDENCE[signal_type])
            if not isinstance(confidence, int | float):
                confidence = DEFAULT_CONFIDENCE[signal_type]
            confidence = max(0.0, min(1.0, float(confidence)))

            evidence_text = item.get("evidence_text")
            if not isinstance(evidence_text, str) or not evidence_text.strip():
                evidence_text = original_text

            dedupe_key = (signal_type, content)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            signals.append(
                MemorySignal(
                    signal_id=str(uuid.uuid4()),
                    signal_type=signal_type,
                    content=content,
                    subject=subject.strip(),
                    scope=scope,
                    confidence=confidence,
                    evidence_text=evidence_text,
                )
            )

        return signals

    def _is_string_list(self, value) -> bool:
        return isinstance(value, list) and all(isinstance(item, str) for item in value)

    def _repair_signal_type(self, signal_type: str, content: str, original_text: str) -> str:
        combined_text = f"{content}\n{original_text}"

        if signal_type in {"ordinary_memory", "active_constraint"} and contains_any(
            combined_text,
            HYPOTHETICAL_KEYWORDS,
        ):
            return "hypothetical"

        if signal_type in {"ordinary_memory", "active_constraint"} and contains_any(
            combined_text,
            QUESTION_PREMISE_KEYWORDS,
        ):
            return "question_premise"

        if signal_type == "ordinary_memory" and contains_any(
            combined_text,
            UNCERTAIN_INTENTION_KEYWORDS,
        ):
            return "uncertain_intention"

        return signal_type

    def _downgrade_unsupported_constraint(
        self,
        signal_type: str,
        content: str,
        original_text: str,
        scope: list[str],
    ) -> str:
        if signal_type != "active_constraint":
            return signal_type

        combined_text = f"{content}\n{original_text}"
        constraint_scopes = {"health", "finance", "privacy", "safety", "task"}
        has_constraint_keyword = contains_any(combined_text, ACTIVE_CONSTRAINT_KEYWORDS + ["预算"])
        has_constraint_scope = bool(constraint_scopes.intersection(scope))

        if has_constraint_keyword or has_constraint_scope:
            return signal_type
        return "ordinary_memory"


class LLMMemorySignalExtractor(MemorySignalExtractor):
    """LLM extractor with rule validation and pollution prevention."""

    def __init__(self, llm_client: LLMClient, validator: RuleValidator | None = None):
        self.llm_client = llm_client
        self.validator = validator or RuleValidator()

    def extract(self, text: str) -> list[MemorySignal]:
        prompt = self._build_prompt(text)
        raw_output = self.llm_client.complete_json(prompt)
        raw_items = raw_output.get("signals", [])
        if not isinstance(raw_items, list):
            raw_items = []
        return self.validator.validate(raw_items, original_text=text)

    def _build_prompt(self, text: str) -> str:
        return f"""
从用户输入中抽取 CADMR memory signals，并只输出严格 JSON。

允许的 signal_type：ordinary_memory, active_constraint, query_intent,
hypothetical, question_premise, uncertain_intention。

要求：
- 不要把假设写成事实。
- 不要把问题中的旧前提写成 active memory。
- 不要把不确定想法写成稳定偏好。
- query_intent 不代表要写入长期记忆。
- 一条输入可以输出多个 signals。

输出格式：
{{
  "signals": [
    {{
      "signal_type": "ordinary_memory | active_constraint | query_intent | hypothetical | question_premise | uncertain_intention",
      "content": "...",
      "subject": "user",
      "scope": ["..."],
      "confidence": 0.0,
      "evidence_text": "..."
    }}
  ]
}}

用户输入：{text}
""".strip()
