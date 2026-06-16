"""OpenRouter client for structured JSON completions."""

import json
import os
import urllib.error
import urllib.request


class OpenRouterClient:
    """OpenRouter Chat Completions client implementing the LLMClient interface."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required to use OpenRouterClient.")
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def complete_json(self, prompt: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict JSON generator. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter request failed: {error.code} {error_body}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"OpenRouter request failed: {error.reason}") from error

        data = json.loads(response_body)
        content = data["choices"][0]["message"]["content"]
        return self._parse_json_content(content)

    @staticmethod
    def _strip_json_fence(content: str) -> str:
        stripped = content.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @classmethod
    def _parse_json_content(cls, content: str) -> dict:
        cleaned = cls._strip_json_fence(content)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as error:
            preview = content[:500]
            raise ValueError(f"Failed to parse OpenRouter JSON content: {preview}") from error
        if not isinstance(parsed, dict):
            raise ValueError(f"OpenRouter JSON content must be an object: {content[:500]}")
        return parsed
