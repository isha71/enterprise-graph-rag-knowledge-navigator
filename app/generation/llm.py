"""LLM provider interface and implementations."""
import asyncio
import json
from typing import Protocol
import httpx
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.exceptions import LLMError

logger = get_logger(__name__)

_GROQ_MAX_RATE_LIMIT_RETRIES = 3
_GROQ_BACKOFF_SCHEDULE = [5, 10, 20]


class LLMClient(Protocol):
    async def generate(self, prompt: str, temperature: float = 0) -> str: ...
    async def generate_json(self, prompt: str, temperature: float = 0, **kwargs) -> dict: ...


class OllamaLLMClient:
    def __init__(self):
        settings = get_settings()
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_model
        if not self._model:
            raise LLMError("OLLAMA_MODEL not configured")
    
    async def generate(self, prompt: str, temperature: float = 0) -> str:
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
    
    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks from model output."""
        import re
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    async def generate_json(self, prompt: str, temperature: float = 0) -> dict:
        raw = await self.generate(prompt, temperature)
        raw = self._strip_thinking(raw).strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMError(f"Failed to parse LLM JSON: {e}\nRaw: {raw[:500]}")


class OpenAILLMClient:
    def __init__(self):
        settings = get_settings()
        self._api_key = settings.openai_api_key
        self._model = settings.openai_model or "gpt-3.5-turbo"
        if not self._api_key:
            raise LLMError("OPENAI_API_KEY not configured")
    
    async def generate(self, prompt: str, temperature: float = 0) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    
    async def generate_json(self, prompt: str, temperature: float = 0) -> dict:
        raw = await self.generate(prompt, temperature)
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMError(f"Failed to parse LLM JSON: {e}\nRaw: {raw[:500]}")


class GroqLLMClient:
    def __init__(self):
        settings = get_settings()
        self._api_key = settings.groq_api_key
        self._base_url = settings.groq_base_url.rstrip("/")
        self._model = settings.groq_model
        if not self._api_key:
            raise LLMError("GROQ_API_KEY not configured")

    async def _post_with_rate_limit_retry(
        self, url: str, headers: dict, payload: dict,
    ) -> httpx.Response:
        """POST with automatic retry on 429 responses."""
        for attempt in range(_GROQ_MAX_RATE_LIMIT_RETRIES + 1):
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload, headers=headers)

            if response.status_code != 429 or attempt == _GROQ_MAX_RATE_LIMIT_RETRIES:
                response.raise_for_status()
                return response

            retry_after = response.headers.get("retry-after")
            if retry_after is not None:
                wait = float(retry_after)
            else:
                wait = _GROQ_BACKOFF_SCHEDULE[min(attempt, len(_GROQ_BACKOFF_SCHEDULE) - 1)]

            logger.warning("Groq rate-limited, waiting %.1fs before retry", wait)
            await asyncio.sleep(wait)

        return response  # unreachable, but satisfies type checkers

    async def generate(self, prompt: str, temperature: float = 0) -> str:
        url = f"{self._base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        response = await self._post_with_rate_limit_retry(url, headers, payload)
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def generate_json(
        self, prompt: str, temperature: float = 0,
        max_completion_tokens: int | None = None,
    ) -> dict:
        url = f"{self._base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "reasoning_effort": "low",
            "reasoning_format": "hidden",
            "max_completion_tokens": max_completion_tokens if max_completion_tokens is not None else 1000,
        }
        response = await self._post_with_rate_limit_retry(url, headers, payload)
        data = response.json()
        raw = data["choices"][0]["message"]["content"].strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMError(f"Failed to parse LLM JSON: {e}\nRaw: {raw[:500]}")


def get_llm_client() -> LLMClient:
    """Factory for LLM client based on configuration."""
    settings = get_settings()
    if settings.llm_provider == "groq":
        return GroqLLMClient()
    if settings.llm_provider == "openai":
        return OpenAILLMClient()
    return OllamaLLMClient()
