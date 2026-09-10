"""Tests for GroqLLMClient: JSON mode, reasoning settings, and rate-limit retry."""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.exceptions import LLMError


def _run(coro):
    return asyncio.run(coro)


def _make_groq_client():
    """Create a GroqLLMClient with mocked settings."""
    from app.generation.llm import GroqLLMClient

    with patch("app.generation.llm.get_settings") as mock:
        mock.return_value.groq_api_key = "gsk-test"
        mock.return_value.groq_base_url = "https://api.groq.com/openai/v1"
        mock.return_value.groq_model = "openai/gpt-oss-20b"
        return GroqLLMClient()


def _mock_response(status_code=200, body=None, headers=None):
    """Build a fake httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.json.return_value = body or {
        "choices": [{"message": {"content": '{"entities": [], "relationships": []}'}}]
    }
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import httpx as _httpx

        def _raise():
            raise _httpx.HTTPStatusError(
                "rate limited", request=MagicMock(), response=resp,
            )

        resp.raise_for_status = _raise
    return resp


# ---- generate_json sends response_format, reasoning_effort, reasoning_format ----

def test_generate_json_sends_json_object_mode():
    """generate_json must include response_format=json_object in the payload."""
    client = _make_groq_client()
    captured = {}

    async def _intercept(*args, **kwargs):
        captured["payload"] = kwargs.get("json", args[1] if len(args) > 1 else None)
        return _mock_response()

    async def _test():
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = _intercept
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            await client.generate_json("Extract entities")

    _run(_test())
    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_generate_json_sends_reasoning_effort_low():
    client = _make_groq_client()
    captured = {}

    async def _intercept(*args, **kwargs):
        captured["payload"] = kwargs.get("json", args[1] if len(args) > 1 else None)
        return _mock_response()

    async def _test():
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = _intercept
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            await client.generate_json("Extract entities")

    _run(_test())
    assert captured["payload"]["reasoning_effort"] == "low"


def test_generate_json_sends_reasoning_format_hidden():
    client = _make_groq_client()
    captured = {}

    async def _intercept(*args, **kwargs):
        captured["payload"] = kwargs.get("json", args[1] if len(args) > 1 else None)
        return _mock_response()

    async def _test():
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = _intercept
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            await client.generate_json("Extract entities")

    _run(_test())
    assert captured["payload"]["reasoning_format"] == "hidden"


def test_generate_json_sends_max_completion_tokens():
    """Default max_completion_tokens is 1000 (graph extraction budget)."""
    client = _make_groq_client()
    captured = {}

    async def _intercept(*args, **kwargs):
        captured["payload"] = kwargs.get("json", args[1] if len(args) > 1 else None)
        return _mock_response()

    async def _test():
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = _intercept
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            await client.generate_json("Extract entities")

    _run(_test())
    assert captured["payload"]["max_completion_tokens"] == 1000


def test_generate_json_custom_max_completion_tokens():
    """Custom max_completion_tokens overrides the default."""
    client = _make_groq_client()
    captured = {}

    async def _intercept(*args, **kwargs):
        captured["payload"] = kwargs.get("json", args[1] if len(args) > 1 else None)
        return _mock_response()

    async def _test():
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = _intercept
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            await client.generate_json("Analyze query", max_completion_tokens=200)

    _run(_test())
    assert captured["payload"]["max_completion_tokens"] == 200


def test_generate_does_not_send_json_mode_fields():
    """Plain generate() must NOT include response_format or reasoning settings."""
    client = _make_groq_client()
    captured = {}

    async def _intercept(*args, **kwargs):
        captured["payload"] = kwargs.get("json", args[1] if len(args) > 1 else None)
        return _mock_response(body={
            "choices": [{"message": {"content": "hello"}}]
        })

    async def _test():
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = _intercept
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            await client.generate("Tell me a joke")

    _run(_test())
    assert "response_format" not in captured["payload"]
    assert "reasoning_effort" not in captured["payload"]
    assert "reasoning_format" not in captured["payload"]
    assert "max_completion_tokens" not in captured["payload"]


# ---- Rate-limit retry with Retry-After header ----

def test_429_respects_retry_after_header():
    """On 429 with Retry-After header, client should sleep for that duration then retry."""
    client = _make_groq_client()
    slept = []
    call_count = [0]

    async def _intercept(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _mock_response(
                status_code=429, headers={"retry-after": "2"},
            )
        return _mock_response()

    async def _fake_sleep(duration):
        slept.append(duration)

    async def _test():
        with (
            patch("httpx.AsyncClient") as MockClient,
            patch("app.generation.llm.asyncio.sleep", side_effect=_fake_sleep),
        ):
            instance = AsyncMock()
            instance.post = _intercept
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = await client.generate("test")

    _run(_test())
    assert slept == [2.0], f"Expected sleep of 2.0s from Retry-After header, got {slept}"
    assert call_count[0] == 2


# ---- Fallback exponential backoff ----

def test_429_exponential_backoff_without_retry_after():
    """Without Retry-After, should use backoff schedule (5s, 10s, 20s)."""
    client = _make_groq_client()
    slept = []
    call_count = [0]

    async def _intercept(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:
            return _mock_response(status_code=429, headers={})
        return _mock_response()

    async def _fake_sleep(duration):
        slept.append(duration)

    async def _test():
        with (
            patch("httpx.AsyncClient") as MockClient,
            patch("app.generation.llm.asyncio.sleep", side_effect=_fake_sleep),
        ):
            instance = AsyncMock()
            instance.post = _intercept
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            await client.generate("test")

    _run(_test())
    assert slept == [5, 10], f"Expected backoff [5, 10], got {slept}"
    assert call_count[0] == 3


# ---- Retries stop after max attempts ----

def test_429_gives_up_after_max_retries():
    """After exhausting retries, the 429 should raise."""
    import httpx as _httpx

    client = _make_groq_client()
    slept = []

    async def _intercept(*args, **kwargs):
        return _mock_response(status_code=429, headers={})

    async def _fake_sleep(duration):
        slept.append(duration)

    async def _test():
        with (
            patch("httpx.AsyncClient") as MockClient,
            patch("app.generation.llm.asyncio.sleep", side_effect=_fake_sleep),
        ):
            instance = AsyncMock()
            instance.post = _intercept
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            with pytest.raises(_httpx.HTTPStatusError):
                await client.generate("test")

    _run(_test())
    assert len(slept) == 3, f"Expected 3 sleeps before giving up, got {len(slept)}"
    assert slept == [5, 10, 20]
