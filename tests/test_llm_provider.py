"""Tests for LLM provider selection and configuration."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.exceptions import LLMError
from app.generation.llm import (
    GroqLLMClient,
    OllamaLLMClient,
    OpenAILLMClient,
    get_llm_client,
)


# ---- Provider selection ----

def test_get_llm_client_ollama_default():
    """Default provider returns OllamaLLMClient."""
    with patch("app.generation.llm.get_settings") as mock:
        mock.return_value.llm_provider = "ollama"
        mock.return_value.ollama_model = "llama3"
        mock.return_value.ollama_base_url = "http://localhost:11434"
        client = get_llm_client()
    assert isinstance(client, OllamaLLMClient)


def test_get_llm_client_openai():
    with patch("app.generation.llm.get_settings") as mock:
        mock.return_value.llm_provider = "openai"
        mock.return_value.openai_api_key = "sk-test"
        mock.return_value.openai_model = "gpt-4"
        client = get_llm_client()
    assert isinstance(client, OpenAILLMClient)


def test_get_llm_client_groq():
    with patch("app.generation.llm.get_settings") as mock:
        mock.return_value.llm_provider = "groq"
        mock.return_value.groq_api_key = "gsk-test"
        mock.return_value.groq_base_url = "https://api.groq.com/openai/v1"
        mock.return_value.groq_model = "openai/gpt-oss-20b"
        client = get_llm_client()
    assert isinstance(client, GroqLLMClient)


def test_get_llm_client_unknown_falls_back_to_ollama():
    """Any unrecognised provider string falls back to Ollama."""
    with patch("app.generation.llm.get_settings") as mock:
        mock.return_value.llm_provider = "unknown_provider"
        mock.return_value.ollama_model = "llama3"
        mock.return_value.ollama_base_url = "http://localhost:11434"
        client = get_llm_client()
    assert isinstance(client, OllamaLLMClient)


# ---- Missing credentials raise LLMError ----

def test_groq_missing_api_key_raises():
    with patch("app.generation.llm.get_settings") as mock:
        mock.return_value.groq_api_key = ""
        mock.return_value.groq_base_url = "https://api.groq.com/openai/v1"
        mock.return_value.groq_model = "openai/gpt-oss-20b"
        with pytest.raises(LLMError, match="GROQ_API_KEY not configured"):
            GroqLLMClient()


def test_openai_missing_api_key_raises():
    with patch("app.generation.llm.get_settings") as mock:
        mock.return_value.openai_api_key = ""
        mock.return_value.openai_model = "gpt-4"
        with pytest.raises(LLMError, match="OPENAI_API_KEY not configured"):
            OpenAILLMClient()


def test_ollama_missing_model_raises():
    with patch("app.generation.llm.get_settings") as mock:
        mock.return_value.ollama_model = ""
        mock.return_value.ollama_base_url = "http://localhost:11434"
        with pytest.raises(LLMError, match="OLLAMA_MODEL not configured"):
            OllamaLLMClient()


# ---- Groq configuration ----

def test_groq_base_url_trailing_slash_stripped():
    with patch("app.generation.llm.get_settings") as mock:
        mock.return_value.groq_api_key = "gsk-test"
        mock.return_value.groq_base_url = "https://api.groq.com/openai/v1/"
        mock.return_value.groq_model = "openai/gpt-oss-20b"
        client = GroqLLMClient()
    assert client._base_url == "https://api.groq.com/openai/v1"


def test_groq_stores_model():
    with patch("app.generation.llm.get_settings") as mock:
        mock.return_value.groq_api_key = "gsk-test"
        mock.return_value.groq_base_url = "https://api.groq.com/openai/v1"
        mock.return_value.groq_model = "mixtral-8x7b"
        client = GroqLLMClient()
    assert client._model == "mixtral-8x7b"


# ---- Config defaults ----

def test_config_groq_defaults():
    from app.core.config import Settings

    s = Settings(
        _env_file=None,
        ollama_model="llama3",
    )
    assert s.groq_api_key == ""
    assert s.groq_base_url == "https://api.groq.com/openai/v1"
    assert s.groq_model == "openai/gpt-oss-20b"
