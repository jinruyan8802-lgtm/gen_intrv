import os
import pytest
from config import Config

# Use non-existent path to avoid loading real .env during tests
FAKE_ENV = "/nonexistent/.env"


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://test.com/v1")
    monkeypatch.setenv("TEXT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LANGUAGE", "en")
    monkeypatch.setenv("QUESTION_COUNT", "15")
    monkeypatch.setenv("TTS_CONCURRENCY", "5")
    monkeypatch.setenv("DEDUP_TFIDF_THRESHOLD", "0.6")
    monkeypatch.setenv("TTS_INTERVIEWER_VOICE", "onyx")
    monkeypatch.setenv("TTS_CANDIDATE_VOICE", "shimmer")

    cfg = Config.from_env(FAKE_ENV)
    assert cfg.openai_api_key == "sk-test"
    assert cfg.openai_base_url == "https://test.com/v1"
    assert cfg.text_model == "gpt-4o-mini"
    assert cfg.language == "en"
    assert cfg.question_count == 15
    assert cfg.tts_concurrency == 5
    assert cfg.dedup_tfidf_threshold == 0.6
    assert cfg.tts_interviewer_voice == "onyx"
    assert cfg.tts_candidate_voice == "shimmer"


def test_config_tts_defaults_to_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-main")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://main.com/v1")
    monkeypatch.delenv("TTS_API_KEY", raising=False)
    monkeypatch.delenv("TTS_BASE_URL", raising=False)

    cfg = Config.from_env(FAKE_ENV)
    assert cfg.tts_api_key == "sk-main"
    assert cfg.tts_base_url == "https://main.com/v1"


def test_config_tts_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-main")
    monkeypatch.setenv("TTS_API_KEY", "sk-tts")
    monkeypatch.setenv("TTS_BASE_URL", "https://tts.com/v1")

    cfg = Config.from_env(FAKE_ENV)
    assert cfg.tts_api_key == "sk-tts"
    assert cfg.tts_base_url == "https://tts.com/v1"


def test_config_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Config.from_env(FAKE_ENV)
