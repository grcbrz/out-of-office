from __future__ import annotations


import pytest

from src.serving.auth import validate_token_at_startup


def test_token_too_short_raises(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "short")
    with pytest.raises(RuntimeError, match="API_TOKEN"):
        validate_token_at_startup()


def test_token_exactly_32_chars_passes(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "a" * 32)
    token = validate_token_at_startup()
    assert len(token) == 32


def test_token_empty_raises(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "")
    with pytest.raises(RuntimeError):
        validate_token_at_startup()
