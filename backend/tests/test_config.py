from pathlib import Path

import pytest
from app.core.config import (
    DEFAULT_DEVICE,
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_ID,
    AppSettings,
)


ENV_KEYS = [
    "APP_MODEL_ID",
    "APP_MAX_INPUT_CHARS",
    "APP_MAX_NEW_TOKENS",
    "APP_DEVICE",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
]


def _clear_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_key in ENV_KEYS:
        monkeypatch.delenv(env_key, raising=False)


def test_from_env_uses_defaults_when_variables_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    _clear_config_env(monkeypatch)

    # When
    settings = AppSettings.from_env()

    # Then
    assert settings.model_id == DEFAULT_MODEL_ID
    assert settings.max_input_chars == DEFAULT_MAX_INPUT_CHARS
    assert settings.max_new_tokens == DEFAULT_MAX_NEW_TOKENS
    assert settings.device == DEFAULT_DEVICE
    assert settings.hf_home is None
    assert settings.hf_hub_cache is None
    assert settings.transformers_cache is None


def test_from_env_falls_back_for_empty_values(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("APP_MODEL_ID", "   ")
    monkeypatch.setenv("APP_MAX_INPUT_CHARS", "")
    monkeypatch.setenv("APP_MAX_NEW_TOKENS", " ")
    monkeypatch.setenv("APP_DEVICE", "\t")
    monkeypatch.setenv("HF_HOME", " ")
    monkeypatch.setenv("HF_HUB_CACHE", " ")
    monkeypatch.setenv("TRANSFORMERS_CACHE", " ")

    # When
    settings = AppSettings.from_env()

    # Then
    assert settings.model_id == DEFAULT_MODEL_ID
    assert settings.max_input_chars == DEFAULT_MAX_INPUT_CHARS
    assert settings.max_new_tokens == DEFAULT_MAX_NEW_TOKENS
    assert settings.device == DEFAULT_DEVICE
    assert settings.hf_home is None
    assert settings.hf_hub_cache is None
    assert settings.transformers_cache is None


@pytest.mark.parametrize(
    ("key", "value"), [("APP_MAX_INPUT_CHARS", "0"), ("APP_MAX_NEW_TOKENS", "-1")]
)
def test_from_env_rejects_non_positive_integer_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
) -> None:
    # Given
    _clear_config_env(monkeypatch)
    monkeypatch.setenv(key, value)

    # When / Then
    with pytest.raises(ValueError, match=key):
        AppSettings.from_env()


@pytest.mark.parametrize("device", ["cpu", "CPU", "cuda", "cuda:0", "mps"])
def test_from_env_accepts_supported_device_values(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    # Given
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("APP_DEVICE", device)

    # When
    settings = AppSettings.from_env()

    # Then
    assert settings.device == device.lower()


@pytest.mark.parametrize("device", ["gpu", "cuda:-1", "cuda:abc"])
def test_from_env_rejects_invalid_device_values(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    # Given
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("APP_DEVICE", device)

    # When / Then
    with pytest.raises(ValueError, match="APP_DEVICE"):
        AppSettings.from_env()


def test_resolve_cache_dir_prefers_hf_hub_cache_then_transformers_then_hf_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("HF_HUB_CACHE", "/tmp/hub")
    monkeypatch.setenv("TRANSFORMERS_CACHE", "/tmp/transformers")
    monkeypatch.setenv("HF_HOME", "/tmp/home")

    # When
    settings = AppSettings.from_env()

    # Then
    assert settings.resolve_cache_dir() == "/tmp/hub"


def test_resolve_cache_dir_falls_back_to_hf_home_hub_when_explicit_cache_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("HF_HOME", "/tmp/home")

    # When
    settings = AppSettings.from_env()

    # Then
    assert settings.resolve_cache_dir() == str(Path("/tmp/home") / "hub")
