from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path

import pytest
from app.core.config import (
    DEFAULT_DEVICE,
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_ID,
    AppSettings,
)

CONFIG_ENV_KEYS: tuple[str, ...] = (
    "APP_MODEL_ID",
    "APP_MAX_INPUT_CHARS",
    "APP_MAX_NEW_TOKENS",
    "APP_DEVICE",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
)


@contextmanager
def _temporary_os_env(overrides: dict[str, str]) -> Iterator[None]:
    previous_values = {env_key: os.environ.get(env_key) for env_key in CONFIG_ENV_KEYS}
    try:
        for env_key in CONFIG_ENV_KEYS:
            os.environ.pop(env_key, None)
        os.environ.update(overrides)
        yield
    finally:
        for env_key, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(env_key, None)
                continue
            os.environ[env_key] = previous_value


def test_from_env_uses_defaults_when_variables_are_missing() -> None:
    # When
    settings = AppSettings.from_env(env={})

    # Then
    assert settings.model_id == DEFAULT_MODEL_ID
    assert settings.max_input_chars == DEFAULT_MAX_INPUT_CHARS
    assert settings.max_new_tokens == DEFAULT_MAX_NEW_TOKENS
    assert settings.device == DEFAULT_DEVICE
    assert settings.hf_home is None
    assert settings.hf_hub_cache is None
    assert settings.transformers_cache is None


def test_from_env_falls_back_for_empty_values() -> None:
    # Given
    env = {
        "APP_MODEL_ID": "   ",
        "APP_MAX_INPUT_CHARS": "",
        "APP_MAX_NEW_TOKENS": " ",
        "APP_DEVICE": "\t",
        "HF_HOME": " ",
        "HF_HUB_CACHE": " ",
        "TRANSFORMERS_CACHE": " ",
    }

    # When
    settings = AppSettings.from_env(env=env)

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
    key: str,
    value: str,
) -> None:
    # Given
    env = {key: value}

    # When / Then
    with pytest.raises(ValueError, match=key):
        AppSettings.from_env(env=env)


def test_from_env_rejects_non_string_env_values() -> None:
    env = {"APP_MAX_INPUT_CHARS": 123}

    with pytest.raises(ValueError, match="APP_MAX_INPUT_CHARS"):
        AppSettings.from_env(env=env)


@pytest.mark.parametrize("device", ["cpu", "CPU", "cuda", "cuda:0", "mps"])
def test_from_env_accepts_supported_device_values(
    device: str,
) -> None:
    # Given
    env = {"APP_DEVICE": device}

    # When
    settings = AppSettings.from_env(env=env)

    # Then
    assert settings.device == device.lower()


@pytest.mark.parametrize("device", ["gpu", "cuda:-1", "cuda:abc"])
def test_from_env_rejects_invalid_device_values(
    device: str,
) -> None:
    # Given
    env = {"APP_DEVICE": device}

    # When / Then
    with pytest.raises(ValueError, match="APP_DEVICE"):
        AppSettings.from_env(env=env)


def test_resolve_cache_dir_prefers_hf_hub_cache_then_transformers_then_hf_home() -> (
    None
):
    # Given
    env = {
        "HF_HUB_CACHE": "/tmp/hub",
        "TRANSFORMERS_CACHE": "/tmp/transformers",
        "HF_HOME": "/tmp/home",
    }

    # When
    settings = AppSettings.from_env(env=env)

    # Then
    assert settings.resolve_cache_dir() == "/tmp/hub"


def test_resolve_cache_dir_falls_back_to_hf_home_hub_when_explicit_cache_is_missing() -> (
    None
):
    # Given
    env = {"HF_HOME": "/tmp/home"}

    # When
    settings = AppSettings.from_env(env=env)

    # Then
    assert settings.resolve_cache_dir() == str(Path("/tmp/home") / "hub")


def test_from_env_reads_default_os_environ_when_env_argument_is_missing() -> None:
    with _temporary_os_env(
        overrides={
            "APP_MODEL_ID": "custom/model",
            "APP_MAX_INPUT_CHARS": "2048",
            "APP_MAX_NEW_TOKENS": "123",
            "APP_DEVICE": "CUDA:0",
            "HF_HOME": "/tmp/home",
        }
    ):
        settings = AppSettings.from_env()

    assert settings.model_id == "custom/model"
    assert settings.max_input_chars == 2048
    assert settings.max_new_tokens == 123
    assert settings.device == "cuda:0"
    assert settings.hf_home == "/tmp/home"
    assert settings.hf_hub_cache is None
    assert settings.transformers_cache is None
