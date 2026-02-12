from collections.abc import Mapping
import os
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_ID = "facebook/nllb-200-distilled-600M"
DEFAULT_MAX_INPUT_TOKENS = 1024
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_FRONTEND_MAX_INPUT_TOKENS = 400
DEFAULT_FRONTEND_MAX_CHARS_PER_TOKEN = 2
DEFAULT_FRONTEND_WARNING_RATIO = 0.75
DEFAULT_REPETITION_PENALTY = 1.2
DEFAULT_DEVICE = "cpu"
DEVICE_PATTERN = re.compile(r"^(cpu|mps|cuda(?::\d+)?)$")


def _read_env_value(name: str, env: Mapping[str, object]) -> str | None:
    """Read environment variable and validate raw value type."""
    raw_value = env.get(name)
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise ValueError(f"Environment variable {name} must be a string.")
    return raw_value.strip()


def _read_str_env(name: str, default: str, env: Mapping[str, object]) -> str:
    """Read a string environment variable and fallback on empty values."""
    normalized_value = _read_env_value(name, env)
    if normalized_value is None:
        return default
    if not normalized_value:
        return default
    return normalized_value


def _read_optional_str_env(name: str, env: Mapping[str, object]) -> str | None:
    """Read an optional string environment variable and normalize empty values to None."""
    normalized_value = _read_env_value(name, env)
    if not normalized_value:
        return None
    return normalized_value


def _read_device_env(name: str, default: str, env: Mapping[str, object]) -> str:
    """Read and validate torch device from environment variables."""
    device_value = _read_str_env(name, default, env).lower()
    if not DEVICE_PATTERN.fullmatch(device_value):
        raise ValueError(
            f"Environment variable {name} must be one of: cpu, mps, cuda, cuda:<index>."
        )
    return device_value


def _read_int_env(name: str, default: int, env: Mapping[str, object]) -> int:
    """Read a positive integer environment variable with a fallback value."""
    normalized_value = _read_env_value(name, env)
    if not normalized_value:
        return default

    try:
        parsed_value = int(normalized_value)
    except ValueError as error:
        raise ValueError(f"Environment variable {name} must be an integer.") from error
    if parsed_value <= 0:
        raise ValueError(f"Environment variable {name} must be greater than zero.")
    return parsed_value


def _read_float_env(name: str, default: float, env: Mapping[str, object]) -> float:
    """Read a positive float environment variable with a fallback value."""
    normalized_value = _read_env_value(name, env)
    if not normalized_value:
        return default

    try:
        parsed_value = float(normalized_value)
    except ValueError as error:
        raise ValueError(f"Environment variable {name} must be a number.") from error
    if parsed_value <= 0:
        raise ValueError(f"Environment variable {name} must be greater than zero.")
    return parsed_value


def _read_ratio_env(name: str, default: float, env: Mapping[str, object]) -> float:
    """Read a ratio in range (0, 1) with a fallback value."""
    parsed_value = _read_float_env(name, default, env)
    if parsed_value >= 1:
        raise ValueError(
            f"Environment variable {name} must be lower than one and greater than zero."
        )
    return parsed_value


def _read_max_input_tokens_env(env: Mapping[str, object]) -> int:
    """Read max input tokens from environment."""
    return _read_int_env("APP_MAX_INPUT_TOKENS", DEFAULT_MAX_INPUT_TOKENS, env)


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Application runtime settings."""

    model_id: str = DEFAULT_MODEL_ID
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    frontend_max_input_tokens: int = DEFAULT_FRONTEND_MAX_INPUT_TOKENS
    frontend_max_chars_per_token: int = DEFAULT_FRONTEND_MAX_CHARS_PER_TOKEN
    frontend_warning_ratio: float = DEFAULT_FRONTEND_WARNING_RATIO
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    device: str = DEFAULT_DEVICE
    hf_home: str | None = None
    hf_hub_cache: str | None = None
    transformers_cache: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, object] | None = None) -> "AppSettings":
        """Build settings from environment variables."""
        resolved_env = env if env is not None else os.environ
        return cls(
            model_id=_read_str_env("APP_MODEL_ID", DEFAULT_MODEL_ID, resolved_env),
            max_input_tokens=_read_max_input_tokens_env(resolved_env),
            max_new_tokens=_read_int_env(
                "APP_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS, resolved_env
            ),
            frontend_max_input_tokens=_read_int_env(
                "APP_FRONTEND_MAX_INPUT_TOKENS",
                DEFAULT_FRONTEND_MAX_INPUT_TOKENS,
                resolved_env,
            ),
            frontend_max_chars_per_token=_read_int_env(
                "APP_FRONTEND_MAX_CHARS_PER_TOKEN",
                DEFAULT_FRONTEND_MAX_CHARS_PER_TOKEN,
                resolved_env,
            ),
            frontend_warning_ratio=_read_ratio_env(
                "APP_FRONTEND_WARNING_RATIO",
                DEFAULT_FRONTEND_WARNING_RATIO,
                resolved_env,
            ),
            repetition_penalty=_read_float_env(
                "APP_REPETITION_PENALTY",
                DEFAULT_REPETITION_PENALTY,
                resolved_env,
            ),
            device=_read_device_env("APP_DEVICE", DEFAULT_DEVICE, resolved_env),
            hf_home=_read_optional_str_env("HF_HOME", resolved_env),
            hf_hub_cache=_read_optional_str_env("HF_HUB_CACHE", resolved_env),
            transformers_cache=_read_optional_str_env(
                "TRANSFORMERS_CACHE", resolved_env
            ),
        )

    def resolve_frontend_max_input_chars(self) -> int:
        """Resolve max input characters for frontend input validation."""
        return self.frontend_max_input_tokens * self.frontend_max_chars_per_token

    def resolve_frontend_warning_input_chars(self) -> int:
        """Resolve warning threshold in characters for frontend input validation."""
        return int(
            self.resolve_frontend_max_input_chars() * self.frontend_warning_ratio
        )

    def resolve_cache_dir(self) -> str | None:
        """Resolve a cache directory for Hugging Face model artifacts."""
        if self.hf_hub_cache:
            return self.hf_hub_cache
        if self.transformers_cache:
            return self.transformers_cache
        if self.hf_home:
            return str(Path(self.hf_home) / "hub")
        return None
