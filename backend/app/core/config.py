import os
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_ID = "facebook/nllb-200-distilled-600M"
DEFAULT_MAX_INPUT_CHARS = 1024
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_DEVICE = "cpu"
DEVICE_PATTERN = re.compile(r"^(cpu|mps|cuda(?::\d+)?)$")


def _read_str_env(name: str, default: str) -> str:
    """Read a string environment variable and fallback on empty values."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized_value = raw_value.strip()
    if not normalized_value:
        return default
    return normalized_value


def _read_optional_str_env(name: str) -> str | None:
    """Read an optional string environment variable and normalize empty values to None."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return None

    normalized_value = raw_value.strip()
    if not normalized_value:
        return None
    return normalized_value


def _read_device_env(name: str, default: str) -> str:
    """Read and validate torch device from environment variables."""
    device_value = _read_str_env(name, default).lower()
    if not DEVICE_PATTERN.fullmatch(device_value):
        raise ValueError(
            f"Environment variable {name} must be one of: cpu, mps, cuda, cuda:<index>."
        )
    return device_value


def _read_int_env(name: str, default: int) -> int:
    """Read a positive integer environment variable with a fallback value."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized_value = raw_value.strip()
    if not normalized_value:
        return default

    try:
        parsed_value = int(normalized_value)
    except ValueError as error:
        raise ValueError(f"Environment variable {name} must be an integer.") from error
    if parsed_value <= 0:
        raise ValueError(f"Environment variable {name} must be greater than zero.")
    return parsed_value


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Application runtime settings."""

    model_id: str = DEFAULT_MODEL_ID
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    device: str = DEFAULT_DEVICE
    hf_home: str | None = None
    hf_hub_cache: str | None = None
    transformers_cache: str | None = None

    @classmethod
    def from_env(cls) -> "AppSettings":
        """Build settings from environment variables."""
        return cls(
            model_id=_read_str_env("APP_MODEL_ID", DEFAULT_MODEL_ID),
            max_input_chars=_read_int_env(
                "APP_MAX_INPUT_CHARS", DEFAULT_MAX_INPUT_CHARS
            ),
            max_new_tokens=_read_int_env("APP_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS),
            device=_read_device_env("APP_DEVICE", DEFAULT_DEVICE),
            hf_home=_read_optional_str_env("HF_HOME"),
            hf_hub_cache=_read_optional_str_env("HF_HUB_CACHE"),
            transformers_cache=_read_optional_str_env("TRANSFORMERS_CACHE"),
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
