from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class FakeTensor:
    """Tensor-like object used to avoid real torch tensors in tests."""

    moved_to_device: Any = None

    def to(self, device: Any) -> "FakeTensor":
        self.moved_to_device = device
        return self


@dataclass(slots=True)
class FakeConfig:
    """Config stub for adapter tests."""

    tie_word_embeddings: bool = True


class FakeTokenizer:
    """Tokenizer stub for adapter tests."""

    def __init__(
        self,
        *,
        lang_code_to_id: dict[str, int] | None = None,
        converted_token_id: int | None = 101,
        decoded_text: str = "translated",
        unk_token_id: int | None = 0,
    ) -> None:
        self.lang_code_to_id = lang_code_to_id
        self._converted_token_id = converted_token_id
        self._decoded_text = decoded_text
        self.unk_token_id = unk_token_id
        self.src_lang: str | None = None
        self.convert_calls: list[str] = []

    def __call__(self, text: str, return_tensors: str) -> dict[str, FakeTensor]:
        _ = text
        _ = return_tensors
        return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

    def convert_tokens_to_ids(self, token: str) -> int | None:
        self.convert_calls.append(token)
        return self._converted_token_id

    def batch_decode(
        self, generated_tokens: list[list[int]], skip_special_tokens: bool = True
    ) -> list[str]:
        _ = generated_tokens
        _ = skip_special_tokens
        return [self._decoded_text]


class FakeModel:
    """Model stub for adapter tests."""

    def __init__(
        self,
        *,
        generated_tokens: list[list[int]] | None = None,
        generate_error: Exception | None = None,
    ) -> None:
        self.generated_tokens = generated_tokens or [[1, 2, 3]]
        self.generate_error = generate_error
        self.moved_to_device: Any = None
        self.eval_called = False
        self.generate_calls: list[dict[str, Any]] = []

    def to(self, device: Any) -> "FakeModel":
        self.moved_to_device = device
        return self

    def eval(self) -> None:
        self.eval_called = True

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.generate_calls.append(kwargs)
        if self.generate_error is not None:
            raise self.generate_error
        return self.generated_tokens
