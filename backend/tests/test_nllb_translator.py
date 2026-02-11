from collections.abc import Callable

import pytest

from app.adapters.outbound import nllb_translator
from app.application.ports.errors import (
    TranslationExecutionError,
    TranslatorUnavailableError,
)
from tests.fakes_nllb import FakeConfig, FakeModel, FakeTensor, FakeTokenizer


def test_init_raises_unavailable_on_tokenizer_load_failure(
    get_nllb_runtime_dependencies: Callable[
        ..., nllb_translator.NllbRuntimeDependencies
    ],
) -> None:
    # Given
    runtime_dependencies = get_nllb_runtime_dependencies(
        tokenizer_error=RuntimeError("load failed")
    )

    # When / Then
    with pytest.raises(
        TranslatorUnavailableError, match="Unable to initialize translator backend"
    ):
        nllb_translator.NllbTranslator(
            model_id="fake-model",
            runtime_dependencies=runtime_dependencies,
        )


def test_init_passes_cache_dir_to_loaders(
    fake_tokenizer_cls: type[FakeTokenizer],
    fake_model_cls: type[FakeModel],
    fake_config_cls: type[FakeConfig],
    get_nllb_runtime_dependencies: Callable[
        ..., nllb_translator.NllbRuntimeDependencies
    ],
) -> None:
    # Given
    tokenizer = fake_tokenizer_cls()
    model = fake_model_cls()
    model_config = fake_config_cls()
    tokenizer_call: dict[str, object] = {}
    config_call: dict[str, object] = {}
    model_call: dict[str, object] = {}
    runtime_dependencies = get_nllb_runtime_dependencies(
        tokenizer=tokenizer,
        model=model,
        config=model_config,
        tokenizer_call=tokenizer_call,
        config_call=config_call,
        model_call=model_call,
    )

    # When
    _ = nllb_translator.NllbTranslator(
        model_id="fake-model",
        cache_dir="/tmp/cache",
        runtime_dependencies=runtime_dependencies,
    )

    # Then
    assert tokenizer_call == {
        "model_id": "fake-model",
        "kwargs": {"cache_dir": "/tmp/cache"},
    }
    assert config_call == {
        "model_id": "fake-model",
        "kwargs": {"cache_dir": "/tmp/cache"},
    }
    assert model_call == {
        "model_id": "fake-model",
        "kwargs": {
            "config": model_config,
            "cache_dir": "/tmp/cache",
        },
    }
    assert model_config.tie_word_embeddings is False
    assert model.eval_called is True


def test_translate_raises_for_invalid_forced_bos_token(
    fake_tokenizer_cls: type[FakeTokenizer],
    fake_model_cls: type[FakeModel],
    get_nllb_runtime_dependencies: Callable[
        ..., nllb_translator.NllbRuntimeDependencies
    ],
) -> None:
    # Given
    tokenizer = fake_tokenizer_cls(lang_code_to_id=None, converted_token_id=None)
    model = fake_model_cls()
    runtime_dependencies = get_nllb_runtime_dependencies(
        tokenizer=tokenizer, model=model
    )
    translator = nllb_translator.NllbTranslator(
        model_id="fake-model",
        runtime_dependencies=runtime_dependencies,
    )

    # When / Then
    with pytest.raises(
        TranslationExecutionError,
        match="does not provide a valid target language token",
    ):
        translator.translate(text="hello", source="en", target="fr")


def test_translate_raises_for_unknown_target_token(
    fake_tokenizer_cls: type[FakeTokenizer],
    fake_model_cls: type[FakeModel],
    get_nllb_runtime_dependencies: Callable[
        ..., nllb_translator.NllbRuntimeDependencies
    ],
) -> None:
    # Given
    tokenizer = fake_tokenizer_cls(
        lang_code_to_id={"fra_Latn": 0},
        converted_token_id=None,
        unk_token_id=0,
    )
    model = fake_model_cls()
    runtime_dependencies = get_nllb_runtime_dependencies(
        tokenizer=tokenizer, model=model
    )
    translator = nllb_translator.NllbTranslator(
        model_id="fake-model",
        runtime_dependencies=runtime_dependencies,
    )

    # When / Then
    with pytest.raises(
        TranslationExecutionError,
        match="resolved target language 'fra_Latn' to unknown token",
    ):
        translator.translate(text="hello", source="en", target="fr")


def test_translate_returns_decoded_text_for_valid_pair(
    fake_tensor_cls: type[FakeTensor],
    fake_tokenizer_cls: type[FakeTokenizer],
    fake_model_cls: type[FakeModel],
    get_nllb_runtime_dependencies: Callable[
        ..., nllb_translator.NllbRuntimeDependencies
    ],
) -> None:
    # Given
    tokenizer = fake_tokenizer_cls(
        lang_code_to_id={"fra_Latn": 17},
        decoded_text="  bonjour  ",
    )
    model = fake_model_cls(generated_tokens=[[7, 8, 9]])
    runtime_dependencies = get_nllb_runtime_dependencies(
        tokenizer=tokenizer, model=model
    )
    translator = nllb_translator.NllbTranslator(
        model_id="fake-model",
        max_new_tokens=77,
        runtime_dependencies=runtime_dependencies,
    )

    # When
    translated_text = translator.translate(text="hello", source="en", target="fr")

    # Then
    assert translated_text == "bonjour"
    assert tokenizer.src_lang == "eng_Latn"
    assert tokenizer.convert_calls == []
    assert len(model.generate_calls) == 1
    generate_kwargs = model.generate_calls[0]
    assert isinstance(generate_kwargs["input_ids"], fake_tensor_cls)
    assert isinstance(generate_kwargs["attention_mask"], fake_tensor_cls)
    assert str(generate_kwargs["input_ids"].moved_to_device) == "cpu"
    assert str(generate_kwargs["attention_mask"].moved_to_device) == "cpu"
    assert generate_kwargs["forced_bos_token_id"] == 17
    assert generate_kwargs["max_new_tokens"] == 77


def test_translate_raises_for_model_runtime_error(
    fake_tokenizer_cls: type[FakeTokenizer],
    fake_model_cls: type[FakeModel],
    get_nllb_runtime_dependencies: Callable[
        ..., nllb_translator.NllbRuntimeDependencies
    ],
) -> None:
    # Given
    tokenizer = fake_tokenizer_cls(lang_code_to_id={"fra_Latn": 17})
    model = fake_model_cls(generate_error=RuntimeError("boom"))
    runtime_dependencies = get_nllb_runtime_dependencies(
        tokenizer=tokenizer, model=model
    )
    translator = nllb_translator.NllbTranslator(
        model_id="fake-model",
        runtime_dependencies=runtime_dependencies,
    )

    # When / Then
    with pytest.raises(
        TranslationExecutionError,
        match="Translation execution failed for language pair 'en' -> 'fr'",
    ):
        translator.translate(text="hello", source="en", target="fr")
