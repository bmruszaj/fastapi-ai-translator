import logging
import threading

import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from app.application.ports.errors import (
    TranslationExecutionError,
    TranslatorUnavailableError,
)
from app.application.ports.translator_port import TranslatorPort

logger = logging.getLogger("uvicorn.error")

ISO_TO_NLLB: dict[str, str] = {
    "de": "deu_Latn",
    "en": "eng_Latn",
    "el": "ell_Grek",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "nl": "nld_Latn",
}


def _to_nllb_language_code(language_code: str) -> str:
    """Convert application language code into NLLB language code."""
    nllb_code = ISO_TO_NLLB.get(language_code)
    if nllb_code is None:
        raise TranslationExecutionError(
            f"Language '{language_code}' is not supported by the NLLB translator."
        )
    return nllb_code


def _resolve_forced_bos_token_id(
    tokenizer: PreTrainedTokenizerBase, target_language_code: str, model_id: str
) -> int:
    """Resolve and validate forced BOS token ID for the target language."""
    language_code_mapping = getattr(tokenizer, "lang_code_to_id", None)
    forced_bos_token_id: int | None = None
    if isinstance(language_code_mapping, dict):
        mapped_id = language_code_mapping.get(target_language_code)
        if isinstance(mapped_id, int):
            forced_bos_token_id = mapped_id

    if forced_bos_token_id is None:
        converted_id = tokenizer.convert_tokens_to_ids(target_language_code)
        if isinstance(converted_id, int):
            forced_bos_token_id = converted_id

    if forced_bos_token_id is None or forced_bos_token_id < 0:
        raise TranslationExecutionError(
            f"Tokenizer for model '{model_id}' does not provide a valid target language token for '{target_language_code}'."
        )

    unknown_token_id = tokenizer.unk_token_id
    if unknown_token_id is not None and forced_bos_token_id == unknown_token_id:
        raise TranslationExecutionError(
            f"Tokenizer for model '{model_id}' resolved target language '{target_language_code}' to unknown token."
        )

    return forced_bos_token_id


def _validate_runtime_device(device: torch.device) -> None:
    """Validate device availability before loading large model artifacts."""
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise TranslatorUnavailableError(
                "APP_DEVICE is set to CUDA but CUDA is not available on this machine."
            )
        if device.index is None:
            return

        device_count = torch.cuda.device_count()
        if device.index < 0 or device.index >= device_count:
            raise TranslatorUnavailableError(
                f"APP_DEVICE points to cuda:{device.index}, but available CUDA devices are in range 0..{device_count - 1}."
            )
        return

    if device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise TranslatorUnavailableError(
                "APP_DEVICE is set to mps but MPS is not available on this machine."
            )


class NllbTranslator(TranslatorPort):
    """Translate text with a startup-loaded NLLB model."""

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 256,
        device: str = "cpu",
        cache_dir: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._lock = threading.Lock()
        try:
            self._device = torch.device(device)
            _validate_runtime_device(self._device)
            load_kwargs: dict[str, str] = {}
            if cache_dir:
                load_kwargs["cache_dir"] = cache_dir
            logger.info(
                "Loading tokenizer for model '%s'. If not cached, files may be downloaded from Hugging Face.",
                model_id,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
            model_config = AutoConfig.from_pretrained(model_id, **load_kwargs)
            if getattr(model_config, "tie_word_embeddings", None) is not False:
                # Keep embeddings untied to match this checkpoint layout and silence tied-weight warnings.
                model_config.tie_word_embeddings = False
            logger.info("Loading model weights for '%s'.", model_id)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, config=model_config, **load_kwargs
            )
            self._model.to(self._device)
            self._model.eval()
            logger.info(
                "Model '%s' loaded successfully on device '%s'.", model_id, self._device
            )
        except TranslatorUnavailableError:
            raise
        except Exception as error:
            raise TranslatorUnavailableError(
                f"Unable to initialize translator backend for model '{model_id}'."
            ) from error

    def translate(self, text: str, source: str, target: str) -> str:
        """Translate text from source language to target language."""
        try:
            with self._lock:
                model_source = _to_nllb_language_code(source)
                model_target = _to_nllb_language_code(target)
                self._tokenizer.src_lang = model_source
                tokenized_input = self._tokenizer(text, return_tensors="pt")
                tokenized_input = {
                    name: values.to(self._device)
                    for name, values in tokenized_input.items()
                }
                forced_bos_token_id = _resolve_forced_bos_token_id(
                    tokenizer=self._tokenizer,
                    target_language_code=model_target,
                    model_id=self._model_id,
                )

                with torch.inference_mode():
                    generated_tokens = self._model.generate(
                        **tokenized_input,
                        forced_bos_token_id=forced_bos_token_id,
                        max_new_tokens=self._max_new_tokens,
                    )
                decoded_text = self._tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
        except TranslationExecutionError:
            raise
        except Exception as error:
            raise TranslationExecutionError(
                f"Translation execution failed for language pair '{source}' -> '{target}'."
            ) from error

        if not decoded_text:
            return ""
        return decoded_text[0].strip()
