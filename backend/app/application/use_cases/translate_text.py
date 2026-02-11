from app.application.dto import TranslateCommand, TranslateResult
from app.application.ports.translator_port import TranslatorPort
from app.domain.errors import EmptyTextError, TextTooLongError
from app.domain.language_rules import validate_pair


class TranslateTextUseCase:
    """Validate and translate text between supported languages."""

    def __init__(
        self, translator: TranslatorPort, model_id: str, max_input_chars: int
    ) -> None:
        self._translator = translator
        self._model_id = model_id
        self._max_input_chars = max_input_chars

    def execute(self, command: TranslateCommand) -> TranslateResult:
        """Execute text translation flow."""
        normalized_text = command.text.strip()
        if not normalized_text:
            raise EmptyTextError()
        if len(normalized_text) > self._max_input_chars:
            raise TextTooLongError(
                max_chars=self._max_input_chars, actual_chars=len(normalized_text)
            )

        normalized_source, normalized_target = validate_pair(
            source_language=command.source_language,
            target_language=command.target_language,
        )
        translated_text = self._translator.translate(
            text=normalized_text,
            source=normalized_source,
            target=normalized_target,
        )

        return TranslateResult(
            translated_text=translated_text,
            source_language=normalized_source,
            target_language=normalized_target,
            model=self._model_id,
        )
