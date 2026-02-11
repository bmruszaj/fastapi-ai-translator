from app.application.ports.translator_port import TranslatorPort


class FakeTranslator(TranslatorPort):
    """Deterministic translator used by tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.ready = True
        self.translate_error: Exception | None = None

    def translate(self, text: str, source: str, target: str) -> str:
        """Return fake output while tracking translation calls."""
        self.calls.append((text, source, target))
        if self.translate_error is not None:
            raise self.translate_error
        return f"{text}|{source}->{target}"

    def is_ready(self) -> bool:
        """Report whether fake translator should be treated as ready."""
        return self.ready
