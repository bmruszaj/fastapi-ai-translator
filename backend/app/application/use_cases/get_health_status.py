from collections.abc import Callable

from app.application.dto import HealthStatus


class GetHealthStatusUseCase:
    """Build health response data from application runtime state."""

    def __init__(self, model_id: str, is_loaded: Callable[[], bool]) -> None:
        self._model_id = model_id
        self._is_loaded = is_loaded

    def execute(self) -> HealthStatus:
        """Return current health status payload."""
        return HealthStatus(
            status="ok",
            model=self._model_id,
            loaded=self._is_loaded(),
        )
