from fastapi import Depends, Request

from app.application.use_cases.get_health_status import GetHealthStatusUseCase
from app.application.use_cases.translate_text import TranslateTextUseCase
from app.bootstrap.container import AppContainer
from app.core.errors import InvalidContainerTypeError, ContainerNotInitializedError


def get_container(request: Request) -> AppContainer:
    """Return the application container stored in FastAPI state."""
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise ContainerNotInitializedError()
    if not isinstance(container, AppContainer):
        raise InvalidContainerTypeError(actual_type=type(container).__name__)
    return container


def get_translate_use_case(
    container: AppContainer = Depends(get_container),
) -> TranslateTextUseCase:
    """Return the translation use case from the application container."""
    return container.translate_text_use_case


def get_health_status_use_case(
    container: AppContainer = Depends(get_container),
) -> GetHealthStatusUseCase:
    """Return the health status use case from the application container."""
    return container.get_health_status_use_case
