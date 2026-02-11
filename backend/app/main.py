from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.adapters.inbound.http.schemas import ErrorResponse
from app.adapters.inbound.http.routes import router
from app.bootstrap.container import AppContainer, build_container, cleanup_container
from app.core.config import AppSettings
from app.core.errors import ContainerAccessError

ContainerFactory = Callable[[AppSettings], AppContainer]
logger = logging.getLogger("uvicorn.error")


def create_app(
    settings: AppSettings | None = None,
    container_factory: ContainerFactory | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    resolved_settings = settings or AppSettings.from_env()
    resolved_container_factory = container_factory or build_container

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "Application startup initiated (model_id=%s, device=%s).",
            resolved_settings.model_id,
            resolved_settings.device,
        )
        container = resolved_container_factory(resolved_settings)
        app_instance.state.container = container
        logger.info("Application startup complete. Container initialized successfully.")
        try:
            yield
        finally:
            logger.info("Application shutdown initiated.")
            await cleanup_container(container)
            logger.info("Application shutdown complete.")

    app_instance = FastAPI(title="Translation Service", lifespan=lifespan)

    @app_instance.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        """Return a consistent API error payload for request validation failures."""
        error_response = ErrorResponse(
            code="validation_error",
            message="Request payload validation failed.",
        )
        return JSONResponse(status_code=422, content=error_response.model_dump())

    @app_instance.exception_handler(ContainerAccessError)
    async def handle_container_access_error(
        _request: Request, error: ContainerAccessError
    ) -> JSONResponse:
        """Return a consistent API error payload for container access failures."""
        logger.error(
            "Container access error during dependency resolution.",
            exc_info=error,
        )
        error_response = ErrorResponse(code=error.code, message=str(error))
        return JSONResponse(status_code=500, content=error_response.model_dump())

    app_instance.include_router(router)
    return app_instance


app = create_app()
