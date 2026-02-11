class ContainerAccessError(RuntimeError):
    """Base error for container access failures in dependency resolution."""

    code = "container_access_error"


class ContainerNotInitializedError(ContainerAccessError):
    """Raised when application container is missing in FastAPI state."""

    code = "container_not_initialized"

    def __init__(self) -> None:
        super().__init__("Application container is not initialized.")


class InvalidContainerTypeError(ContainerAccessError):
    """Raised when application container has unexpected runtime type."""

    code = "invalid_container_type"

    def __init__(self, actual_type: str) -> None:
        super().__init__(f"Application container has invalid type: {actual_type}.")
