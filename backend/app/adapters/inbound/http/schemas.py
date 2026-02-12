from pydantic import BaseModel, ConfigDict, Field


class TranslateRequest(BaseModel):
    """Request payload for text translation."""

    model_config = ConfigDict(strict=True, extra="forbid")

    text: str = Field(...)
    source_language: str = Field(...)
    target_language: str = Field(...)


class TranslateResponse(BaseModel):
    """Successful translation response payload."""

    translated_text: str
    source_language: str
    target_language: str
    model: str


class LanguagesResponse(BaseModel):
    """Response payload with supported language codes."""

    languages: list[str]


class FrontendConfigResponse(BaseModel):
    """Frontend runtime configuration payload."""

    max_input_tokens: int
    max_chars_per_token: int
    max_input_chars: int
    warning_input_chars: int


class HealthResponse(BaseModel):
    """Health endpoint response payload."""

    status: str
    model: str
    loaded: bool


class ErrorResponse(BaseModel):
    """Error response payload."""

    code: str
    message: str
