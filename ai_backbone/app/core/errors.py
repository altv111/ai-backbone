from typing import Any, Optional

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.contracts.common import ErrorDetail, ErrorResponse
from app.core.request_context import get_request_id


class AppError(Exception):
    code = "internal_error"
    status_code = 500

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ProviderNotFoundError(AppError):
    code = "provider_not_found"
    status_code = 404


class UnsupportedOperationError(AppError):
    code = "unsupported_operation"
    status_code = 400


class ProviderBusyError(AppError):
    code = "provider_busy"
    status_code = 429


class ProviderTimeoutError(AppError):
    code = "provider_timeout"
    status_code = 504


class ProviderUnavailableError(AppError):
    code = "provider_unavailable"
    status_code = 503


class InvalidProviderResponseError(AppError):
    code = "invalid_provider_response"
    status_code = 502


class InvalidRequestError(AppError):
    code = "invalid_request"
    status_code = 400


class ProviderFailedError(AppError):
    code = "provider_failed"
    status_code = 502


def to_error_response(code: str, message: str, details: Optional[dict[str, Any]] = None) -> ErrorResponse:
    return ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            details=details or {},
            request_id=get_request_id(),
        )
    )


async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    payload = to_error_response(exc.code, exc.message, exc.details)
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


async def request_validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    payload = to_error_response(
        "invalid_request",
        "Request payload failed validation",
        {"errors": exc.errors()},
    )
    return JSONResponse(status_code=400, content=payload.model_dump())


def register_exception_handlers(app) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, request_validation_error_handler)
