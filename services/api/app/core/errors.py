from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from finsight_core.config import get_settings
from finsight_core.schemas import ErrorDetail, ProblemDetails


class AppException(Exception):
    def __init__(
        self,
        status_code: int = 400,
        code: str = "BAD_REQUEST",
        title: str = "Bad Request",
        detail: str | None = None,
        errors: list[ErrorDetail] | None = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(detail or title)
        self.status_code = status_code
        self.code = code
        self.title = title
        self.detail = detail
        self.errors = errors or []
        self.headers = headers


class NotFoundException(AppException):
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(
            status_code=404,
            code="NOT_FOUND",
            title="Not Found",
            detail=detail,
        )


class RateLimitedException(AppException):
    def __init__(self, retry_after: int, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=429,
            code="RATE_LIMITED",
            title="Too Many Requests",
            detail=detail,
            headers={"Retry-After": str(max(1, retry_after))},
        )


async def app_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    headers: dict[str, str] | None = None
    if isinstance(exc, AppException):
        code = exc.code
        title = exc.title
        status_code = exc.status_code
        detail = exc.detail
        errors = exc.errors
        headers = exc.headers
    else:
        code = "INTERNAL_ERROR"
        title = "Internal Server Error"
        status_code = 500
        detail = (
            str(exc)
            if get_settings().APP_ENV == "local"
            else "An unexpected internal error occurred."
        )
        errors = []

    problem = ProblemDetails(
        type=f"https://finsight.local/errors/{code.lower()}",
        title=title,
        status=status_code,
        detail=detail,
        instance=request.url.path,
        code=code,
        request_id=request.headers.get("X-Request-ID"),
        errors=errors,
    )
    return JSONResponse(
        status_code=status_code,
        content=problem.model_dump(),
        media_type="application/problem+json",
        headers=headers,
    )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    errors: list[ErrorDetail] = []
    if isinstance(exc, RequestValidationError):
        for err in exc.errors():
            field = ".".join(str(loc) for loc in err["loc"] if loc not in ("body", "query", "path"))
            errors.append(ErrorDetail(field=field, message=err["msg"], code=err["type"]))

    problem = ProblemDetails(
        type="https://finsight.local/errors/validation",
        title="Request validation failed",
        status=422,
        detail="One or more validation errors occurred.",
        instance=request.url.path,
        code="VALIDATION_ERROR",
        request_id=request.headers.get("X-Request-ID"),
        errors=errors,
    )
    return JSONResponse(
        status_code=422,
        content=problem.model_dump(),
        media_type="application/problem+json",
    )
