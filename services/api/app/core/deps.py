from app.core.errors import AppException
from app.infra.db import get_db_session
from fastapi import Header
from finsight_core.config import get_settings

__all__ = ["CurrentUser", "get_current_user", "get_db_session"]

settings = get_settings()


class CurrentUser:
    def __init__(self, id: str, email: str, role: str = "user"):
        self.id = id
        self.email = email
        self.role = role


async def get_current_user(
    authorization: str | None = Header(default=None),
) -> CurrentUser:
    if settings.AUTH_MODE == "single_user":
        # In single_user mode, return default demo user
        return CurrentUser(
            id="00000000-0000-7000-8000-000000000001",
            email="demo@finsight.local",
            role="admin",
        )

    if not authorization or not authorization.startswith("Bearer "):
        raise AppException(
            status_code=401,
            code="UNAUTHORIZED",
            title="Unauthorized",
            detail="Missing or invalid Authorization header",
        )

    _token = authorization.split(" ", 1)[1]
    # JWT decoding will be expanded in diploma phase
    return CurrentUser(
        id="00000000-0000-7000-8000-000000000001",
        email="demo@finsight.local",
        role="user",
    )
