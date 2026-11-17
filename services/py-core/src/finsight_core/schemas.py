from datetime import datetime
from typing import Any, Literal

import uuid_utils as uuid
from pydantic import BaseModel, Field


class CandleDTO(BaseModel):
    t: datetime
    o: float
    h: float
    l: float
    c: float
    v: float
    closed: bool = True


class InstrumentDTO(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    symbol: str
    name: str
    asset_class: Literal["crypto", "stock", "etf", "index"]
    exchange: str
    quote_currency: str = "USDT"
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ErrorDetail(BaseModel):
    field: str | None = None
    message: str
    code: str | None = None


class ProblemDetails(BaseModel):
    type: str = "https://finsight.local/errors/general"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
    code: str = "INTERNAL_ERROR"
    request_id: str | None = None
    errors: list[ErrorDetail] = Field(default_factory=list)
