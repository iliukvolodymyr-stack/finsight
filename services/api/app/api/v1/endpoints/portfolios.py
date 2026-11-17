from app.core.deps import CurrentUser, get_current_user
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

router = APIRouter(prefix="/portfolios", tags=["Portfolios"])


class PortfolioResponse(BaseModel):
    id: str
    name: str
    base_currency: str = "USD"
    created_at: str


class CreatePortfolioRequest(BaseModel):
    name: str
    base_currency: str = "USD"


class RiskSnapshotResponse(BaseModel):
    var_95: float
    cvar_95: float
    sharpe: float
    sortino: float
    max_drawdown: float
    annual_volatility: float
    risk_level: str


@router.get("", response_model=list[PortfolioResponse])
async def list_portfolios(
    current_user: CurrentUser = Depends(get_current_user),
) -> list[PortfolioResponse]:
    return [
        PortfolioResponse(
            id="01933c00-0000-7000-8000-000000000001",
            name="Main Crypto Portfolio",
            base_currency="USD",
            created_at="2026-11-17T12:00:00Z",
        )
    ]


@router.post("", response_model=PortfolioResponse, status_code=status.HTTP_201_CREATED)
async def create_portfolio(
    req: CreatePortfolioRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> PortfolioResponse:
    return PortfolioResponse(
        id="01933c00-0000-7000-8000-000000000002",
        name=req.name,
        base_currency=req.base_currency,
        created_at="2026-11-17T14:00:00Z",
    )


@router.get("/{id}/risk", response_model=RiskSnapshotResponse)
async def get_portfolio_risk(
    id: str,
    current_user: CurrentUser = Depends(get_current_user),
) -> RiskSnapshotResponse:
    return RiskSnapshotResponse(
        var_95=0.042,
        cvar_95=0.058,
        sharpe=1.65,
        sortino=2.10,
        max_drawdown=0.142,
        annual_volatility=0.28,
        risk_level="medium",
    )
