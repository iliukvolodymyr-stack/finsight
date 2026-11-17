from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/predictions", tags=["Predictions"])


class PriceRange(BaseModel):
    low: float
    high: float
    coverage: float = 0.8


class ModelInfo(BaseModel):
    name: str = "meta_ensemble"
    version: str = "1.0.0"


class ExplanationItem(BaseModel):
    feature: str
    contribution: float


class PredictionResponse(BaseModel):
    instrument_id: str
    symbol: str
    timeframe: str
    as_of: str
    horizon_bars: int
    signal: str
    confidence: float
    probabilities: dict[str, float]
    price_range: PriceRange
    model: ModelInfo
    regime: str = "bull"
    explanation: list[ExplanationItem] = []


@router.get("", response_model=list[PredictionResponse])
async def list_predictions(
    instrument_ids: list[str] | None = Query(None),
    timeframe: str = Query("1h"),
    latest: bool = Query(True),
) -> list[PredictionResponse]:
    return [
        PredictionResponse(
            instrument_id="7f0c0000-0000-7000-8000-000000000001",
            symbol="BTCUSDT",
            timeframe="1h",
            as_of="2026-11-17T14:00:00Z",
            horizon_bars=24,
            signal="BUY",
            confidence=0.73,
            probabilities={"buy": 0.73, "sell": 0.11, "hold": 0.16},
            price_range=PriceRange(low=90200.0, high=93800.0, coverage=0.8),
            model=ModelInfo(name="lgbm_baseline", version="0.1.0"),
            regime="bull",
            explanation=[
                ExplanationItem(feature="rsi14", contribution=0.18),
                ExplanationItem(feature="ema_ratio_20_50", contribution=0.12),
            ],
        )
    ]


@router.get("/{id}", response_model=PredictionResponse)
async def get_prediction(id: str) -> PredictionResponse:
    return PredictionResponse(
        instrument_id="7f0c0000-0000-7000-8000-000000000001",
        symbol="BTCUSDT",
        timeframe="1h",
        as_of="2026-11-17T14:00:00Z",
        horizon_bars=24,
        signal="BUY",
        confidence=0.73,
        probabilities={"buy": 0.73, "sell": 0.11, "hold": 0.16},
        price_range=PriceRange(low=90200.0, high=93800.0, coverage=0.8),
        model=ModelInfo(name="lgbm_baseline", version="0.1.0"),
        regime="bull",
        explanation=[],
    )
