from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.backtests import router as backtests_router
from app.api.v1.endpoints.candles import router as candles_router
from app.api.v1.endpoints.instruments import router as instruments_router
from app.api.v1.endpoints.portfolios import router as portfolios_router
from app.api.v1.endpoints.predictions import router as predictions_router
from fastapi import APIRouter

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(auth_router)
api_v1_router.include_router(instruments_router)
api_v1_router.include_router(candles_router)
api_v1_router.include_router(predictions_router)
api_v1_router.include_router(portfolios_router)
api_v1_router.include_router(backtests_router)
