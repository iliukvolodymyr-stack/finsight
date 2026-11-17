from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecaster":
        """Fit the model on training data."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities: [P(SELL), P(HOLD), P(BUY)]."""
        pass

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Save model artifacts to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> "BaseForecaster":
        """Load model artifacts from disk."""
        pass
