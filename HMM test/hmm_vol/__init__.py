"""
HMM Volatility Forecasting - Core Module
=========================================

Core functionality for HMM-based volatility forecasting including:
- Data fetching from IBKR
- HMM model fitting on log-variance
- Volatility forecasting
"""

from .model import (
    get_data,
    parkinson_daily_variance,
    fit_hmm_logvar,
    forecast_hday_vol,
    state_contributions
)

from .data import fetch_stock_data, fetch_multiple_stocks

__version__ = "0.1.0"
__all__ = [
    "get_data",
    "parkinson_daily_variance", 
    "fit_hmm_logvar",
    "forecast_hday_vol",
    "state_contributions",
    "fetch_stock_data",
    "fetch_multiple_stocks"
]

