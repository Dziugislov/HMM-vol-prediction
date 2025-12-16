"""
Validation Module
=================

Comprehensive validation suite for HMM volatility forecasting:
- Realized volatility forecast validation (volatility.py)
- Regime classification validation (regime.py)

Author: FinData Sage
Purpose: Educational demonstration of model validation techniques
Disclaimer: Educational purposes only; not financial advice.
"""

from .volatility import walk_forward_eval, print_validation_results
from .regime import walk_forward_regime_eval

__all__ = [
    "walk_forward_eval",
    "print_validation_results",
    "walk_forward_regime_eval"
]

