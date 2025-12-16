#!/usr/bin/env python3
"""
HMM Model Validation Module - Realized Volatility (RV)
=======================================================

Walk-forward evaluation and validation utilities for HMM-based realized variance forecasting.

Features:
- Walk-forward evaluation with ratio-cutoff filtering
- MAPE calculation on variance forecasts
- Bias diagnostics for model assessment
- Support for both Rogers-Satchell and Parkinson variance proxies

Author: FinData Sage
Purpose: Educational demonstration of model validation techniques
Disclaimer: Educational purposes only; not financial advice.
"""

import numpy as np
import pandas as pd
import argparse
import os
from typing import Dict, Tuple
from hmm_vol.model import (
    get_data, parkinson_daily_variance, 
    fit_hmm_logvar
)


def close_to_close_logret(df: pd.DataFrame) -> pd.Series:
    if "Close" in df.columns:
        return np.log(df["Close"] / df["Close"].shift(1)).dropna()
    raise ValueError("No Close column found")


def rolling_mad_abs(r: pd.Series, window: int) -> pd.Series:
    return r.abs().rolling(window=window, min_periods=1).median()


def window_parkinson_rv(dv: pd.Series, window: int) -> pd.Series:
    return np.sqrt(dv.rolling(window=window, min_periods=1).sum())


def winsorize(s: pd.Series, qlo: float, qhi: float) -> pd.Series:
    q_low = np.quantile(s.dropna(), qlo)
    q_high = np.quantile(s.dropna(), qhi)
    return s.clip(lower=q_low, upper=q_high)


def estimate_c_fixed(dv: pd.Series, r: pd.Series, window: int, qlo: float = 0.01, qhi: float = 0.99, train_frac: float = 0.70) -> float:
    n = len(dv)
    train_end = int(np.floor(train_frac * n))
    
    dv_train = dv.iloc[:train_end]
    r_train = r.iloc[:train_end]
    
    rvW = window_parkinson_rv(dv_train, window)
    madW = rolling_mad_abs(r_train, window)
    
    ratio = rvW / np.clip(madW, 1e-12, None)
    ratio_winsorized = winsorize(ratio, qlo, qhi)
    
    c_med = float(np.median(ratio_winsorized.dropna()))
    return max(1.6, c_med)


def build_mad_vol_path(df: pd.DataFrame, window: int, train_frac: float, qlo: float, qhi: float, annualize: bool) -> pd.DataFrame:
    dv = parkinson_daily_variance(df).dropna()
    r = close_to_close_logret(df).reindex_like(dv)
    
    c = estimate_c_fixed(dv, r, window, qlo, qhi, train_frac)
    
    madW = rolling_mad_abs(r, window)
    sigma_hat = c * madW
    
    if annualize:
        madW = madW * np.sqrt(252)
        sigma_hat = sigma_hat * np.sqrt(252)
    
    result = pd.DataFrame({
        'mad': madW,
        'c': c,
        'sigma_hat': sigma_hat
    }, index=dv.index)
    
    return result


def forecast_hday_var_validation(A: np.ndarray, mu: np.ndarray, tau2: np.ndarray, pT: np.ndarray, horizon: int) -> float:
    """
    Forecast H-day realized variance (sum of daily variances).
    
    Parameters:
    - A: Transition matrix
    - mu: State means on log-variance scale
    - tau2: State variances on log-variance scale
    - pT: Final state probabilities
    - horizon: Forecast horizon in days
    
    Returns:
    - float: Forecasted H-day realized variance
    """
    m = np.exp(mu + 0.5*tau2)
    pt = pT.copy()
    rv_sum = 0.0
    for _ in range(horizon):
        pt = pt @ A
        rv_sum += float(pt @ m)
    return float(rv_sum)


def walk_forward_eval(df: pd.DataFrame, horizon: int, train_frac: float, 
                     step: int, ratio_cutoff: float, states: int, min_gap: float,
                     band_mult: float = 1.25) -> Dict[str, float]:
    """
    Walk-forward evaluation of H-day realized variance forecasts with terminal return band hit-rate (iron condor proxy).
    
    This function implements a robust evaluation framework that:
    1. Performs walk-forward forecasting with specified training fraction
    2. Computes terminal log returns over forecast horizon
    3. Filters out extreme moves using ratio cutoff (q_ret = |r| / sigma_pred)
    4. Computes accuracy metrics on the "kept zone" (filtered observations)
    5. Provides bias diagnostics for model assessment
    
    Band hit-rate is defined as iron condor proxy: |r| <= band_mult * sigma_pred
    
    Parameters:
    - df: DataFrame with OHLC data (must contain 'Close' column)
    - horizon: Forecast horizon in days
    - train_frac: Training set fraction (e.g., 0.7 for 70% training)
    - step: Step size for walk-forward (0 = use horizon)
    - ratio_cutoff: Ratio cutoff for excluding extreme moves (q_ret >= ratio_cutoff excluded)
    - states: Number of HMM states
    - min_gap: Minimum gap between state means
    
    Returns:
    - dict: Evaluation metrics including:
        * n_forecasts: Total number of forecasts
        * n_kept: Number of forecasts kept after filtering
        * excluded_pct: Percentage of forecasts excluded
        * mape_vol_mid: MAPE on volatility after excluding extremes
        * med_ratio_vol_kept: Median return/volatility ratio (q_ret) on kept observations
        * mean_log_ratio_vol_kept: Mean log return/volatility ratio on kept observations
        * over_share_vol_kept: Share of over-forecasts in kept zone
        * band_mult: Band multiplier used
        * in_band_share_kept: Share of kept forecasts within band (iron condor proxy)
        * in_band_share_all: Share of all forecasts within band
    """
    # Compute daily variance using Parkinson proxy
    dv = parkinson_daily_variance(df)
    n = dv.dropna().shape[0]
    t0 = int(np.floor(train_frac * n)) - 1
    
    # Set step size
    if step <= 0:
        step = horizon
    
    idx = dv.dropna().index
    
    # Ensure Close column exists and align with dv index
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column for terminal return calculation")
    
    # Align Close prices with dv index (after dropna)
    close_aligned = df["Close"].reindex(idx)
    
    real_vs = []
    hat_vs = []
    terminal_returns = []
    
    # Walk-forward loop
    for anchor in range(t0, n - horizon, step):
        # Fit HMM on training data up to anchor point
        train_data = dv.loc[:idx[anchor]]
        logvar = np.log(train_data + 1e-12)
        res = fit_hmm_logvar(logvar, n_states=states, min_gap=min_gap)
        
        # Forecast H-day variance
        vhat = forecast_hday_var_validation(res["A"], res["mu"], res["tau2"], res["pT"], horizon)
        
        # Compute realized variance over forecast horizon (for MAPE calculation)
        vreal = float(dv.loc[idx[anchor+1]:idx[anchor+horizon]].sum())
        
        # Compute terminal log return: r = log(Close[t+H] / Close[t])
        anchor_date = idx[anchor]
        terminal_date = idx[anchor + horizon]
        
        # Check if Close prices exist at both dates
        if anchor_date in close_aligned.index and terminal_date in close_aligned.index:
            close_anchor = close_aligned.loc[anchor_date]
            close_terminal = close_aligned.loc[terminal_date]
            if pd.notna(close_anchor) and pd.notna(close_terminal) and close_anchor > 0:
                r = np.log(close_terminal / close_anchor)
            else:
                r = float("nan")
        else:
            r = float("nan")
        
        hat_vs.append(vhat)
        real_vs.append(vreal)
        terminal_returns.append(r)
    
    # Convert to numpy arrays for vectorized operations
    V_hat = np.array(hat_vs, dtype=float)
    V_real = np.array(real_vs, dtype=float)
    r_array = np.array(terminal_returns, dtype=float)
    
    # Compute predicted volatility: sigma_pred = sqrt(V_pred)
    eps = 1e-16
    sigma_pred = np.sqrt(np.clip(V_hat, 0, None))
    sigma_real = np.sqrt(np.clip(V_real, 0, None))
    
    # Compute dimensionless ratio: q_ret = |r| / sigma_pred
    q_ret = np.abs(r_array) / np.clip(sigma_pred, eps, None)
    
    # Apply ratio cutoff filter: exclude extremes where q_ret >= ratio_cutoff
    mask = q_ret < ratio_cutoff
    
    # Filter out NaN returns (missing Close data)
    valid_returns = ~np.isnan(r_array)
    mask = mask & valid_returns
    
    # Compute evaluation metrics
    mape_vol_mid = float(np.mean(np.abs(sigma_real[mask] - sigma_pred[mask]) / np.clip(sigma_real[mask], eps, None))) if np.any(mask) else float("nan")
    excluded_pct = float(1.0 - np.mean(mask))
    med_ratio_vol_kept = float(np.median(q_ret[mask])) if np.any(mask) else float("nan")
    mean_log_ratio_vol_kept = float(np.mean(np.log(np.clip(q_ret[mask], eps, None)))) if np.any(mask) else float("nan")
    over_share_vol_kept = float(np.mean(sigma_pred[mask] >= sigma_real[mask])) if np.any(mask) else float("nan")
    
    # Band coverage: iron condor proxy - in-band if |r| <= band_mult * sigma_pred
    in_band = np.abs(r_array) <= (band_mult * sigma_pred)
    # Only count valid returns
    in_band = in_band & valid_returns
    in_band_share_kept = float(np.mean(in_band[mask])) if np.any(mask) else float("nan")
    in_band_share_all = float(np.mean(in_band[valid_returns])) if np.any(valid_returns) else float("nan")
    
    return {
        'n_forecasts': int(len(V_real)),
        'n_kept': int(mask.sum()),
        'excluded_pct': excluded_pct,
        'mape_vol_mid': mape_vol_mid,
        'med_ratio_vol_kept': med_ratio_vol_kept,
        'mean_log_ratio_vol_kept': mean_log_ratio_vol_kept,
        'over_share_vol_kept': over_share_vol_kept,
        'band_mult': float(band_mult),
        'in_band_share_kept': in_band_share_kept,
        'in_band_share_all': in_band_share_all
    }


def print_validation_results(results: Dict[str, float], symbol: str, horizon: int, 
                           ratio_cutoff: float) -> None:
    """
    Print formatted validation results with interpretation.
    
    Parameters:
    - results: Dictionary of validation metrics
    - symbol: Stock symbol
    - horizon: Forecast horizon
    - ratio_cutoff: Ratio cutoff applied
    """
    print("=" * 60)
    print(f"HMM MODEL VALIDATION RESULTS (Realized Volatility)")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print("Variance Proxy: Parkinson")
    print(f"Forecast Horizon: {horizon} days")
    print(f"Ratio Cutoff: {ratio_cutoff}")
    print("-" * 60)
    
    print(f"Total Forecasts: {results['n_forecasts']}")
    print(f"Forecasts Kept: {results['n_kept']}")
    print(f"Excluded %: {results['excluded_pct']:.1%}")
    print()
    
    print("ACCURACY METRICS (Volatility):")
    print(f"MAPE (mid-zone): {results['mape_vol_mid']:.3f}")
    print(f"Median Ratio (kept): {results['med_ratio_vol_kept']:.3f}")
    print()
    
    # Interpretation
    print("INTERPRETATION:")
    if results['excluded_pct'] < 0.15:
        print("✓ Good exclusion rate - not over-filtering")
    elif results['excluded_pct'] > 0.30:
        print("⚠ High exclusion rate - may be over-filtering")
    else:
        print("~ Moderate exclusion rate")
    
    if results['mape_vol_mid'] < 0.5:
        print("✓ Good accuracy on mid-zone volatility forecasts")
    elif results['mape_vol_mid'] > 0.8:
        print("⚠ High error rate on mid-zone volatility forecasts")
    else:
        print("~ Moderate accuracy on mid-zone volatility forecasts")
    
    print("=" * 60)


def main():
    """Main function for HMM model validation (RV)."""
    p = argparse.ArgumentParser(description="HMM Model Validation - Walk-forward evaluation (RV)")
    p.add_argument("--symbol", type=str, default="AMZN", help="Stock symbol")
    p.add_argument("--years", type=int, default=10, help="Years of historical data")
    p.add_argument("--horizon", type=int, default=60, help="Forecast horizon in days")
    p.add_argument("--states", type=int, default=2, help="Number of HMM states")
    p.add_argument("--min_gap", type=float, default=0.10, help="Minimum gap between state means")
    p.add_argument("--train_frac", type=float, default=0.70, 
                   help="Training set fraction (0.7 = 70%% training)")
    p.add_argument("--step", type=int, default=0, 
                   help="Step size for walk-forward (0 = use horizon)")
    p.add_argument("--ratio_cutoff", type=float, default=2.2, 
                   help="Ratio cutoff for excluding extreme under-predictions")
    p.add_argument("--band_mult", type=float, default=1.25,
                   help="Band multiplier for coverage evaluation (default: 1.25)")
    p.add_argument("--quiet", action="store_true", help="Quiet mode - minimal output")
    
    p.add_argument("--mad-mode", action="store_true", help="Compute and save robust MAD vol path")
    p.add_argument("--mad-window", type=int, default=20, help="MAD rolling window size")
    p.add_argument("--winsor-lo", type=float, default=0.01, help="Winsorization lower quantile")
    p.add_argument("--winsor-hi", type=float, default=0.99, help="Winsorization upper quantile")
    p.add_argument("--annualize", action="store_true", help="Annualize MAD and sigma_hat")
    
    args = p.parse_args()
    
    try:
        # Fetch and prepare data
        if not args.quiet:
            print(f"Fetching {args.years} years of data for {args.symbol}...")
        
        df = get_data(args.symbol, args.years)
        
        # MAD mode: compute and save robust vol path
        if args.mad_mode:
            mad_result = build_mad_vol_path(
                df=df,
                window=args.mad_window,
                train_frac=args.train_frac,
                qlo=args.winsor_lo,
                qhi=args.winsor_hi,
                annualize=args.annualize
            )
            
            # Ensure output directory exists
            os.makedirs("outputs/series", exist_ok=True)
            
            # Save to CSV
            output_file = f"outputs/series/mad_vol_{args.symbol}_W{args.mad_window}.csv"
            mad_result.to_csv(output_file)
            
            # Print summary
            print(f"c_fixed: {mad_result['c'].iloc[0]:.6f}")
            print(f"mad_window: {args.mad_window}")
            print(f"annualized: {1 if args.annualize else 0}")
            print(f"first_date: {mad_result.index[0]}")
            print(f"last_date: {mad_result.index[-1]}")
            print(f"n_obs: {len(mad_result)}")
            print(f"\nSample values (first 5 rows):")
            print(mad_result.head())
            print(f"\nSaved to: {output_file}")
            
            return 0
        
        # Existing HMM RV validation path
        if not args.quiet:
            print(f"Running walk-forward evaluation...")
            print(f"Training fraction: {args.train_frac:.1%}")
            print(f"Step size: {args.step if args.step > 0 else args.horizon}")
        
        # Run validation
        results = walk_forward_eval(
            df=df,
            horizon=args.horizon,
            train_frac=args.train_frac,
            step=args.step,
            ratio_cutoff=args.ratio_cutoff,
            states=args.states,
            min_gap=args.min_gap
        )
        
        # Print results
        if args.quiet:
            # Minimal output for scripting
            print(f"n_forecasts: {results['n_forecasts']}")
            print(f"n_kept: {results['n_kept']}")
            print(f"excluded_pct: {results['excluded_pct']:.6f}")
            print(f"mape_vol_mid: {results['mape_vol_mid']:.6f}")
            print(f"med_ratio_vol_kept: {results['med_ratio_vol_kept']:.6f}")
            print(f"mean_log_ratio_vol_kept: {results['mean_log_ratio_vol_kept']:.6f}")
            print(f"over_share_vol_kept: {results['over_share_vol_kept']:.6f}")
            print(f"band_mult: {results['band_mult']:.6f}")
            print(f"in_band_share_kept: {results['in_band_share_kept']:.6f}")
            print(f"in_band_share_all: {results['in_band_share_all']:.6f}")
        else:
            # Full formatted output
            print_validation_results(results, args.symbol, args.horizon, args.ratio_cutoff)
            
    except Exception as e:
        print(f"Error: {e}")
        print("Try a different symbol, adjust parameters, or wait for rate limits to reset")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

