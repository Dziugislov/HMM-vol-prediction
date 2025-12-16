#!/usr/bin/env python3
"""
HMM Regime Prediction Validation with Sanity Checks
====================================================

Walk-forward evaluation of HMM regime classification with no-peeking constraint
plus comprehensive sanity checks to detect leakage, class imbalance, and overconfidence.

Evaluates:
- Hard accuracy: mean(argmax(p_pred) == viterbi_label)
- Avg p(true state): mean(p_pred[viterbi_label])
- Log loss: -mean(log p_pred[viterbi_label])
- Geometric mean p(true): exp(-logloss)

Plus sanity checks:
- Base rate benchmark
- Boundary/low-confidence performance
- Edge-window (transition) performance  
- Calibration bins (reliability)
- Confusion matrix
- Observable proxy check (vs Parkinson percentile)
- Shuffle test

Author: FinData Sage
Purpose: Educational demonstration of regime classification validation
Disclaimer: Educational purposes only; not financial advice.
"""

import numpy as np
import pandas as pd
import argparse
import csv
import warnings
from typing import Dict, List, Tuple, Optional
from hmm_vol.model import get_data, parkinson_daily_variance, fit_hmm_logvar

# Suppress RuntimeWarning about module imports
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*found in sys.modules.*')

EPS = 1e-12


def _regime_metrics(p_list: List[np.ndarray], s_list: List[int]) -> Dict[str, float]:
    """
    Compute regime classification metrics.
    
    Parameters:
    - p_list: List of probability vectors (predictions)
    - s_list: List of true state labels (ground truth from Viterbi)
    
    Returns:
    - dict: Regime classification metrics
    """
    # Extract p(true state) for each prediction
    ptrue = np.array([max(EPS, float(p[s])) for p, s in zip(p_list, s_list)], dtype=float)
    
    # Hard classification accuracy
    hard = np.array([int(np.argmax(p) == s) for p, s in zip(p_list, s_list)], dtype=float)
    
    # Log loss
    logloss = -np.mean(np.log(ptrue))
    
    return {
        "reg_hard_acc": float(np.mean(hard)),
        "reg_ptrue_avg": float(np.mean(ptrue)),
        "reg_logloss": float(logloss),
        "reg_ptrue_geom": float(np.exp(-logloss)),
        "n_points": int(len(ptrue))
    }


def _compute_sanity_checks(
    p_list: List[np.ndarray],
    s_list: List[int],
    high_state_idx: int,
    dv_series: pd.Series,
    eval_indices: List[int],
    calib_bins: int,
    edge_k: int,
    conf_thresholds: List[float],
    proxy_hi_pct: float,
    shuffle_test: bool,
    train_frac: float
) -> Dict[str, float]:
    """
    Compute comprehensive sanity checks.
    
    Returns dict with all sanity metrics.
    """
    results = {}
    
    # Convert to arrays
    p_arr = np.array([p for p in p_list], dtype=float)  # shape: (n, n_states)
    s_arr = np.array(s_list, dtype=int)
    n = len(s_arr)
    
    if n == 0:
        return {k: float("nan") for k in [
            "base_rate", "acc_lift", "acc_boundary", "ptrue_boundary", "n_boundary",
            "acc_medconf", "ptrue_medconf", "n_medconf", "acc_edge", "ptrue_edge", "n_edge",
            "acc_nonedge", "ptrue_nonedge", "n_nonedge", "calib_mae", "precision_high",
            "recall_high", "f1_high", "proxy_acc", "proxy_ptrue", "proxy_logloss", "n_proxy"
        ]}
    
    # --- 1) Base rate benchmark ---
    unique, counts = np.unique(s_arr, return_counts=True)
    base_rate = float(np.max(counts) / n)
    acc = float(np.mean(np.argmax(p_arr, axis=1) == s_arr))
    results["base_rate"] = base_rate
    results["acc_lift"] = acc - base_rate
    
    # --- 2) Boundary / low-confidence difficulty ---
    pmax = np.max(p_arr, axis=1)
    
    # Boundary: 0.4 <= pmax <= 0.6
    boundary_mask = (pmax >= 0.4) & (pmax <= 0.6)
    if boundary_mask.sum() > 0:
        results["acc_boundary"] = float(np.mean(np.argmax(p_arr[boundary_mask], axis=1) == s_arr[boundary_mask]))
        ptrue_boundary = [max(EPS, float(p[s])) for p, s in zip(p_arr[boundary_mask], s_arr[boundary_mask])]
        results["ptrue_boundary"] = float(np.mean(ptrue_boundary))
    else:
        results["acc_boundary"] = float("nan")
        results["ptrue_boundary"] = float("nan")
    results["n_boundary"] = int(boundary_mask.sum())
    
    # Medium-conf: conf_thresholds[0] < pmax < conf_thresholds[1]
    if len(conf_thresholds) >= 2:
        medconf_mask = (pmax > conf_thresholds[0]) & (pmax < conf_thresholds[1])
        if medconf_mask.sum() > 0:
            results["acc_medconf"] = float(np.mean(np.argmax(p_arr[medconf_mask], axis=1) == s_arr[medconf_mask]))
            ptrue_medconf = [max(EPS, float(p[s])) for p, s in zip(p_arr[medconf_mask], s_arr[medconf_mask])]
            results["ptrue_medconf"] = float(np.mean(ptrue_medconf))
        else:
            results["acc_medconf"] = float("nan")
            results["ptrue_medconf"] = float("nan")
        results["n_medconf"] = int(medconf_mask.sum())
    else:
        results["acc_medconf"] = float("nan")
        results["ptrue_medconf"] = float("nan")
        results["n_medconf"] = 0
    
    # --- 3) Edge-window (transition) performance ---
    # Find where Viterbi label changes
    edge_mask = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if s_arr[i] != s_arr[i-1]:
            # Mark ±edge_k around transition
            start = max(0, i - edge_k)
            end = min(n, i + edge_k + 1)
            edge_mask[start:end] = True
    
    if edge_mask.sum() > 0:
        results["acc_edge"] = float(np.mean(np.argmax(p_arr[edge_mask], axis=1) == s_arr[edge_mask]))
        ptrue_edge = [max(EPS, float(p[s])) for p, s in zip(p_arr[edge_mask], s_arr[edge_mask])]
        results["ptrue_edge"] = float(np.mean(ptrue_edge))
    else:
        results["acc_edge"] = float("nan")
        results["ptrue_edge"] = float("nan")
    results["n_edge"] = int(edge_mask.sum())
    
    # Non-edge
    nonedge_mask = ~edge_mask
    if nonedge_mask.sum() > 0:
        results["acc_nonedge"] = float(np.mean(np.argmax(p_arr[nonedge_mask], axis=1) == s_arr[nonedge_mask]))
        ptrue_nonedge = [max(EPS, float(p[s])) for p, s in zip(p_arr[nonedge_mask], s_arr[nonedge_mask])]
        results["ptrue_nonedge"] = float(np.mean(ptrue_nonedge))
    else:
        results["acc_nonedge"] = float("nan")
        results["ptrue_nonedge"] = float("nan")
    results["n_nonedge"] = int(nonedge_mask.sum())
    
    # --- 4) Calibration bins ---
    p_high = p_arr[:, high_state_idx]
    is_high = (s_arr == high_state_idx).astype(float)
    
    bin_edges = np.linspace(0, 1, calib_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    calib_results = []
    weighted_errors = []
    
    for i in range(calib_bins):
        mask = (p_high >= bin_edges[i]) & (p_high < bin_edges[i+1])
        if i == calib_bins - 1:  # Include right edge in last bin
            mask = (p_high >= bin_edges[i]) & (p_high <= bin_edges[i+1])
        
        if mask.sum() > 0:
            mean_pred = float(np.mean(p_high[mask]))
            emp_freq = float(np.mean(is_high[mask]))
            count = int(mask.sum())
            error = abs(emp_freq - mean_pred)
            calib_results.append((bin_centers[i], mean_pred, emp_freq, count))
            weighted_errors.append(error * count)
    
    if len(weighted_errors) > 0:
        results["calib_mae"] = float(np.sum(weighted_errors) / n)
    else:
        results["calib_mae"] = float("nan")
    
    results["calib_bins_data"] = calib_results  # Store for printing
    
    # --- 5) Confusion matrix ---
    pred_hard = np.argmax(p_arr, axis=1)
    
    tp = np.sum((pred_hard == high_state_idx) & (s_arr == high_state_idx))
    fp = np.sum((pred_hard == high_state_idx) & (s_arr != high_state_idx))
    fn = np.sum((pred_hard != high_state_idx) & (s_arr == high_state_idx))
    tn = np.sum((pred_hard != high_state_idx) & (s_arr != high_state_idx))
    
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else float("nan")
    
    results["precision_high"] = precision
    results["recall_high"] = recall
    results["f1_high"] = f1
    
    # --- 6) Observable proxy check ---
    # Build expanding percentile label (no peeking)
    dv_full = dv_series.values
    proxy_labels = []
    
    # Compute training split point
    train_end_idx = int(np.floor(train_frac * len(dv_full))) - 1
    
    for eval_idx in eval_indices:
        # Use data up to eval_idx-1 for percentile (no peeking at current day)
        if eval_idx > train_end_idx:
            hist_data = dv_full[train_end_idx:eval_idx]
            if len(hist_data) > 0:
                threshold = np.percentile(hist_data, proxy_hi_pct * 100)
                is_proxy_high = int(dv_full[eval_idx] > threshold)
                proxy_labels.append(is_proxy_high)
            else:
                proxy_labels.append(np.nan)
        else:
            proxy_labels.append(np.nan)
    
    proxy_labels = np.array(proxy_labels)
    valid_proxy = ~np.isnan(proxy_labels)
    
    if valid_proxy.sum() > 0:
        proxy_labels_clean = proxy_labels[valid_proxy].astype(int)
        p_high_clean = p_high[valid_proxy]
        pred_proxy = (p_high_clean > 0.5).astype(int)
        
        results["proxy_acc"] = float(np.mean(pred_proxy == proxy_labels_clean))
        ptrue_proxy = [p_high_clean[i] if proxy_labels_clean[i] == 1 else (1 - p_high_clean[i]) 
                      for i in range(len(proxy_labels_clean))]
        ptrue_proxy = np.clip(ptrue_proxy, EPS, 1 - EPS)
        results["proxy_ptrue"] = float(np.mean(ptrue_proxy))
        results["proxy_logloss"] = float(-np.mean(np.log(ptrue_proxy)))
        results["n_proxy"] = int(valid_proxy.sum())
    else:
        results["proxy_acc"] = float("nan")
        results["proxy_ptrue"] = float("nan")
        results["proxy_logloss"] = float("nan")
        results["n_proxy"] = 0
    
    # --- 7) Shuffle test ---
    if shuffle_test:
        s_shuffled = np.random.permutation(s_arr)
        shuffle_metrics = _regime_metrics([p for p in p_arr], s_shuffled.tolist())
        results["shuffle_acc"] = shuffle_metrics["reg_hard_acc"]
        results["shuffle_ptrue"] = shuffle_metrics["reg_ptrue_avg"]
        results["shuffle_logloss"] = shuffle_metrics["reg_logloss"]
    else:
        results["shuffle_acc"] = float("nan")
        results["shuffle_ptrue"] = float("nan")
        results["shuffle_logloss"] = float("nan")
    
    return results


def walk_forward_horizon_regime_eval(
    df: pd.DataFrame,
    horizon: int,
    train_frac: float,
    step: int,
    states: int,
    min_gap: float,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Walk-forward evaluation of horizon-H regime prediction with no lookahead.
    
    At each anchor time a:
    1. Fit HMM on data up to a
    2. Predict regime at a+h: p_hat_{a+h} = p_a @ A^h
    3. Get reference label at a+h by fitting HMM on data up to a+h only
    4. Score: correct = 1{hat_z == tilde_z}, p_true = hat_p[tilde_z]
    
    Parameters:
    - df: DataFrame with OHLC data
    - horizon: Forecast horizon H (predict regime at t+H from info up to t)
    - train_frac: Training set fraction
    - step: Step size for walk-forward
    - states: Number of HMM states
    - min_gap: Minimum gap between state means
    - seed: Random seed for reproducibility
    
    Returns:
    - dict: regime_acc, regime_ptrue_avg, n_forecasts
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Compute daily variance using Parkinson proxy
    dv = parkinson_daily_variance(df).dropna()
    idx = dv.index
    n = len(dv)
    
    # Initial training period
    t0 = int(np.floor(train_frac * n)) - 1
    
    # Ensure we have enough data for horizon
    if t0 + horizon >= n:
        return {
            "regime_acc": float("nan"),
            "regime_ptrue_avg": float("nan"),
            "n_forecasts": 0
        }
    
    correct_list = []
    ptrue_list = []
    
    # Loop over anchors where we can evaluate a label at anchor + horizon
    last_anchor = n - horizon - 1
    
    for anchor in range(t0, last_anchor, step):
        try:
            # 1. Fit HMM up to anchor point a
            train_data_a = dv.loc[:idx[anchor]]
            logvar_a = np.log(train_data_a + EPS)
            res_a = fit_hmm_logvar(logvar_a, n_states=states, min_gap=min_gap)
            
            # Get filtered probability at anchor: p_a
            p_a = res_a["pT"]
            A = res_a["A"]
            
            # 2. Predict distribution at horizon: p_hat_{a+h} = p_a @ A^h
            p_hat_ah = p_a.copy()
            for _ in range(horizon):
                p_hat_ah = p_hat_ah @ A
            
            # Predicted regime label: z_hat_{a+h} = argmax(p_hat_{a+h})
            z_hat_ah = int(np.argmax(p_hat_ah))
            
            # 3. Compute reference label at a+h (no lookahead)
            # Fit HMM on data up to a+h only
            target_idx = anchor + horizon
            if target_idx >= n:
                continue
            
            train_data_ah = dv.loc[:idx[target_idx]]
            logvar_ah = np.log(train_data_ah + EPS)
            res_ah = fit_hmm_logvar(logvar_ah, n_states=states, min_gap=min_gap)
            
            # Get Viterbi state at a+h (last state from Viterbi)
            z_tilde_ah = int(res_ah["states"][-1])
            
            # 4. Map states to consistent labels (by emission mean)
            # Both fits already sort by mean, so indices are consistent
            # But we need to ensure we're comparing the same state ordering
            # Since both are sorted by mean, we can compare directly
            
            # 5. Score
            correct = 1 if z_hat_ah == z_tilde_ah else 0
            p_true = float(p_hat_ah[z_tilde_ah])
            
            correct_list.append(correct)
            ptrue_list.append(p_true)
            
        except Exception as e:
            # Skip if fitting fails
            continue
    
    # Aggregate metrics
    if len(correct_list) == 0:
        return {
            "regime_acc": float("nan"),
            "regime_logloss": float("nan"),
            "n_forecasts": 0
        }
    
    # Compute log loss: -mean(log(p_true))
    ptrue_arr = np.array(ptrue_list)
    ptrue_clipped = np.clip(ptrue_arr, EPS, 1 - EPS)  # Avoid log(0)
    logloss = float(-np.mean(np.log(ptrue_clipped)))
    
    return {
        "regime_acc": float(np.mean(correct_list)),
        "regime_logloss": logloss,
        "n_forecasts": int(len(correct_list))
    }


def walk_forward_regime_eval(
    df: pd.DataFrame,
    train_frac: float,
    step: int,
    states: int,
    min_gap: float,
    one_step: bool,
    # Sanity check parameters
    calib_bins: int = 10,
    edge_k: int = 3,
    conf_thresholds: List[float] = [0.6, 0.8],
    proxy_hi_pct: float = 0.8,
    shuffle_test: bool = False
) -> Dict[str, float]:
    """
    Walk-forward regime classification evaluation with no-peeking constraint
    plus comprehensive sanity checks.
    
    This function:
    1. Fits HMM up to anchor point t
    2. Predicts regime using filtered probabilities (or one-step ahead)
    3. Labels using Viterbi state computed only up to evaluation time (no peeking)
    4. Computes classification metrics
    5. Runs sanity checks
    
    Parameters:
    - df: DataFrame with OHLC data
    - train_frac: Training set fraction (e.g., 0.7 for 70% training)
    - step: Step size for walk-forward (1 = daily evaluation)
    - states: Number of HMM states
    - min_gap: Minimum gap between state means
    - one_step: If True, use one-step-ahead prediction (p_{t+1} = p_t @ A)
    - calib_bins: Number of bins for calibration analysis
    - edge_k: Edge window size (±k days around transitions)
    - conf_thresholds: [low, high] for medium-confidence analysis
    - proxy_hi_pct: Percentile threshold for observable proxy
    - shuffle_test: If True, run label shuffle test
    
    Returns:
    - dict: Regime classification metrics + sanity checks
    """
    # Compute daily variance using Parkinson proxy
    dv = parkinson_daily_variance(df).dropna()
    idx = dv.index
    n = len(dv)
    
    # Initial training period
    t0 = int(np.floor(train_frac * n)) - 1
    
    # Default to step=1 for regime scoring (daily evaluation)
    if step <= 0:
        step = 1
    
    p_pred_list = []
    s_label_list = []
    eval_indices = []  # Track which indices we evaluated
    high_state_idx = None  # Will be set from first fit
    
    # Loop over anchors where we can evaluate a label
    # If same-day: label at t; if one-step: label at t+1 (needs an extra day)
    last_anchor = (n - 2) if one_step else (n - 1)
    
    for anchor in range(t0, last_anchor, step):
        try:
            # Fit HMM up to anchor point t
            train_data = dv.loc[:idx[anchor]]
            logvar_t = np.log(train_data + EPS)
            res_t = fit_hmm_logvar(logvar_t, n_states=states, min_gap=min_gap)
            
            # Identify high-vol state (higher mean log-variance)
            if high_state_idx is None:
                high_state_idx = int(np.argmax(res_t["mu"]))
            
            # Prediction: filtered probabilities at t
            p_t = res_t["pT"]
            
            if one_step:
                # One-step-ahead prediction: p_{t+1} = p_t @ A
                p_pred = p_t @ res_t["A"]
                
                # Label at t+1: fit up to anchor+1 (to avoid peeking)
                train_data_tp1 = dv.loc[:idx[anchor + 1]]
                logvar_tp1 = np.log(train_data_tp1 + EPS)
                res_tp1 = fit_hmm_logvar(logvar_tp1, n_states=states, min_gap=min_gap)
                
                # Viterbi state at t+1 (aligned by fit_hmm_logvar)
                s_label = int(res_tp1["states"][-1])
                eval_idx = anchor + 1
            else:
                # Same-day prediction: use filtered probabilities at t
                p_pred = p_t
                
                # Viterbi state at t (aligned by fit_hmm_logvar)
                s_label = int(res_t["states"][-1])
                eval_idx = anchor
            
            # Store prediction and label
            p_pred_list.append(np.asarray(p_pred, dtype=float))
            s_label_list.append(s_label)
            eval_indices.append(eval_idx)
            
        except Exception as e:
            # Skip if fitting fails (rare, but possible with extreme data)
            continue
    
    # Compute core metrics
    if len(p_pred_list) == 0:
        return {
            "reg_hard_acc": float("nan"),
            "reg_ptrue_avg": float("nan"),
            "reg_logloss": float("nan"),
            "reg_ptrue_geom": float("nan"),
            "n_points": 0
        }
    
    results = _regime_metrics(p_pred_list, s_label_list)
    
    # Compute sanity checks
    if high_state_idx is not None:
        sanity = _compute_sanity_checks(
            p_pred_list, s_label_list, high_state_idx, dv, eval_indices,
            calib_bins, edge_k, conf_thresholds, proxy_hi_pct, shuffle_test, train_frac
        )
        results.update(sanity)
    
    return results


def print_sanity_checks(results: Dict, edge_k: int, calib_bins: int, proxy_hi_pct: float):
    """Print formatted sanity checks."""
    print()
    print("SANITY CHECKS")
    print("-" * 60)
    
    # Base rate
    print(f"Base rate (majority):              {results.get('base_rate', float('nan')):.3f}")
    print(f"Accuracy lift over base:           {results.get('acc_lift', float('nan')):.3f}")
    print()
    
    # Boundary/confidence
    print(f"Boundary (0.4-0.6) acc/ptrue/n:    {results.get('acc_boundary', float('nan')):.3f} / "
          f"{results.get('ptrue_boundary', float('nan')):.3f} / {results.get('n_boundary', 0)}")
    print(f"Medium-conf (0.6-0.8) acc/ptrue/n: {results.get('acc_medconf', float('nan')):.3f} / "
          f"{results.get('ptrue_medconf', float('nan')):.3f} / {results.get('n_medconf', 0)}")
    print()
    
    # Edge window
    print(f"Edge-window (±{edge_k}) acc/ptrue: {results.get('acc_edge', float('nan')):.3f} / "
          f"{results.get('ptrue_edge', float('nan')):.3f}  (n={results.get('n_edge', 0)})")
    print(f"Non-edge acc/ptrue:                {results.get('acc_nonedge', float('nan')):.3f} / "
          f"{results.get('ptrue_nonedge', float('nan')):.3f}  (n={results.get('n_nonedge', 0)})")
    print()
    
    # Calibration
    print(f"Calibration (High state):")
    print(f"  bins: {calib_bins}, weighted MAE: {results.get('calib_mae', float('nan')):.3f}")
    
    # Print sample calibration bins
    if "calib_bins_data" in results and len(results["calib_bins_data"]) > 0:
        calib_data = results["calib_bins_data"]
        # Show first 5 bins or all if fewer
        n_show = min(5, len(calib_data))
        print(f"  Sample bins (showing {n_show}):")
        for i in range(n_show):
            center, mean_pred, emp_freq, count = calib_data[i]
            print(f"    {center:.2f}: pred={mean_pred:.3f}, emp={emp_freq:.3f}, n={count}")
    print()
    
    # Confusion matrix
    print(f"Confusion (High as positive):")
    print(f"  precision/recall/F1:              {results.get('precision_high', float('nan')):.3f} / "
          f"{results.get('recall_high', float('nan')):.3f} / {results.get('f1_high', float('nan')):.3f}")
    print()
    
    # Observable proxy
    print(f"Observable-proxy (Parkinson > {proxy_hi_pct:.0%}):")
    print(f"  acc/ptrue/logloss/n:              {results.get('proxy_acc', float('nan')):.3f} / "
          f"{results.get('proxy_ptrue', float('nan')):.3f} / "
          f"{results.get('proxy_logloss', float('nan')):.3f} / {results.get('n_proxy', 0)}")
    print()
    
    # Shuffle test
    if not np.isnan(results.get('shuffle_acc', float('nan'))):
        print(f"Shuffle test:")
        print(f"  acc/ptrue/logloss:                {results.get('shuffle_acc', float('nan')):.3f} / "
              f"{results.get('shuffle_ptrue', float('nan')):.3f} / "
              f"{results.get('shuffle_logloss', float('nan')):.3f}")
        print()


def write_sanity_csv(results: Dict, symbol: str, args, output_file: str = None):
    """Write sanity check results to CSV."""
    if output_file is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"outputs/results/regime_sanity_{symbol}_{ts}.csv"
    
    # Build row
    row = {
        "symbol": symbol,
        "years": args.years,
        "states": args.states,
        "train_frac": args.train_frac,
        "one_step": int(args.one_step),
        "n_points": results.get("n_points", 0),
        # Core metrics
        "reg_hard_acc": results.get("reg_hard_acc", float("nan")),
        "reg_ptrue_avg": results.get("reg_ptrue_avg", float("nan")),
        "reg_logloss": results.get("reg_logloss", float("nan")),
        "reg_ptrue_geom": results.get("reg_ptrue_geom", float("nan")),
        # Sanity metrics
        "base_rate": results.get("base_rate", float("nan")),
        "acc_lift": results.get("acc_lift", float("nan")),
        "acc_boundary": results.get("acc_boundary", float("nan")),
        "ptrue_boundary": results.get("ptrue_boundary", float("nan")),
        "n_boundary": results.get("n_boundary", 0),
        "acc_medconf": results.get("acc_medconf", float("nan")),
        "ptrue_medconf": results.get("ptrue_medconf", float("nan")),
        "n_medconf": results.get("n_medconf", 0),
        "acc_edge": results.get("acc_edge", float("nan")),
        "ptrue_edge": results.get("ptrue_edge", float("nan")),
        "n_edge": results.get("n_edge", 0),
        "acc_nonedge": results.get("acc_nonedge", float("nan")),
        "ptrue_nonedge": results.get("ptrue_nonedge", float("nan")),
        "n_nonedge": results.get("n_nonedge", 0),
        "calib_mae": results.get("calib_mae", float("nan")),
        "precision_high": results.get("precision_high", float("nan")),
        "recall_high": results.get("recall_high", float("nan")),
        "f1_high": results.get("f1_high", float("nan")),
        "proxy_acc": results.get("proxy_acc", float("nan")),
        "proxy_ptrue": results.get("proxy_ptrue", float("nan")),
        "proxy_logloss": results.get("proxy_logloss", float("nan")),
        "n_proxy": results.get("n_proxy", 0),
        "shuffle_acc": results.get("shuffle_acc", float("nan")),
        "shuffle_ptrue": results.get("shuffle_ptrue", float("nan")),
        "shuffle_logloss": results.get("shuffle_logloss", float("nan")),
    }
    
    # Write CSV
    fieldnames = list(row.keys())
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    
    print(f"Sanity CSV saved: {output_file}")


def main():
    """Main function for HMM regime prediction validation."""
    ap = argparse.ArgumentParser(
        description="HMM Regime Prediction Validation (no peeking) with Sanity Checks"
    )
    ap.add_argument("--symbol", type=str, default="AMZN", help="Stock symbol")
    ap.add_argument("--years", type=int, default=5, help="Years of historical data")
    ap.add_argument("--states", type=int, default=2, help="Number of HMM states")
    ap.add_argument("--min_gap", type=float, default=0.10, 
                   help="Minimum gap between state means")
    ap.add_argument("--train_frac", type=float, default=0.70,
                   help="Training set fraction (0.7 = 70%% training)")
    ap.add_argument("--step", type=int, default=1,
                   help="Walk-forward step; 1 for daily regime scoring")
    ap.add_argument("--horizon", type=int, default=None,
                   help="Forecast horizon H (predict regime at t+H from info up to t). If set, uses horizon-H evaluation.")
    ap.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    ap.add_argument("--one_step", action="store_true",
                   help="Use one-step-ahead regime prediction p_{t+1}=p_t A with label at t+1 (ignored if --horizon is set)")
    ap.add_argument("--quiet", action="store_true",
                   help="Quiet mode - minimal output for scripting")
    
    # Sanity check flags
    ap.add_argument("--calib_bins", type=int, default=10,
                   help="Number of bins for calibration analysis")
    ap.add_argument("--edge_k", type=int, default=3,
                   help="Edge window size (±k days around regime transitions)")
    ap.add_argument("--conf_thresholds", nargs=2, type=float, default=[0.6, 0.8],
                   help="Confidence thresholds [low high] for medium-conf analysis")
    ap.add_argument("--proxy_hi_pct", type=float, default=0.8,
                   help="Percentile threshold for observable High-vol proxy")
    ap.add_argument("--shuffle_test", action="store_true",
                   help="Run label shuffle test (sanity guard)")
    ap.add_argument("--csv_sanity", action="store_true",
                   help="Write sanity metrics to CSV")
    
    args = ap.parse_args()
    
    try:
        # Fetch and prepare data
        if not args.quiet:
            print(f"Fetching {args.years} years of data for {args.symbol}...")
        
        df = get_data(args.symbol, args.years)
        
        # Use horizon-H evaluation if horizon is specified
        if args.horizon is not None:
            if not args.quiet:
                print(f"Running horizon-H regime evaluation...")
                print(f"Horizon: {args.horizon}")
                print(f"Training fraction: {args.train_frac:.1%}")
                print(f"Step size: {args.step}")
            
            # Run horizon-H validation
            results = walk_forward_horizon_regime_eval(
                df=df,
                horizon=args.horizon,
                train_frac=args.train_frac,
                step=args.step,
                states=args.states,
                min_gap=args.min_gap,
                seed=args.seed
            )
            
            # Print results
            if args.quiet:
                # Minimal output for scripting (core metrics only)
                print(f"regime_acc: {results.get('regime_acc', float('nan')):.6f}")
                print(f"regime_logloss: {results.get('regime_logloss', float('nan')):.6f}")
                print(f"n_forecasts: {results.get('n_forecasts', 0)}")
            else:
                # Full formatted output
                print("=" * 60)
                print("HMM REGIME-PREDICTION VALIDATION (HORIZON-H)")
                print("=" * 60)
                print(f"Symbol: {args.symbol} | Years: {args.years} | States: {args.states} | Horizon: {args.horizon}")
                print(f"Train frac: {args.train_frac:.0%} | Step: {args.step}")
                print("-" * 60)
                print(f"Regime accuracy:       {results.get('regime_acc', float('nan')):.3f}")
                print(f"Log loss:              {results.get('regime_logloss', float('nan')):.3f}")
                print(f"Forecasts:             {results.get('n_forecasts', 0)}")
                print("=" * 60)
        else:
            # Use original one-step evaluation
            if not args.quiet:
                print(f"Running walk-forward regime evaluation...")
                print(f"Training fraction: {args.train_frac:.1%}")
                print(f"Step size: {args.step}")
                print(f"One-step-ahead: {args.one_step}")
            
            # Run validation with sanity checks
            results = walk_forward_regime_eval(
                df=df,
                train_frac=args.train_frac,
                step=args.step,
                states=args.states,
                min_gap=args.min_gap,
                one_step=args.one_step,
                calib_bins=args.calib_bins,
                edge_k=args.edge_k,
                conf_thresholds=args.conf_thresholds,
                proxy_hi_pct=args.proxy_hi_pct,
                shuffle_test=args.shuffle_test
            )
            
            # Print results
            if args.quiet:
                # Minimal output for scripting (core metrics only)
                for k in ["reg_hard_acc", "reg_ptrue_avg", "reg_logloss", "reg_ptrue_geom", "n_points"]:
                    if k == "n_points":
                        print(f"{k}: {results[k]}")
                    else:
                        print(f"{k}: {results[k]:.6f}")
                # Add key sanity metrics
                print(f"base_rate: {results.get('base_rate', float('nan')):.6f}")
                print(f"acc_lift: {results.get('acc_lift', float('nan')):.6f}")
                print(f"calib_mae: {results.get('calib_mae', float('nan')):.6f}")
            else:
                # Full formatted output
                print("=" * 60)
                print("HMM REGIME-PREDICTION VALIDATION")
                print("=" * 60)
                print(f"Symbol: {args.symbol} | Years: {args.years} | States: {args.states} | one_step={args.one_step}")
                print(f"Train frac: {args.train_frac:.0%} | Step: {args.step}")
                print("-" * 60)
                print(f"Hard accuracy:       {results['reg_hard_acc']:.3f}")
                print(f"Avg p(true state):   {results['reg_ptrue_avg']:.3f}")
                print(f"Log loss:            {results['reg_logloss']:.3f}")
                print(f"Exp(log loss):       {results['reg_ptrue_geom']:.3f}")
                print(f"Obs (points):        {results['n_points']}")
                print()
                
                # Interpretation
                print("INTERPRETATION:")
                if results['reg_hard_acc'] > 0.70:
                    print("✓ High classification accuracy - model confidently predicts regimes")
                elif results['reg_hard_acc'] > 0.55:
                    print("~ Moderate classification accuracy - better than random")
                else:
                    print("⚠ Low classification accuracy - struggles to predict regimes")
                
                if results['reg_ptrue_avg'] > 0.70:
                    print("✓ High average p(true) - confident and accurate predictions")
                elif results['reg_ptrue_avg'] > 0.55:
                    print("~ Moderate average p(true) - reasonable confidence")
                else:
                    print("⚠ Low average p(true) - low confidence in predictions")
                
                if results['reg_ptrue_geom'] > 0.65:
                    print("✓ High geometric mean p(true) - consistently confident")
                elif results['reg_ptrue_geom'] > 0.50:
                    print("~ Moderate geometric mean p(true)")
                else:
                    print("⚠ Low geometric mean p(true) - some very uncertain predictions")
                
                # Print sanity checks
                print_sanity_checks(results, args.edge_k, args.calib_bins, args.proxy_hi_pct)
                
                print("=" * 60)
        
        # Write CSV if requested
        if args.csv_sanity:
            write_sanity_csv(results, args.symbol, args)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("Try a different symbol, adjust parameters, or wait for rate limits to reset")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
