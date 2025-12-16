#!/usr/bin/env python3
"""
Volatility Validation Suite Runner (multi-symbol, multi-horizon)
=================================================================

Runs comprehensive volatility forecast validation tests for multiple symbols
across multiple horizons with dynamic step ranges and saves a CSV + txt summary.

Default: 10 years of history, broad universe of liquid assets.

Usage:
    python scripts/run_vol_validation.py                    # Full suite, 10 years
    python scripts/run_vol_validation.py --years 15         # 15-year history
    python scripts/run_vol_validation.py --symbols "SPY,QQQ" # Custom symbols
    python scripts/run_vol_validation.py --steps auto       # Dynamic step ranges
    python scripts/run_vol_validation.py --horizons "45,60,90" --steps auto
"""

import subprocess
import time
import random
from datetime import datetime
import csv
import os
import sys
import argparse
from typing import List, Union, Optional
import pandas as pd

def generate_steps(horizon: int, step_min: int = 5, step_incr: int = 10, 
                   include_H: bool = True) -> List[int]:
    """
    Generate step values for a given horizon using auto mode.
    
    Args:
        horizon: Forecast horizon in days
        step_min: Minimum step size (default: 5)
        step_incr: Step increment (default: 15)
        include_H: Whether to include non-overlapping H step (default: True)
    
    Returns:
        List of step values sorted in ascending order
    """
    if step_min > horizon:
        if include_H:
            return [horizon]
        else:
            return []
    
    # Generate steps from step_min, but if next step would exceed H, use H instead
    steps = []
    current = step_min
    
    while current < horizon:
        steps.append(current)
        next_val = current + step_incr
        if next_val >= horizon:
            # If next step would exceed or equal H, use H instead (if requested)
            if include_H and horizon not in steps:
                steps.append(horizon)
            break
        current = next_val
    
    # If we haven't added any steps yet and include_H is True, add H
    if not steps and include_H:
        steps.append(horizon)
    
    return sorted(steps)

def generate_uniform_steps(max_horizon: int, step_min: int = 5, step_incr: int = 10) -> List[int]:
    """
    Generate uniform step values across all horizons to avoid overlap.
    
    Args:
        max_horizon: Maximum horizon across all tests
        step_min: Minimum step size (default: 5)
        step_incr: Step increment (default: 10)
    
    Returns:
        List of step values sorted in ascending order
    """
    # Generate steps from step_min to max_horizon in increments of step_incr
    steps = list(range(step_min, max_horizon + 1, step_incr))
    return sorted(steps)

def parse_steps(steps_str: str, horizons: List[int], step_min: int = 5, 
                step_incr: int = 10, include_H: bool = True) -> dict:
    """
    Parse step specification and return step sets for each horizon.
    
    Args:
        steps_str: Step specification ("auto" or comma-separated list)
        horizons: List of horizons
        step_min: Minimum step for auto mode
        step_incr: Step increment for auto mode
        include_H: Whether to include H in auto mode
    
    Returns:
        Dictionary mapping horizon to list of steps
    """
    if steps_str == "auto":
        # Generate steps for each horizon independently
        # This allows testing the same step size across different horizons
        return {H: generate_steps(H, step_min, step_incr, include_H) for H in horizons}
    else:
        # Parse comma-separated list, handling legacy "H" format
        step_parts = [s.strip() for s in steps_str.split(",")]
        step_list = []
        for part in step_parts:
            if part == "H":
                # For legacy "H" format, we'll use the horizon value
                continue  # Will be handled per horizon below
            else:
                step_list.append(int(part))
        
        # Handle legacy "H" format
        if "H" in step_parts:
            return {H: step_list + [H] for H in horizons}
        else:
            return {H: step_list for H in horizons}

def run_validation(symbol, years, states, horizon, step, ratio_cutoff, train_frac, band_mult, quiet=True):
    # Get the project root directory (parent of scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    volatility_script = os.path.join(project_root, "validation", "volatility.py")
    
    cmd = [
        sys.executable, volatility_script,
        "--symbol", symbol,
        "--years", str(years),
        "--states", str(states),
        "--horizon", str(horizon),
        "--step", str(step),
        "--ratio_cutoff", str(ratio_cutoff),
        "--train_frac", str(train_frac),
        "--band_mult", str(band_mult),
        "--quiet"
    ]
    
    # Set PYTHONPATH to include project root so imports work
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{pythonpath}"
    else:
        env["PYTHONPATH"] = project_root
    
    try:
        print(f"Running: {symbol} | H={horizon}, step={step}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            # Check for connection errors specifically
            stderr_text = result.stderr if result.stderr else ""
            stdout_text = result.stdout if result.stdout else ""
            error_text = stderr_text + stdout_text
            
            if "ConnectionRefusedError" in error_text or "Failed to connect to IBKR Gateway" in error_text:
                print(f"Connection error for {symbol} H={horizon} step={step}: IBKR Gateway not accessible")
                print(f"  Make sure IBKR Gateway is running and API is enabled on port 4001")
            else:
                print(f"Error for {symbol} H={horizon} step={step}: {stderr_text[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"Timeout for {symbol} H={horizon} step={step}")
        return None
    except Exception as e:
        print(f"Exception for {symbol} H={horizon} step={step}: {e}")
        return None

def parse_results(output):
    if not output:
        return None
    res = {}
    for line in output.splitlines():
        if ":" in line:
            k, v = [x.strip() for x in line.split(":", 1)]
            try:
                res[k] = float(v) if "." in v else int(v)
            except ValueError:
                res[k] = v
    return res

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='HMM Volatility Validation Suite')
    parser.add_argument('--years', type=int, default=20, 
                       help='Years of historical data (default: 20, will use maximum available if less)')
    parser.add_argument('--symbols', type=str, default=None, 
                       help='Comma-separated list of symbols (default: broad universe)')
    parser.add_argument('--horizons', type=str, default="45,60,75,90",
                       help='Comma-separated list of horizons (default: 45,60,75,90)')
    parser.add_argument('--steps', type=str, default="auto",
                       help='Step specification: "auto" for dynamic generation or comma-separated list (default: auto)')
    parser.add_argument('--step_min', type=int, default=5,
                       help='Minimum step size for auto mode (default: 5)')
    parser.add_argument('--step_incr', type=int, default=15,
                       help='Step increment for auto mode (default: 15)')
    parser.add_argument('--include_H', type=str, default="true",
                       help='Include non-overlapping H step in auto mode (default: true)')
    parser.add_argument('--states', type=int, default=2,
                       help='Number of HMM states (default: 2)')
    parser.add_argument('--ratio_cutoff', type=float, default=1.48,
                       help='Ratio cutoff for excluding extreme predictions (default: 1.48)')
    parser.add_argument('--train_frac', type=float, default=0.70,
                       help='Training set fraction (default: 0.70)')
    parser.add_argument('--band_mult', type=float, default=1.25,
                       help='Band multiplier for coverage evaluation (default: 1.25)')
    
    args = parser.parse_args()
    
    # --- Global config ---
    years = args.years
    states = args.states
    ratio_cutoff = args.ratio_cutoff
    train_frac = args.train_frac
    band_mult = args.band_mult

    # Broader symbol universe: Index/sector ETFs + rates/credit + commodities + EM + mega-cap equities
    default_symbols = [
        # Major index ETFs
        "SPY", "QQQ", "IWM", "DIA", 
        # Rates and credit
        "TLT", "HYG", 
        # Commodities
        "GLD", "GDX", "USO", 
        # Emerging markets
        "EEM", "FXI", 
        # Sector ETFs
        "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU",
        # Mega-cap equities
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "IBM", "INTC", "AMD", "NFLX",
        # Financials
        "JPM", "BAC", 
        # Energy
        "XOM", "CVX", 
        # Healthcare/Consumer
        "UNH", "V", "MA", "KO", "MCD", "NKE"
    ]
    
    # Symbol selection
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = default_symbols

    # Parse horizons
    horizons = [int(h.strip()) for h in args.horizons.split(",")]
    
    # Parse include_H
    include_H = args.include_H.lower() in ["true", "1", "yes", "y"]
    
    # Parse steps
    step_sets = parse_steps(args.steps, horizons, args.step_min, args.step_incr, include_H)

    # Output prep
    os.makedirs("outputs/results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_name = f"outputs/results/validation_results_{ts}.txt"
    xlsx_name = f"outputs/results/validation_results_{ts}.xlsx"

    header = [
        "symbol","years","states","horizon","step","ratio_cutoff","train_frac",
        "band_mult",
        "n_forecasts","n_kept","excluded_pct",
        "mape_vol_mid",
        "in_band_share_kept",
        "med_ratio_vol_kept","mean_log_ratio_vol_kept","over_share_vol_kept"
    ]

    all_rows = []
    print("="*80)
    print("HMM VALIDATION SUITE (multi-symbol, multi-horizon)")
    print("="*80)
    print(f"Symbols: {len(symbols)} | Horizons: {horizons} | States: {states}")
    print(f"Train fraction: {train_frac:.1%} | Ratio cutoff: {ratio_cutoff}")
    print(f"Step mode: {args.steps} | Step min: {args.step_min} | Step incr: {args.step_incr}")
    print("-"*80)

    # Print step sets for each horizon
    for H in horizons:
        steps = step_sets[H]
        if not steps:
            print(f"WARNING: No valid steps for horizon {H} (step_min={args.step_min} > H)")
            continue
        print(f"H={H}: steps={steps}")

    print("-"*80)

    for sym in symbols:
        for H in horizons:
            steps = step_sets[H]
            if not steps:
                continue
                
            for step in steps:
                # Small random delay before starting subprocess to stagger connections
                time.sleep(random.uniform(0.5, 1.5))
                
                out = run_validation(
                    symbol=sym, years=years, states=states,
                    horizon=H, step=step, ratio_cutoff=ratio_cutoff,
                    train_frac=train_frac, band_mult=band_mult, quiet=True
                )
                time.sleep(5.0)  # be kind to API and IBKR Gateway (increased from 1.5s)
                if not out:
                    continue
                r = parse_results(out)
                if not r:
                    continue

                row = {
                    "symbol": sym, "years": years, "states": states,
                    "horizon": H, "step": step, "ratio_cutoff": ratio_cutoff, "train_frac": train_frac,
                    "band_mult": r.get("band_mult", band_mult),
                    "n_forecasts": r.get("n_forecasts", 0),
                    "n_kept": r.get("n_kept", 0),
                    "excluded_pct": r.get("excluded_pct", float("nan")),
                    "mape_vol_mid": r.get("mape_vol_mid", float("nan")),
                    "in_band_share_kept": r.get("in_band_share_kept", float("nan")),
                    "med_ratio_vol_kept": r.get("med_ratio_vol_kept", float("nan")),
                    "mean_log_ratio_vol_kept": r.get("mean_log_ratio_vol_kept", float("nan")),
                    "over_share_vol_kept": r.get("over_share_vol_kept", float("nan")),
                }
                all_rows.append(row)
                print(f"[OK] {sym} H={H} step={step}: MAPE={row['mape_vol_mid']:.3f}, band={row['in_band_share_kept']*100:.1f}%, excl={row['excluded_pct']*100:.1f}%, kept={row['n_kept']}/{row['n_forecasts']}")

    # Write Excel file (primary output)
    if all_rows:
        # Create DataFrame from results
        df_results = pd.DataFrame(all_rows, columns=header)
        
        # Write Excel file
        df_results.to_excel(xlsx_name, index=False, engine='openpyxl')
        print(f"\nExcel saved: {xlsx_name}")
    else:
        # No results - write empty Excel file with just headers
        df_results = pd.DataFrame(columns=header)
        df_results.to_excel(xlsx_name, index=False, engine='openpyxl')
        print(f"\nExcel saved (no results): {xlsx_name}")

    # Write TXT summary
    with open(txt_name, "w") as f:
        f.write("HMM MODEL VALIDATION RESULTS - MULTI SYMBOL\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Symbols tested: {len(symbols)}\n")
        f.write(f"Horizons: {horizons}\n")
        f.write(f"Step mode: {args.steps}\n")
        f.write(f"Proxy: Parkinson, States: {states}, train_frac={train_frac}, ratio_cutoff={ratio_cutoff}\n")
        f.write("-"*80 + "\n\n")
        
        # Print step sets used
        f.write("STEP SETS USED:\n")
        for H in horizons:
            steps = step_sets[H]
            if steps:
                f.write(f"H={H}: {steps}\n")
        f.write("\n")
        
        if not all_rows:
            f.write("No successful results.\n")
        else:
            # Best per horizon summary (using best step for each horizon)
            f.write("BEST RESULTS PER HORIZON:\n")
            for H in horizons:
                subset = [r for r in all_rows if r["horizon"] == H]
                if subset:
                    best = min(subset, key=lambda x: x["mape_vol_mid"])
                    f.write(f"H={H}: {best['symbol']} step={best['step']} (MAPE={best['mape_vol_mid']:.3f}, excl={best['excluded_pct']*100:.1f}%)\n")
            f.write("\nExcel saved to: " + xlsx_name + "\n")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print(f"Total runs: {len(all_rows)}")
    print(f"Excel saved: {xlsx_name}")
    print(f"Summary saved: {txt_name}")
    print("="*80)

if __name__ == "__main__":
    main()

