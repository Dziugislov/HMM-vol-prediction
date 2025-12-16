#!/usr/bin/env python3
"""
Regime Validation Suite Runner (multi-symbol, multi-horizon)
=============================================================

Runs comprehensive regime prediction validation tests for multiple symbols
across multiple forecast horizons with proper horizon-H evaluation.

Default: 20 years of history, broad universe of liquid assets.

Usage:
    python scripts/run_regime_validation.py                    # Full suite
    python scripts/run_regime_validation.py --years 15         # 15-year history
    python scripts/run_regime_validation.py --symbols "SPY,QQQ" # Custom symbols
    python scripts/run_regime_validation.py --steps "H"        # Step = horizon
"""

import subprocess
import time
from datetime import datetime
import os
import sys
import argparse
from typing import List, Optional
import pandas as pd


def parse_steps(steps_str: str, horizons: List[int]) -> dict:
    """
    Parse step specification and return step sets for each horizon.
    
    Args:
        steps_str: Step specification ("H" or comma-separated list)
        horizons: List of horizons
    
    Returns:
        Dictionary mapping horizon to list of steps
    """
    if steps_str == "H":
        # Set step = horizon for each horizon
        return {H: [H] for H in horizons}
    else:
        # Parse comma-separated list
        step_parts = [s.strip() for s in steps_str.split(",")]
        step_list = []
        for part in step_parts:
            if part == "H":
                # For "H" format, use horizon value per horizon
                continue  # Will be handled per horizon below
            else:
                step_list.append(int(part))
        
        # Handle "H" in list
        if "H" in step_parts:
            return {H: step_list + [H] for H in horizons}
        else:
            return {H: step_list for H in horizons}


def run_validation(symbol, years, states, horizon, step, train_frac, seed=None, quiet=True):
    """Run regime validation for a single symbol/horizon/step combination."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    regime_script = os.path.join(project_root, "validation", "regime.py")
    
    cmd = [
        sys.executable, regime_script,
        "--symbol", symbol,
        "--years", str(years),
        "--states", str(states),
        "--horizon", str(horizon),
        "--step", str(step),
        "--train_frac", str(train_frac),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if quiet:
        cmd.append("--quiet")
    
    # Set PYTHONPATH to include project root so imports work
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{pythonpath}"
    else:
        env["PYTHONPATH"] = project_root
    
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        if r.returncode != 0:
            print(f"[ERR] {symbol} H={horizon} step={step}: {r.stderr}")
            return None
        
        # Parse output
        out = {}
        for line in r.stdout.splitlines():
            if ":" in line:
                k, v = [x.strip() for x in line.split(":", 1)]
                try:
                    if k == "n_forecasts":
                        out[k] = int(v)
                    else:
                        out[k] = float(v)
                except ValueError:
                    pass
        
        return out
        
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {symbol} H={horizon} step={step}")
        return None
    except Exception as e:
        print(f"[EXCEPTION] {symbol} H={horizon} step={step}: {e}")
        return None


def main():
    """Main function for multi-symbol regime validation."""
    parser = argparse.ArgumentParser(description='HMM Regime Validation Suite')
    parser.add_argument('--years', type=int, default=20, 
                       help='Years of historical data (default: 20)')
    parser.add_argument('--symbols', type=str, default=None, 
                       help='Comma-separated list of symbols (default: broad universe)')
    parser.add_argument('--horizons', type=str, default="20,45,60,90",
                       help='Comma-separated list of horizons (default: 20,45,60,90)')
    parser.add_argument('--steps', type=str, default="H",
                       help='Step specification: "H" for step=horizon or comma-separated list (default: H)')
    parser.add_argument('--states', type=int, default=2,
                       help='Number of HMM states (default: 2)')
    parser.add_argument('--train_frac', type=float, default=0.70,
                       help='Training set fraction (default: 0.70)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # --- Global config ---
    years = args.years
    states = args.states
    train_frac = args.train_frac
    seed = args.seed
    
    # Broader symbol universe
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
    
    # Parse steps
    step_sets = parse_steps(args.steps, horizons)
    
    # Output prep
    os.makedirs("outputs/results", exist_ok=True)
    xlsx_name = "outputs/results/regime_summary.xlsx"
    
    header = [
        "symbol", "years", "states", "horizon", "step", "train_frac", "seed",
        "n_forecasts",
        "regime_acc",
        "regime_logloss"
    ]
    
    all_rows = []
    
    print("=" * 64)
    print("HMM REGIME PREDICTION VALIDATION SUITE")
    print("=" * 64)
    print(f"Symbols: {len(symbols)} | Years: {years} | Horizons: {horizons}")
    print(f"States: {states} | train_frac={train_frac}")
    print(f"Step mode: {args.steps}")
    print("-" * 64)
    
    # Run validation for each symbol, horizon, and step
    for sym in symbols:
        for H in horizons:
            steps = step_sets[H]
            if not steps:
                continue
            
            for step in steps:
                print(f"[RUN] {sym} H={H} step={step}...")
                r = run_validation(sym, years, states, H, step, train_frac, seed, quiet=True)
                
                time.sleep(0.5)  # Rate limiting
                
                if not r:
                    continue
                
                row = {
                    "symbol": sym, "years": years, "states": states,
                    "horizon": H, "step": step, "train_frac": train_frac, "seed": seed if seed is not None else "",
                    "n_forecasts": r.get("n_forecasts", 0),
                    "regime_acc": r.get("regime_acc", float("nan")),
                    "regime_logloss": r.get("regime_logloss", float("nan")),
                }
                all_rows.append(row)
                
                # Print quick summary
                acc = row['regime_acc']
                logloss = row['regime_logloss']
                n = row['n_forecasts']
                print(f"[OK] {sym} H={H} step={step}: acc={acc:.3f}, logloss={logloss:.3f}, n={n}")
    
    # Write Excel file
    if all_rows:
        # Create DataFrame from results
        df_results = pd.DataFrame(all_rows, columns=header)
        
        # Write Excel file with all_runs sheet
        with pd.ExcelWriter(xlsx_name, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='all_runs', index=False)
            
            # Create best_cells sheet: best step per (symbol, horizon) by highest regime_acc
            best_cells = []
            for sym in df_results['symbol'].unique():
                for h in df_results['horizon'].unique():
                    subset = df_results[(df_results['symbol'] == sym) & (df_results['horizon'] == h)]
                    if not subset.empty:
                        # Sort by regime_acc (desc), then regime_logloss (asc - lower is better)
                        subset_sorted = subset.sort_values(
                            by=['regime_acc', 'regime_logloss'], 
                            ascending=[False, True], 
                            na_position='last'
                        )
                        best = subset_sorted.iloc[0]
                        best_cells.append(best.to_dict())
            
            df_best = pd.DataFrame(best_cells)
            df_best.to_excel(writer, sheet_name='best_cells', index=False)
        
        print(f"\nExcel saved: {xlsx_name}")
    else:
        # No results - write empty Excel file with just headers
        df_results = pd.DataFrame(columns=header)
        with pd.ExcelWriter(xlsx_name, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='all_runs', index=False)
            df_results.to_excel(writer, sheet_name='best_cells', index=False)
        print(f"\nExcel saved (no results): {xlsx_name}")
    
    # Summary statistics
    if all_rows:
        import numpy as np
        accs = [r["regime_acc"] for r in all_rows if not np.isnan(r["regime_acc"])]
        loglosses = [r["regime_logloss"] for r in all_rows if not np.isnan(r["regime_logloss"])]
        
        if accs:
            print(f"\nSummary:")
            print(f"- Total runs completed: {len(all_rows)}")
            print(f"- Avg regime accuracy: {np.mean(accs):.3f}")
            print(f"- Avg log loss: {np.mean(loglosses):.3f}")
    
    print("=" * 64)


if __name__ == "__main__":
    main()
