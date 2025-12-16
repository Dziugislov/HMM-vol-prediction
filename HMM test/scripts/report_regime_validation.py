#!/usr/bin/env python3
"""
HMM Regime Validation Report Generator
=======================================

Generates validation reports from Excel results with heatmaps and summary tables.

Usage:
    python scripts/report_regime_validation.py                                    # Uses latest file in outputs/results/
    python scripts/report_regime_validation.py --xlsx outputs/results/regime_summary_20250101_120000.xlsx
    python scripts/report_regime_validation.py --xlsx results.xlsx --out_dir outputs/report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import glob
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def find_latest_regime_file(results_dir: str = "outputs/results") -> Optional[str]:
    """
    Find the latest regime_summary_*.xlsx file in the results directory.
    
    Args:
        results_dir: Directory to search for Excel files
        
    Returns:
        Path to the latest Excel file, or None if no file found
    """
    if not os.path.exists(results_dir):
        return None
    
    # Find all regime_summary*.xlsx files (with or without timestamp)
    pattern1 = os.path.join(results_dir, "regime_summary_*.xlsx")
    pattern2 = os.path.join(results_dir, "regime_summary.xlsx")
    files = glob.glob(pattern1) + ([pattern2] if os.path.exists(pattern2) else [])
    
    if not files:
        return None
    
    # Return the most recently modified file
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def load_and_validate_data(xlsx_path: str) -> pd.DataFrame:
    """
    Load and validate the validation Excel data.
    
    Args:
        xlsx_path: Path to the validation Excel (.xlsx) file
        
    Returns:
        DataFrame with validation results
        
    Raises:
        FileNotFoundError: If Excel file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")
    
    # Try to read from all_runs sheet, fallback to first sheet
    try:
        df = pd.read_excel(xlsx_path, sheet_name='all_runs', engine='openpyxl')
    except:
        df = pd.read_excel(xlsx_path, engine='openpyxl')
    
    # Validate required columns
    required_cols = [
        'symbol', 'horizon', 'step', 'regime_acc', 'regime_logloss', 'n_forecasts'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean data
    df = df.dropna(subset=['regime_acc', 'regime_logloss'])
    df = df[df['n_forecasts'] > 0]  # Remove invalid forecasts
    
    return df

def get_best_step_per_symbol_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the best step for each (symbol, horizon) combination based on regime_acc.
    
    Args:
        df: Validation results DataFrame
        
    Returns:
        DataFrame with best step per symbol/horizon
    """
    # Sort by regime_acc (desc), then regime_logloss (asc - lower is better)
    best_results = df.sort_values(
        by=['regime_acc', 'regime_logloss'], 
        ascending=[False, True], 
        na_position='last'
    ).groupby(['symbol', 'horizon']).first().reset_index()
    
    return best_results

def create_acc_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """
    Create regime accuracy heatmap (symbols × horizons).
    
    Args:
        df: Validation results DataFrame
        output_dir: Output directory for plots
        
    Returns:
        Path to saved plot
    """
    best_df = get_best_step_per_symbol_horizon(df)
    
    # Check if step is fixed
    step_fixed = df['step'].nunique() == 1
    step_value = df['step'].iloc[0] if step_fixed else None
    
    # Create pivot table
    pivot = best_df.pivot_table(
        values='regime_acc', 
        index='symbol', 
        columns='horizon', 
        aggfunc='first'
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Accuracy'}, ax=ax, vmin=0, vmax=1)
    
    if step_fixed:
        ax.set_title(f'Regime Accuracy - Step = {step_value}', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Regime Accuracy - Best Step per Symbol/Horizon', fontsize=14, fontweight='bold')
    ax.set_xlabel('Horizon (days)')
    ax.set_ylabel('Symbol')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'regime_acc_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_logloss_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """
    Create log-loss heatmap (symbols × horizons). Lower is better.
    
    Args:
        df: Validation results DataFrame
        output_dir: Output directory for plots
        
    Returns:
        Path to saved plot
    """
    best_df = get_best_step_per_symbol_horizon(df)
    
    # Check if step is fixed
    step_fixed = df['step'].nunique() == 1
    step_value = df['step'].iloc[0] if step_fixed else None
    
    # Create pivot table
    pivot = best_df.pivot_table(
        values='regime_logloss', 
        index='symbol', 
        columns='horizon', 
        aggfunc='first'
    )
    
    # Create heatmap (reversed colormap: green=low/good, red=high/bad)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Log Loss (lower = better)'}, ax=ax)
    
    if step_fixed:
        ax.set_title(f'Regime Log Loss (Cross-Entropy) - Step = {step_value}', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Regime Log Loss (Cross-Entropy) - Best Step per Symbol/Horizon', fontsize=14, fontweight='bold')
    ax.set_xlabel('Horizon (days)')
    ax.set_ylabel('Symbol')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'regime_logloss_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def generate_html_report(df: pd.DataFrame, acc_path: str, logloss_path: str, 
                         output_dir: str, xlsx_path: str) -> str:
    """
    Generate HTML report with heatmaps and summary table.
    
    Args:
        df: Validation results DataFrame
        acc_path: Path to accuracy heatmap
        logloss_path: Path to log-loss heatmap
        output_dir: Output directory
        xlsx_path: Path to source Excel file
        
    Returns:
        Path to saved HTML file
    """
    best_df = get_best_step_per_symbol_horizon(df)
    
    # Get configuration from data
    states = df['states'].iloc[0] if 'states' in df.columns else 'N/A'
    train_frac = df['train_frac'].iloc[0] if 'train_frac' in df.columns else 'N/A'
    step_fixed = df['step'].nunique() == 1
    step_mode = f"Step = {df['step'].iloc[0]}" if step_fixed else "Best step per symbol/horizon"
    seed = df['seed'].iloc[0] if 'seed' in df.columns and not pd.isna(df['seed'].iloc[0]) else 'N/A'
    
    # Compute averages
    avg_acc = best_df['regime_acc'].mean()
    avg_logloss = best_df['regime_logloss'].mean()
    
    # Get top 10 symbols by average regime_acc
    symbol_avg = best_df.groupby('symbol')['regime_acc'].mean().sort_values(ascending=False)
    top_symbols = symbol_avg.head(10)
    
    # Create summary table rows
    table_rows = ""
    for idx, row in best_df.iterrows():
        table_rows += f"""
        <tr>
            <td>{row['symbol']}</td>
            <td>{int(row['horizon'])}</td>
            <td>{int(row['step'])}</td>
            <td>{row['regime_acc']:.3f}</td>
            <td>{row['regime_logloss']:.3f}</td>
            <td>{int(row['n_forecasts'])}</td>
        </tr>
        """
    
    # Get relative paths for images
    acc_filename = os.path.basename(acc_path)
    logloss_filename = os.path.basename(logloss_path)
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>HMM Regime Validation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .config {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .config p {{
            margin: 5px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .summary-stats {{
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>HMM Regime Validation Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="config">
            <h2>Configuration Recap</h2>
            <p><strong>States:</strong> {states}</p>
            <p><strong>Train Fraction:</strong> {train_frac}</p>
            <p><strong>Step Mode:</strong> {step_mode}</p>
            <p><strong>Seed:</strong> {seed}</p>
            <p><strong>Source File:</strong> {os.path.basename(xlsx_path)}</p>
        </div>
        
        <div class="summary-stats">
            <h2>Summary Statistics</h2>
            <p><strong>Average Regime Accuracy:</strong> {avg_acc:.3f}</p>
            <p><strong>Average Log Loss:</strong> {avg_logloss:.3f} (lower = better)</p>
            <p><strong>Total Runs:</strong> {len(df)}</p>
            <p><strong>Unique Symbols:</strong> {df['symbol'].nunique()}</p>
            <p><strong>Unique Horizons:</strong> {sorted(df['horizon'].unique())}</p>
        </div>
        
        <h2>Top 10 Symbols by Average Accuracy</h2>
        <ul>
    """
    
    for sym, acc in top_symbols.items():
        html_content += f"<li><strong>{sym}</strong>: {acc:.3f}</li>\n"
    
    html_content += f"""
        </ul>
        
        <h2>Regime Accuracy Heatmap</h2>
        <img src="{acc_filename}" alt="Regime Accuracy Heatmap">
        
        <h2>Regime Log Loss Heatmap</h2>
        <img src="{logloss_filename}" alt="Regime Log Loss Heatmap">
        
        <h2>Summary Table (Best Step per Symbol/Horizon)</h2>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Horizon</th>
                    <th>Step</th>
                    <th>Regime Accuracy</th>
                    <th>Log Loss</th>
                    <th>N Forecasts</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
</body>
</html>
    """
    
    output_path = os.path.join(output_dir, 'regime_report.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path

def main():
    """Main function for generating regime validation report."""
    parser = argparse.ArgumentParser(description='HMM Regime Validation Report Generator')
    parser.add_argument('--xlsx', type=str, default=None,
                       help='Path to regime_summary Excel file (default: latest in outputs/results/)')
    parser.add_argument('--out_dir', type=str, default=None,
                       help='Output directory for report (default: same as Excel file directory)')
    
    args = parser.parse_args()
    
    # Find Excel file
    if args.xlsx:
        xlsx_path = args.xlsx
    else:
        xlsx_path = find_latest_regime_file()
        if not xlsx_path:
            print("Error: No regime_summary Excel file found. Please specify --xlsx")
            return 1
        print(f"Using latest file: {xlsx_path}")
    
    # Set output directory
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.dirname(xlsx_path)
        if not out_dir:
            out_dir = "outputs/report"
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {xlsx_path}...")
    df = load_and_validate_data(xlsx_path)
    print(f"Loaded {len(df)} validation runs")
    
    # Generate heatmaps
    print("Generating heatmaps...")
    acc_path = create_acc_heatmap(df, out_dir)
    logloss_path = create_logloss_heatmap(df, out_dir)
    print(f"  - {os.path.basename(acc_path)}")
    print(f"  - {os.path.basename(logloss_path)}")
    
    # Generate HTML report
    print("Generating HTML report...")
    html_path = generate_html_report(df, acc_path, logloss_path, out_dir, xlsx_path)
    print(f"  - {os.path.basename(html_path)}")
    
    # Copy Excel file to report directory as regime_summary.xlsx
    import shutil
    summary_dest = os.path.join(out_dir, "regime_summary.xlsx")
    if xlsx_path != summary_dest:
        shutil.copy2(xlsx_path, summary_dest)
        print(f"  - {os.path.basename(summary_dest)}")
    
    print(f"\nReport generated successfully in: {out_dir}")
    print(f"Files created:")
    print(f"  - {os.path.basename(acc_path)}")
    print(f"  - {os.path.basename(logloss_path)}")
    print(f"  - {os.path.basename(html_path)}")
    if xlsx_path != summary_dest:
        print(f"  - {os.path.basename(summary_dest)}")
    
    return 0

if __name__ == "__main__":
    exit(main())

