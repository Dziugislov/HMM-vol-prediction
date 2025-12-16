#!/usr/bin/env python3
"""
HMM Volatility Validation Report Generator
==========================================

Generates validation reports from Excel results with heatmaps and summary tables.

Usage:
    python scripts/report_vol_validation.py                                    # Uses latest file in outputs/results/
    python scripts/report_vol_validation.py --xlsx outputs/results/validation_results_20250101_120000.xlsx
    python scripts/report_vol_validation.py --xlsx results.xlsx --out_dir outputs/report --title "Validation Report"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import glob
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def find_latest_validation_file(results_dir: str = "outputs/results") -> Optional[str]:
    """
    Find the latest validation_results_*.xlsx file in the results directory.
    
    Args:
        results_dir: Directory to search for Excel files
        
    Returns:
        Path to the latest Excel file, or None if no file found
    """
    if not os.path.exists(results_dir):
        return None
    
    # Find all validation_results_*.xlsx files
    pattern = os.path.join(results_dir, "validation_results_*.xlsx")
    files = glob.glob(pattern)
    
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
    
    df = pd.read_excel(xlsx_path, engine='openpyxl')
    
    # Validate required columns
    required_cols = [
        'symbol', 'horizon', 'step', 'excluded_pct', 'n_forecasts', 'n_kept'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for MAPE column (prefer mape_vol_mid, fallback to mape_mid)
    if 'mape_vol_mid' not in df.columns and 'mape_mid' not in df.columns:
        raise ValueError("Missing required column: mape_vol_mid or mape_mid")
    
    # Use mape_vol_mid if available, else fallback to mape_mid
    if 'mape_vol_mid' not in df.columns:
        df['mape_vol_mid'] = df['mape_mid']
    
    # Clean data
    df = df.dropna(subset=['mape_vol_mid', 'excluded_pct'])
    df = df[df['mape_vol_mid'] > 0]  # Remove invalid MAPE values
    
    return df

def get_best_step_per_symbol_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the best step for each (symbol, horizon) combination based on MAPE.
    
    Args:
        df: Validation results DataFrame
        
    Returns:
        DataFrame with best step per symbol/horizon
    """
    best_results = df.loc[df.groupby(['symbol', 'horizon'])['mape_vol_mid'].idxmin()]
    return best_results.reset_index(drop=True)

def create_mape_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """
    Create MAPE heatmap (symbols × horizons).
    
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
        values='mape_vol_mid', 
        index='symbol', 
        columns='horizon', 
        aggfunc='first'
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'MAPE'}, ax=ax)
    
    if step_fixed:
        ax.set_title(f'MAPE on Volatility (sqrt variance) - Step = {step_value}', fontsize=14, fontweight='bold')
    else:
        ax.set_title('MAPE on Volatility (sqrt variance) - Best Step per Symbol/Horizon', fontsize=14, fontweight='bold')
    ax.set_xlabel('Horizon (days)')
    ax.set_ylabel('Symbol')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'mape_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_band_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """
    Create terminal return band hit-rate heatmap (iron condor proxy) (symbols × horizons).
    
    Args:
        df: Validation results DataFrame
        output_dir: Output directory for plots
        
    Returns:
        Path to saved plot, or None if column missing
    """
    best_df = get_best_step_per_symbol_horizon(df)
    
    if 'in_band_share_kept' not in best_df.columns:
        return None
    
    # Check if step is fixed
    step_fixed = df['step'].nunique() == 1
    step_value = df['step'].iloc[0] if step_fixed else None
    
    # Create pivot table
    pivot = best_df.pivot_table(
        values='in_band_share_kept', 
        index='symbol', 
        columns='horizon', 
        aggfunc='first'
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn', 
                cbar_kws={'label': 'Terminal Return Band Hit-Rate (Iron Condor Proxy)'}, ax=ax, vmin=0, vmax=1)
    
    if step_fixed:
        ax.set_title(f'Terminal Return Band Hit-Rate (Iron Condor Proxy) - Step = {step_value}', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Terminal Return Band Hit-Rate (Iron Condor Proxy) - Best Step per Symbol/Horizon', fontsize=14, fontweight='bold')
    ax.set_xlabel('Horizon (days)')
    ax.set_ylabel('Symbol')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'band_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_excluded_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """
    Create excluded share heatmap (symbols × horizons).
    
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
        values='excluded_pct', 
        index='symbol', 
        columns='horizon', 
        aggfunc='first'
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlBu_r', 
                cbar_kws={'label': 'Excluded Share'}, ax=ax)
    
    if step_fixed:
        ax.set_title(f'Excluded Share - Step = {step_value}', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Excluded Share - Best Step per Symbol/Horizon', fontsize=14, fontweight='bold')
    ax.set_xlabel('Horizon (days)')
    ax.set_ylabel('Symbol')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'excluded_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_html_report(df: pd.DataFrame, output_dir: str, xlsx_path: str, 
                        title: str = "HMM Volatility Body-Validation Report") -> str:
    """
    Generate HTML report with heatmaps and summary table.
    
    Args:
        df: Validation results DataFrame
        output_dir: Output directory
        xlsx_path: Path to input Excel file
        title: Report title
        
    Returns:
        Path to generated HTML file
    """
    # Calculate key statistics
    best_df = get_best_step_per_symbol_horizon(df)
    
    # Check if step is fixed
    step_fixed = df['step'].nunique() == 1
    step_value = df['step'].iloc[0] if step_fixed else None
    step_mode = f"Fixed (step = {step_value})" if step_fixed else "Variable (best step per symbol/horizon)"
    
    # Key takeaways
    avg_mape = best_df['mape_vol_mid'].mean()
    avg_excluded = best_df['excluded_pct'].mean()
    avg_band_share = best_df['in_band_share_kept'].mean() if 'in_band_share_kept' in best_df.columns else float('nan')
    avg_band_share_str = f"{avg_band_share:.1%}" if not pd.isna(avg_band_share) else 'N/A'
    
    # Get config values
    states = df['states'].iloc[0] if 'states' in df.columns else 'N/A'
    train_frac = f"{df['train_frac'].iloc[0]:.1%}" if 'train_frac' in df.columns and pd.notna(df['train_frac'].iloc[0]) else 'N/A'
    ratio_cutoff = df['ratio_cutoff'].iloc[0] if 'ratio_cutoff' in df.columns else 'N/A'
    band_mult = best_df['band_mult'].iloc[0] if 'band_mult' in best_df.columns else (df['band_mult'].iloc[0] if 'band_mult' in df.columns else 'N/A')
    
    # Prepare summary table
    summary_cols = ['symbol', 'horizon', 'step', 'mape_vol_mid', 'excluded_pct', 'n_forecasts', 'n_kept']
    if 'in_band_share_kept' in best_df.columns:
        summary_cols.insert(5, 'in_band_share_kept')
    
    summary_df = best_df[[c for c in summary_cols if c in best_df.columns]].copy()
    summary_df = summary_df.sort_values(['symbol', 'horizon'])
    
    # Generate HTML table rows
    table_rows = ""
    for _, row in summary_df.iterrows():
        table_rows += "<tr>"
        table_rows += f"<td>{row['symbol']}</td>"
        table_rows += f"<td>{row['horizon']}</td>"
        table_rows += f"<td>{row['step']}</td>"
        table_rows += f"<td>{row['mape_vol_mid']:.3f}</td>"
        table_rows += f"<td>{row['excluded_pct']:.1%}</td>"
        if 'in_band_share_kept' in row:
            table_rows += f"<td>{row['in_band_share_kept']:.1%}</td>"
        table_rows += f"<td>{int(row['n_forecasts'])}</td>"
        table_rows += f"<td>{int(row['n_kept'])}</td>"
        table_rows += "</tr>\n"
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #2c3e50; }}
            .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
            .config {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; }}
            .takeaways {{ background-color: #d4edda; padding: 15px; border-radius: 5px; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="config">
            <h2>Configuration Recap</h2>
            <ul>
                <li><strong>Symbols tested:</strong> {len(df['symbol'].unique())}</li>
                <li><strong>Horizons:</strong> {sorted(df['horizon'].unique())}</li>
                <li><strong>Total runs:</strong> {len(df)}</li>
                <li><strong>States:</strong> {states}</li>
                <li><strong>Train fraction:</strong> {train_frac}</li>
                <li><strong>Ratio cutoff:</strong> {ratio_cutoff}</li>
                <li><strong>Band multiplier:</strong> {band_mult}</li>
                <li><strong>Step mode:</strong> {step_mode}</li>
                <li><strong>Years of data:</strong> {df['years'].iloc[0] if 'years' in df.columns else 'N/A'}</li>
            </ul>
        </div>
        
        <h2>Summary Table</h2>
        <p><strong>Averages:</strong> MAPE = {avg_mape:.3f}, Excluded = {avg_excluded:.1%}, Terminal Return Band Hit-Rate = {avg_band_share_str}</p>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Horizon</th>
                    <th>Step</th>
                    <th>MAPE</th>
                    <th>Excluded %</th>
                    {'<th>Terminal Return Band Hit-Rate</th>' if 'in_band_share_kept' in summary_df.columns else ''}
                    <th>N Forecasts</th>
                    <th>N Kept</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        
        <h2>Visualizations</h2>
        
        <h3>MAPE Heatmap</h3>
        <img src="mape_heatmap.png" alt="MAPE Heatmap">
        
    """
    
    if os.path.exists(os.path.join(output_dir, 'band_heatmap.png')):
        html_content += '<h3>Terminal Return Band Hit-Rate Heatmap (Iron Condor Proxy)</h3>\n<img src="band_heatmap.png" alt="Terminal Return Band Hit-Rate Heatmap">\n'
    
    html_content += """
        <h3>Excluded Share Heatmap</h3>
        <img src="excluded_heatmap.png" alt="Excluded Share Heatmap">
        
    </body>
    </html>
    """
    
    # Write HTML file
    html_path = os.path.join(output_dir, 'report.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return html_path

def main():
    """Main function for generating validation reports."""
    parser = argparse.ArgumentParser(description='HMM Volatility Validation Report Generator')
    parser.add_argument('--xlsx', type=str, default=None, 
                       help='Path to validation Excel (.xlsx) file (default: use latest in outputs/results/)')
    parser.add_argument('--out_dir', type=str, default='outputs/report',
                       help='Output directory for plots and report (default: outputs/report)')
    parser.add_argument('--title', type=str, default='HMM Volatility Body-Validation Report',
                       help='Report title (default: HMM Volatility Body-Validation Report)')
    
    args = parser.parse_args()
    
    try:
        # Determine Excel file path
        if args.xlsx is None:
            # Auto-detect latest file
            xlsx_path = find_latest_validation_file()
            if xlsx_path is None:
                print("Error: No validation_results_*.xlsx file found in outputs/results/")
                print("Please run run_vol_validation.py first, or specify --xlsx path manually")
                return 1
            print(f"Auto-detected latest file: {xlsx_path}")
        else:
            xlsx_path = args.xlsx
        
        # Create output directory
        os.makedirs(args.out_dir, exist_ok=True)
        
        # Load and validate data
        print(f"Loading data from {xlsx_path}...")
        df = load_and_validate_data(xlsx_path)
        print(f"Loaded {len(df)} validation results")
        
        # Find and copy summary.xlsx
        xlsx_dir = os.path.dirname(xlsx_path)
        summary_source = os.path.join(xlsx_dir, "summary.xlsx")
        if not os.path.exists(summary_source):
            # Try to find summary_*.xlsx with matching timestamp
            xlsx_basename = os.path.basename(xlsx_path)
            if "validation_results_" in xlsx_basename:
                ts_part = xlsx_basename.replace("validation_results_", "").replace(".xlsx", "")
                summary_pattern = os.path.join(xlsx_dir, f"summary_{ts_part}.xlsx")
                if os.path.exists(summary_pattern):
                    summary_source = summary_pattern
                else:
                    # Try to find any summary_*.xlsx in the directory
                    summary_files = glob.glob(os.path.join(xlsx_dir, "summary_*.xlsx"))
                    if summary_files:
                        summary_source = max(summary_files, key=os.path.getmtime)
        
        summary_dest = os.path.join(args.out_dir, "summary.xlsx")
        if os.path.exists(summary_source):
            shutil.copy2(summary_source, summary_dest)
            print(f"Summary Excel copied: {summary_dest}")
        else:
            # Generate summary.xlsx from loaded data
            best_df = get_best_step_per_symbol_horizon(df)
            summary_cols = ['symbol', 'horizon', 'step', 'mape_vol_mid', 'excluded_pct', 'n_forecasts', 'n_kept']
            if 'in_band_share_kept' in best_df.columns:
                summary_cols.insert(5, 'in_band_share_kept')
            best_cells = best_df[[c for c in summary_cols if c in best_df.columns]].copy()
            
            with pd.ExcelWriter(summary_dest, engine='openpyxl') as writer:
                best_cells.to_excel(writer, sheet_name='best_cells', index=False)
                df.to_excel(writer, sheet_name='all_runs', index=False)
            print(f"Summary Excel generated: {summary_dest}")
        
        # Generate all visualizations
        print("Generating visualizations...")
        
        # Heatmaps
        mape_heatmap_path = create_mape_heatmap(df, args.out_dir)
        band_heatmap_path = create_band_heatmap(df, args.out_dir)
        excluded_heatmap_path = create_excluded_heatmap(df, args.out_dir)
        heatmap_msg = f"Heatmaps: {mape_heatmap_path}, {excluded_heatmap_path}"
        if band_heatmap_path:
            heatmap_msg += f", {band_heatmap_path}"
        print(heatmap_msg)
        
        # Generate HTML report
        print("Generating HTML report...")
        html_path = generate_html_report(df, args.out_dir, xlsx_path, args.title)
        print(f"HTML report: {html_path}")
        
        print("\n" + "="*80)
        print("REPORT GENERATION COMPLETE")
        print(f"Output directory: {args.out_dir}")
        print(f"Files generated: mape_heatmap.png, band_heatmap.png, excluded_heatmap.png, report.html, summary.xlsx")
        print("="*80)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
