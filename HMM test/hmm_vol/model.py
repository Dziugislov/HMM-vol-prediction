import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from hmmlearn.hmm import GaussianHMM
import argparse
from hmm_vol.data import fetch_stock_data

def _coerce_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize DataFrame for time series analysis.
    
    Ensures data is in chronological order with proper datetime indexing.
    This is essential for HMM fitting since the model learns state transitions
    over time - mixed up dates would give incorrect patterns.
    
    Steps:
    1. Check if data exists
    2. Convert Date column to index if present
    3. Convert index to datetime format
    4. Sort by date for chronological order
    
    Example transformation:
    Input:  Date        Open    High    Low     Close
           2023-01-15  100     105     98      102
           2023-01-13  99      101     97      100
    
    Output:            Open    High    Low     Close
           2023-01-13  99      101     97      100
           2023-01-15  100     105     98      102
    """
    if df is None or df.empty:
        raise ValueError("No data available")
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

def get_data(symbol: str, years: int = 5) -> pd.DataFrame:
    """
    Retrieve stock data with robust fallback strategies.
    
    HMM fitting requires sufficient data points to reliably identify volatility
    regimes. This function ensures you get maximum available historical data
    with intelligent fallback strategies.
    
    Fallback Strategy:
    1. Try requested years (e.g., "20y" for 20 years)
    2. If that fails, try 15 years, then 10 years, then 5 years
    3. Final fallback: get maximum available data
    
    Example usage:
    get_data("AAPL", 20)  # Try to get 20 years, fallback to maximum available
    get_data("TSLA", 10)  # Try to get 10 years, fallback to maximum available
    
    The function handles various failure modes:
    - API rate limits
    - Invalid symbol names
    - Insufficient historical data
    - Network connectivity issues
    """
    # Small random delay to stagger connections when multiple subprocesses start
    import time
    import random
    time.sleep(random.uniform(0.1, 0.5))
    
    # Try requested years first
    try:
        df = fetch_stock_data(symbol, f"{years}y")
        return _coerce_df(df)
    except:
        pass
    
    # Fallback strategy: try decreasing years
    fallback_years = [15, 10, 5, 3, 1]
    for fallback_year in fallback_years:
        if fallback_year < years:  # Only try if less than requested
            try:
                df = fetch_stock_data(symbol, f"{fallback_year}y")
                return _coerce_df(df)
            except:
                continue
    
    # Final fallback: try maximum available
    try:
        df = fetch_stock_data(symbol, "max")
        return _coerce_df(df)
    except:
        raise ValueError(f"Could not fetch data for {symbol} with any fallback strategy")
    
    # Fallback strategies (commented out for cleaner approach):
    # try:
    #     # Secondary attempt: calculate exact date range
    #     end = datetime.today().date()
    #     start = end - timedelta(days=365*years + 10)  # +10 days buffer
    #     df = fetch_stock_data(symbol, str(start), str(end))
    #     return _coerce_df(df)
    # except:
    #     # Final fallback: just get 1 year of data
    #     df = fetch_stock_data(symbol, "1y")
    #     return _coerce_df(df)


def parkinson_daily_variance(df: pd.DataFrame) -> pd.Series:
    H,L = df["High"], df["Low"]
    return (1.0/(4.0*np.log(2.0))) * (np.log(H/L)**2)

def fit_hmm_logvar(logvar: pd.Series, n_states: int = 2, min_gap: float = 0.10):
    """
    Fit a Gaussian Hidden Markov Model to log-variance data to identify volatility regimes.
    
    This is the core function that finds hidden states (like "low vol" and "high vol" periods)
    in your volatility data. The HMM learns to identify when the market switches between
    different volatility regimes and how these regimes transition over time.
    
    The function returns all the HMM parameters needed for forecasting:
    - pi: Initial state probabilities
    - A: Transition matrix (how states change over time)
    - mu: Mean log-variance for each state
    - tau2: Variance of log-variance for each state
    - pT: Current state probabilities
    """
    # Step 1: Clean and prepare the data
    lv = logvar.dropna().astype(float).copy()  # Remove missing values, ensure numeric
    lq, uq = np.quantile(lv, [0.01, 0.99])     # Find 1st and 99th percentiles
    lv = lv.clip(lq, uq)                       # Remove extreme outliers (winsorization)
    X = lv.values.reshape(-1, 1)               # Convert to 2D array for HMM

    # Step 2: Create the HMM model
    model = GaussianHMM(
        n_components=n_states,      # Number of hidden states (e.g., 2 for low/high vol)
        covariance_type="diag",     # Use diagonal covariance (simpler, more stable)
        random_state=0,             # For reproducible results
        n_iter=200,                 # Maximum EM iterations
        tol=1e-4,                   # Convergence tolerance
        init_params="stc"           # We'll set means manually; EM updates them
    )
    
    # Step 3: Smart initialization using quantiles
    # This prevents the EM algorithm from getting stuck in poor local minima
    if n_states == 2:
        q = np.quantile(X.ravel(), [0.25, 0.75])  # Use 25th and 75th percentiles
    elif n_states == 3:
        q = np.quantile(X.ravel(), [0.10, 0.50, 0.90])  # Use 10th, 50th, 90th percentiles
    else:
        q = np.quantile(X.ravel(), np.linspace(0.1, 0.9, n_states))  # Spread evenly

    model.means_ = q.reshape(-1, 1)  # Set initial means
    model.fit(X)                     # Run EM algorithm to fit the model

    # Step 4: Extract fitted parameters
    means = model.means_.flatten()    # Get fitted means
    covs  = model.covars_.flatten()  # Get fitted variances
    order = np.argsort(means)        # Sort states by mean (low to high)
    means_sorted = means[order]      # Sorted means
    gap = np.min(np.diff(means_sorted)) if len(means_sorted) > 1 else float("inf")  # Check separation

    # Step 5: Quality check - ensure states are well-separated
    if gap < min_gap:
        raise ValueError(
            f"HMM state means too close on log-scale: min_gap={gap:.4f} < threshold={min_gap:.4f}. "
            f"Try fewer states (e.g., --states 2), more data, or stronger winsorization."
        )

    # Step 6: Extract and organize all HMM parameters
    pi  = model.startprob_[order]           # Initial state probabilities (sorted)
    A   = model.transmat_[order][:, order] # Transition matrix (sorted)
    mu  = means_sorted                      # Mean log-variance for each state
    tau2= covs[order]                      # Variance of log-variance for each state
    pT  = model.predict_proba(X)[-1][order] # Current state probabilities (sorted)

    # Step 7: Get state assignments for each observation
    inv = np.empty_like(order)              # Create inverse mapping
    inv[order] = np.arange(len(order))      # Map sorted indices back to original
    states = inv[model.predict(X)]         # Get state assignments for each day
    loglik = model.score(X)                 # Log-likelihood of the fitted model

    return dict(model=model, X=X, pi=pi, A=A, mu=mu, tau2=tau2, pT=pT, states=states, loglik=loglik)

def forecast_hday_vol(A: np.ndarray, mu: np.ndarray, tau2: np.ndarray, pT: np.ndarray, horizon: int) -> float:
    """
    Forecast volatility over a given horizon using HMM parameters.
    
    This function uses the HMM's transition probabilities to predict future state
    probabilities, then converts these to expected volatility. It's the core of
    volatility forecasting - predicting how volatile the market will be over the
    next N days.
    
    Mathematical approach:
    1. Convert log-variance means to variance means: m = exp(mu + 0.5*tau2)
    2. For each day in horizon:
       - Update state probabilities using transition matrix: pt = pt @ A
       - Add expected variance for this day: rv_sum += pt @ m
    3. Return square root of total variance (volatility)
    
    Example:
    A = [[0.8, 0.2], [0.3, 0.7]]  # Transition matrix
    mu = [-2.1, -1.5]             # Mean log-variance for each state
    tau2 = [0.3, 0.2]            # Variance of log-variance for each state
    pT = [0.4, 0.6]              # Current state probabilities
    horizon = 5                  # Forecast 5 days ahead
    
    Returns: 0.15 (15% volatility over 5 days)
    """
    # Step 1: Convert log-variance means to variance means
    m = np.exp(mu + 0.5*tau2)  # m = [0.12, 0.22] (variance means)
    pt = pT.copy()             # Start with current state probabilities
    rv_sum = 0.0               # Accumulate total variance
    
    # Step 2: For each day in horizon, update probabilities and add variance
    for _ in range(horizon):
        pt = pt @ A             # Update state probabilities using transition matrix
        rv_sum += float(pt @ m) # Add expected variance for this day
    
    # Step 3: Return volatility (square root of total variance)
    return float(np.sqrt(rv_sum))


def state_contributions(A: np.ndarray, mu: np.ndarray, tau2: np.ndarray, pT: np.ndarray, horizon: int) -> np.ndarray:
    """
    Calculate how much each hidden state contributes to the total forecasted volatility.
    
    This function shows you which volatility regimes are driving your forecast.
    It's useful for understanding whether your forecast is coming from low-volatility
    or high-volatility states, and how this changes over the forecast horizon.
    
    Mathematical approach:
    1. Convert log-variance means to variance means: m = exp(mu + 0.5*tau2)
    2. For each day in horizon:
       - Update state probabilities using transition matrix: pt = pt @ A
       - Add each state's contribution: contrib += pt * m
    3. Return contribution from each state
    
    Example:
    A = [[0.8, 0.2], [0.3, 0.7]]  # Transition matrix
    mu = [-2.1, -1.5]             # Mean log-variance for each state
    tau2 = [0.3, 0.2]            # Variance of log-variance for each state
    pT = [0.4, 0.6]              # Current state probabilities
    horizon = 5                  # Forecast 5 days ahead
    
    Returns: [0.08, 0.12] (State 0 contributes 0.08, State 1 contributes 0.12 to total variance)
    """
    # Step 1: Convert log-variance means to variance means
    m = np.exp(mu + 0.5*tau2)  # m = [0.12, 0.22] (variance means)
    pt = pT.copy()             # Start with current state probabilities
    contrib = np.zeros_like(m) # Initialize contributions for each state
    
    # Step 2: For each day in horizon, update probabilities and accumulate contributions
    for _ in range(horizon):
        pt = pt @ A             # Update state probabilities using transition matrix
        contrib += pt * m      # Add each state's contribution to total variance
    
    return contrib

def main():
    p = argparse.ArgumentParser()
    
    # Define command-line arguments with defaults
    p.add_argument("--symbol", type=str, default="AMZN")        # Stock symbol to analyze
    p.add_argument("--years", type=int, default=5)               # Years of historical data
    p.add_argument("--horizon", type=int, default=45)           # Forecast horizon in days
    p.add_argument("--no-annualize", action="store_true")        # Don't convert to annualized volatility
    p.add_argument("--states", type=int, default=2)              # Number of HMM states
    p.add_argument("--min_gap", type=float, default=0.10)       # Minimum gap between state means
    
    # Parse command-line arguments
    args = p.parse_args()

    # Try to retrieve stock data and calculate daily variance
    try:
        df = get_data(args.symbol, args.years)                  # Get stock data for specified symbol and years
        dv = parkinson_daily_variance(df)                      # Calculate daily variance using Parkinson estimator
    except Exception as e:
        # Handle any errors (network issues, invalid symbols, etc.)
        print(f"Error: {e}")
        print("Try a different symbol or wait for rate limits to reset")
        return

    # Convert raw variance to log-variance for HMM fitting
    logvar = np.log(dv + 1e-12)                                 # Add small constant to avoid log(0)
    res = fit_hmm_logvar(logvar, n_states=args.states, min_gap=args.min_gap)  # Fit HMM to log-variance data

    # Forecast volatility over the specified horizon
    volH = forecast_hday_vol(res["A"], res["mu"], res["tau2"], res["pT"], args.horizon)  # Forecast volatility
    
    # Convert to annualized volatility if requested
    ann = float(volH * np.sqrt(252.0/args.horizon)) if not args.no_annualize else None  # Annualize if not disabled
    
    # Calculate state contributions to the forecast
    contrib = state_contributions(res["A"], res["mu"], res["tau2"], res["pT"], args.horizon)  # Get state contributions

    # Display results
    print("Symbol:", args.symbol)                               # Show which stock was analyzed
    print("Observations:", int(res["X"].shape[0]))             # Show number of data points used
    print("LogLikelihood:", float(res["loglik"]))              # Show model fit quality
    print("pi:", np.round(res["pi"], 6))                       # Show initial state probabilities
    print("A:")                                                 # Show transition matrix
    for row in res["A"]:                                        # Print each row of transition matrix
        print(np.round(row, 6))
    print("means logvar (sorted):", np.round(res["mu"], 6))    # Show mean log-variance for each state
    print("stds  logvar (sorted):", np.round(np.sqrt(res["tau2"]), 6))  # Show standard deviation of log-variance
    unique, counts = np.unique(res["states"], return_counts=True)  # Count days in each state
    print("state counts:", {int(k): int(v) for k,v in zip(unique, counts)})  # Show state distribution
    if ann is not None:                                         # If annualized volatility was calculated
        print(f"vol_{args.horizon}d_annualized:", float(ann))  # Show annualized volatility
    else:                                                       # If annualization was disabled
        print(f"vol_{args.horizon}d:", float(volH))            # Show raw volatility
    print("contrib_by_state:", np.round(contrib, 6))           # Show state contributions to forecast

if __name__ == "__main__":
    main()
