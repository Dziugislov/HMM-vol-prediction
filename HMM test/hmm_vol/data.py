# stock_data_fetcher.py
# IBKR-backed daily OHLCV fetcher with split-adjusted prices by default.

from ib_insync import *
import pandas as pd
from datetime import datetime, date
from typing import Optional, Dict, List, Union
import random
import threading
import atexit
import time
import os

# Start event loop for better asyncio handling
try:
    util.startLoop()
except:
    pass  # Already started or not needed

HOST = "127.0.0.1"

# Flexible platform selection via environment variable
# Set IB_PLATFORM to "gateway" (default) or "tws"
IB_PLATFORM = os.getenv("IB_PLATFORM", "gateway").lower()

# Auto-select port based on platform (live only)
if IB_PLATFORM == "gateway":
    PORT = 4001
elif IB_PLATFORM == "tws":
    PORT = 7496
else:
    # Fallback to Gateway live if invalid platform specified
    PORT = 4001
    print(f"Warning: Unknown IB_PLATFORM '{IB_PLATFORM}', defaulting to Gateway (port 4001)")

CLIENT_ID = 1
MARKET_DATA_TYPE = 3  # 1=live, 2=frozen, 3=delayed, 4=delayed-frozen (IRRELEVANT for historical data)
# Note: Always uses ADJUSTED_LAST (split-adjusted prices) to avoid artificial jumps from:
# - Stock splits (e.g., 2-for-1 split: price halves, shares double)
# - Stock dividends (e.g., 10% stock dividend)
# - Spin-offs (e.g., company splits into two)

# Persistent connection pool
_connection_lock = threading.Lock()
_ib_connection: Optional[IB] = None
_connection_client_id: int = CLIENT_ID


def _get_connection(max_retries: int = 1, retry_delay: float = 5.0) -> IB:
    """Get or create IBKR connection. Reduced retries for IB Gateway stability."""
    global _ib_connection, _connection_client_id

    with _connection_lock:
        if _ib_connection is None or not _ib_connection.isConnected():
            # Skip socket verification - let IB connection handle it directly
            # _verify_gateway_connection()  # Commented out - can interfere with Gateway

            last_error = None
            for attempt in range(max_retries):
                try:
                    _ib_connection = IB()
                    _connection_client_id = CLIENT_ID

                    _ib_connection.connect(
                        HOST,
                        PORT,
                        clientId=_connection_client_id,
                        timeout=60  # Increased timeout for Gateway
                    )

                    _ib_connection.reqMarketDataType(MARKET_DATA_TYPE)
                    time.sleep(0.5)  # Brief pause after connection
                    return _ib_connection

                except Exception as e:
                    last_error = e
                    _ib_connection = None
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        platform_name = "IB Gateway" if IB_PLATFORM == "gateway" else "TWS"
                        raise ConnectionError(
                            f"Failed to connect to IBKR {platform_name} on {HOST}:{PORT}: {e}\n"
                            f"Make sure {platform_name} is running and restart it if needed.\n"
                            f"Current platform: IB_PLATFORM={IB_PLATFORM}"
                        ) from e

        return _ib_connection


def _close_connection():
    """
    Close the persistent connection (called on exit).
    """
    global _ib_connection
    with _connection_lock:
        if _ib_connection is not None and _ib_connection.isConnected():
            try:
                _ib_connection.disconnect()
            except:
                pass
            _ib_connection = None


# Register cleanup on exit
atexit.register(_close_connection)


def test_connection() -> bool:
    """
    Test if IBKR Gateway is accessible.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        ib = _get_connection(max_retries=1, retry_delay=1.0)
        return ib.isConnected()
    except:
        return False


def _verify_gateway_connection():
    """
    Verify Gateway connection before proceeding.
    Note: Socket test may fail even if port is listening if API is not enabled.
    This is just a quick check - actual IB connection will provide better error.
    """
    import socket
    # Quick socket test (informational only - may fail even if port is listening)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        result = sock.connect_ex((HOST, PORT))
        sock.close()
        # Don't raise error here - let IB connection attempt provide better diagnostics
    except:
        sock.close()
        # Don't raise - let IB connection handle it

def _ib_end(end: Optional[Union[str, date]]) -> str:
    """
    Helper: Convert end date to IBKR format string.
    
    Takes a date (string like "2023-12-31" or date object) and converts it to 
    IBKR's required format: "YYYYMMDD HH:MM:SS"

    """
    if not end:
        return ""  # If no end date provided, return empty string
    if isinstance(end, str):
        # If input is a string like "2023-12-31", parse it
        end_dt = datetime.fromisoformat(end)
    elif isinstance(end, date):
        # If input is a date object, convert to datetime (with time set to midnight)
        end_dt = datetime.combine(end, datetime.min.time())
    else:
        # If input is neither string nor date, raise error
        raise ValueError("end must be ISO string or date")
    # Convert to IBKR format: "20231231 00:00:00"
    return end_dt.strftime("%Y%m%d %H:%M:%S")

def _duration_str(period_or_start: str, end: Optional[str]) -> (str, str):
    """
    Helper: Convert user input to IBKR format for duration and end date.
    
    Returns a tuple (duration, endDateTime) in IBKR format.
    
    Function signature:
    - -> (str, str) = Returns a TUPLE (ordered pair) of two strings
      * First string: duration (e.g., "5 Y" or "30 D")
      * Second string: end date in IBKR format (e.g., "20231231 00:00:00" or "")
    
    Two modes:
    A) period_or_start = "5y" (no end date) → Returns ("5 Y", "")
    B) period_or_start = "2020-01-01", end = "2023-12-31" → Returns ("30 D", "20231231 00:00:00")
    """
    # Case A: "5y" style (simple duration, no specific end date)
    p = str(period_or_start).strip().lower()  # Convert to lowercase, remove spaces
    if p.endswith("y") and p[:-1].isdigit() and end is None:
        # Check if input is like "5y" (ends with 'y', digits before 'y', no end date)
        years = int(p[:-1])  # Extract the number (e.g., "5y" → 5)
        return f"{years} Y", ""  # Return ("5 Y", "") - IBKR format for 5 years, no end date
    
    # Case B: explicit ISO start/end dates (e.g., "2020-01-01", "2023-12-31")
    if end is not None:
        # Parse start and end dates
        s_dt = datetime.fromisoformat(period_or_start)  # "2020-01-01" → datetime object
        e_dt = datetime.fromisoformat(end)  # "2023-12-31" → datetime object
        
        # Calculate number of days between start and end
        days = max((e_dt.date() - s_dt.date()).days, 1)  # At least 1 day
        
        # For adjusted data, avoid end date restrictions by fetching more data
        # We'll filter it later in the DataFrame
        if days > 365:
            # More than 1 year: use years format
            years = max(1, days // 365) + 1  # Add buffer year (e.g., 800 days → 3 years)
            return f"{years} Y", ""  # No end date to avoid ADJUSTED_LAST restriction
        else:
            # Less than 1 year: use days format
            return f"{days} D", _ib_end(end)  # Return ("30 D", "20231231 00:00:00")
    
    # Fallback: raise error if input doesn't match either pattern
    raise ValueError(
        f"Invalid input: period_or_start='{period_or_start}', end={end}. "
        f"Expected either: (1) '5y' style with no end date, or (2) ISO date range with end date."
    )

def _to_ohlcv_df(bars) -> Optional[pd.DataFrame]:
    """
    Helper: Convert IBKR bars data to clean pandas DataFrame.
    
    Takes raw bar data from IBKR and converts it to a standard OHLCV DataFrame
    with proper date index and capitalized column names.
    
    Input: IBKR bars object (list of bar objects)
    Output: DataFrame with columns [Open, High, Low, Close, Volume] indexed by date
    
    Steps:
    1. Convert bars to DataFrame using ib_insync utility
    2. Check if data exists
    3. Convert date column to datetime
    4. Set date as index
    5. Rename columns to capitalized standard format
    6. Select only OHLCV columns
    7. Sort by date (oldest to newest)
    """
    # Step 1: Convert IBKR bars object to pandas DataFrame
    df = util.df(bars)
    
    # Step 2: Check if we got valid data
    if df is None or df.empty:
        return None
    
    # Step 3: Convert date column to proper datetime format
    df["date"] = pd.to_datetime(df["date"])
    
    # Step 4: Set date as index and rename columns to standard format
    # IBKR returns lowercase column names, we want capitalized
    df = df.set_index("date").rename(
        columns={
            "open": "Open",
            "high": "High", 
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }
    )[["Open", "High", "Low", "Close", "Volume"]]  # Select only OHLCV columns
    
    # Step 5: Sort by date (oldest to newest)
    return df.sort_index()

def _fetch_ib(symbol: str, duration: str, endDT: str, barSize: str, rth: bool) -> Optional[pd.DataFrame]:
    """
    Helper: Fetch historical data from IBKR using persistent connection.
    
    Uses a persistent connection pool to avoid creating new connections for each fetch.
    """
    # First attempt
    try:
        ib = _get_connection()
        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=endDT,
            durationStr=duration,
            barSizeSetting=barSize,
            whatToShow="ADJUSTED_LAST",  # Always use split-adjusted prices
            useRTH=rth,
            formatDate=2
        )
        df = _to_ohlcv_df(bars)
        return df
    except ConnectionError:
        # Don't retry connection errors - they indicate Gateway is not available
        raise
    except Exception as e:
        # For other errors (data fetch issues), reset connection and retry once
        global _ib_connection
        with _connection_lock:
            if _ib_connection is not None:
                try:
                    _ib_connection.disconnect()
                except:
                    pass
                _ib_connection = None
        
        # Single retry attempt for non-connection errors
        try:
            ib = _get_connection()
            contract = Stock(symbol, "SMART", "USD")
            ib.qualifyContracts(contract)
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=endDT,
                durationStr=duration,
                barSizeSetting=barSize,
                whatToShow="ADJUSTED_LAST",
                useRTH=rth,
                formatDate=2
            )
            df = _to_ohlcv_df(bars)
            return df
        except ConnectionError:
            # Re-raise connection errors without wrapping
            raise
        except Exception as retry_error:
            raise RuntimeError(f"Failed to fetch data for {symbol} after retry: {retry_error}") from retry_error

def fetch_stock_data(
    ticker_symbol: str,
    period_or_start: str = "1y",
    end: Optional[str] = None,
    *,
    rth: bool = True,
    barSize: str = "1 day"
) -> Optional[pd.DataFrame]:
    """
    IBKR-backed fetch with split-adjusted prices (ADJUSTED_LAST).
    
    Compatible with existing calls in your codebase:
      - fetch_stock_data("AAPL", "5y")
      - fetch_stock_data("AAPL", "YYYY-MM-DD", "YYYY-MM-DD")
    
    Returns Date-indexed OHLCV DataFrame with split-adjusted prices.
    """
    duration, endDT = _duration_str(period_or_start, end)
    df = _fetch_ib(ticker_symbol, duration, endDT, barSize, rth)
    
    # Filter to requested date range if we had to fetch more data to avoid IBKR restrictions
    if df is not None and end is not None:
        try:
            start_date = datetime.fromisoformat(period_or_start).date()
            end_date = datetime.fromisoformat(end).date()
            df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
        except ValueError:
            # If date parsing fails, return the data as-is
            pass
    
    return df

def fetch_multiple_stocks(
    ticker_symbols: List[str],
    period: str = "1y",
    *,
    rth: bool = True,
    barSize: str = "1 day"
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Fetch historical data for multiple stocks with split-adjusted prices.
    
    Returns a dictionary mapping ticker symbols to their OHLCV DataFrames.
    """
    out = {}
    for t in ticker_symbols:
        out[t] = fetch_stock_data(
            t, period,
            rth=rth, barSize=barSize
        )
    return out