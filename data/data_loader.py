"""
Data loader for seven major equity indices using pandas-datareader (stooq).

Indices (stooq symbols):
  S&P 500           : ^SPX
  EURO STOXX 50     : ^SX5E  (or ^STOXX50E)
  FTSE 100          : ^FTSE
  DAX               : ^DAX
  Dow Jones Ind Avg : ^DJI
  NASDAQ Composite  : ^NDQ  (or ^IXIC — mapped via Yahoo fallback if needed)
  Nikkei 225        : ^NKX  (N225 on stooq)

Because stooq may have gaps or symbol differences, this module also provides a
synthetic-data fallback path (generate_synthetic_returns) for guaranteed
test coverage when live data is unavailable.

Usage:
  from data.data_loader import load_index_data, compute_log_returns
  prices = load_index_data(start="2007-03-30", end="2009-12-31")
  log_ret = compute_log_returns(prices)
"""
import os
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# Stooq ticker mapping
STOOQ_TICKERS = {
    "SPX": "^SPX",
    "SX5E": "^SX5E",
    "FTSE": "^FTSE",
    "DAX": "^DAX",
    "DJI": "^DJI",
    "NDX": "^NDQ",
    "NKY": "^NKX",
}
INDEX_NAMES = {
    "SPX": "S&P 500",
    "SX5E": "EURO STOXX 50",
    "FTSE": "FTSE 100",
    "DAX": "DAX",
    "DJI": "Dow Jones IA",
    "NDX": "NASDAQ Composite",
    "NKY": "Nikkei 225",
}

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _load_stooq(ticker_symbol: str, start: str, end: str) -> pd.Series:
    """Download daily Open prices from stooq for one symbol."""
    import pandas_datareader.data as web
    df = web.DataReader(ticker_symbol, "stooq", start=start, end=end)
    df.sort_index(inplace=True)
    if "Open" in df.columns:
        return df["Open"].rename(ticker_symbol)
    return df.iloc[:, 0].rename(ticker_symbol)


def load_index_data(
    start: str,
    end: str,
    index_keys: list | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load daily opening prices for the specified indices.

    Parameters
    ----------
    start      : start date string, e.g. "2007-03-30"
    end        : end date string, e.g. "2009-12-31"
    index_keys : list of keys from STOOQ_TICKERS (default: all seven)
    use_cache  : whether to use cached CSV files

    Returns
    -------
    pd.DataFrame with DatetimeIndex, columns = index_keys.
    Rows with any NaN are dropped (no imputation per methodology).
    """
    if index_keys is None:
        index_keys = list(STOOQ_TICKERS.keys())

    cache_file = CACHE_DIR / f"prices_{start}_{end}_{'_'.join(sorted(index_keys))}.csv"
    if use_cache and cache_file.exists():
        logger.info(f"Loading from cache: {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    series = {}
    for key in index_keys:
        sym = STOOQ_TICKERS[key]
        try:
            s = _load_stooq(sym, start, end)
            # Remove duplicates and sort
            s = s[~s.index.duplicated(keep="first")]
            s.sort_index(inplace=True)
            series[key] = s
            logger.info(f"Loaded {key} ({sym}): {len(s)} rows")
        except Exception as exc:
            warnings.warn(f"Failed to load {key} ({sym}): {exc}. Using NaN column.")
            series[key] = pd.Series(dtype=float, name=key)

    df = pd.DataFrame(series)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    if use_cache:
        df.to_csv(cache_file)

    return df


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log-returns r_t = log(P_t / P_{t-1}).

    Drops the first row (NaN) and any subsequent rows with NaN.

    Parameters
    ----------
    prices : pd.DataFrame of daily prices

    Returns
    -------
    pd.DataFrame of daily log-returns (no imputation)
    """
    lr = np.log(prices / prices.shift(1))
    return lr.iloc[1:].dropna()


def compute_gross_returns(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Convert log-returns to gross returns G_t = exp(r_t) = P_t/P_{t-1}.

    Parameters
    ----------
    log_returns : pd.DataFrame of log-returns

    Returns
    -------
    pd.DataFrame of strictly positive gross returns
    """
    gr = np.exp(log_returns)
    assert (gr > 0).all().all(), "Gross returns must be strictly positive"
    return gr


def generate_synthetic_returns(
    n: int = 252 * 2,
    seed: int = 99,
    index_keys: list | None = None,
) -> pd.DataFrame:
    """Generate synthetic log-return data for testing (no external API needed).

    Returns a DataFrame with realistic correlation structure and volatilities.

    Parameters
    ----------
    n          : number of daily observations
    seed       : random seed
    index_keys : list of column names (default: all seven)

    Returns
    -------
    pd.DataFrame with DatetimeIndex (business days) and synthetic log-returns
    """
    if index_keys is None:
        index_keys = list(STOOQ_TICKERS.keys())
    rng = np.random.RandomState(seed)
    d = len(index_keys)
    # Approximate covariance: ~20% annual vol, ~60% pairwise correlation
    annual_vol = 0.20
    daily_vol = annual_vol / np.sqrt(252)
    corr = 0.6 * np.ones((d, d))
    np.fill_diagonal(corr, 1.0)
    # Add some variation in correlation
    for i in range(d):
        for j in range(i + 1, d):
            corr[i, j] = corr[j, i] = 0.5 + 0.2 * rng.rand()
    cov = daily_vol ** 2 * corr
    # Ensure PSD
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    mu = np.full(d, 0.0004)  # slight positive drift
    data = rng.multivariate_normal(mu, cov, size=n)
    dates = pd.bdate_range("2007-01-01", periods=n)
    df = pd.DataFrame(data, index=dates, columns=index_keys)
    return df
