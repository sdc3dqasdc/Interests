"""Price data for the Nasdaq-vs-S&P rotation index.

Alpaca's free IEX feed is used elsewhere in this project, but this index needs
two decades of ETF history to see more than one tech cycle (the 2000 bust, the
2008 crash, 2013-2021 tech dominance, the 2022 rate shock).  yfinance is the
only free source here with that reach, and the handful of ETFs involved makes
one download cheap.

Cached per calendar day in .cache/, matching the convention the rest of the
project uses: a second run on the same day reuses the first run's fetch.
"""
from __future__ import annotations

import hashlib
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(".cache")

# What each series is here for.  Only TECH and STEADY are traded; the rest feed
# the index.
TECH = "QQQ"        # Nasdaq-100 — the growth/tech leg
STEADY = "SPY"      # S&P 500 — the steady leg
SEMIS = "SMH"       # semiconductors: leads tech, both up and down
LONG_BOND = "TLT"   # 20y+ Treasuries: proxy for the rate pressure on long-duration growth
INT_BOND = "IEF"    # 7-10y Treasuries: the duration-matched leg of the credit spread
CREDIT = "HYG"      # high yield: HYG/IEF is a risk-appetite spread proxy
EQUAL_WT = "RSP"    # equal-weight S&P: SPY/RSP measures mega-cap concentration

# The underlying indices, for signal history only — never traded.  QQQ lists in
# 1999 and the z-scores need two years of warm-up, which would push the first
# live reading to 2001 and cost the backtest the entire dot-com bust: the one
# episode that matters most for a "tech vs steady" rule.  ^NDX/^GSPC go back to
# 1985, so the ratio legs are warm before QQQ even exists.  These are price
# indices (no dividends); SPX yields ~1%/yr more than NDX, but every ratio leg
# is a deviation from its own trailing mean, which absorbs a slow constant drift.
TECH_IDX = "^NDX"
STEADY_IDX = "^GSPC"

TICKERS = [TECH, STEADY, SEMIS, LONG_BOND, INT_BOND, CREDIT, EQUAL_WT,
           TECH_IDX, STEADY_IDX]

# Early enough that every z-score is warm before QQQ starts trading in
# March 1999, so the backtest can include the dot-com bust.
DEFAULT_START = "1990-01-01"


def _today_str() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _cache_path(tickers: list[str], start: str) -> Path:
    key = hashlib.sha1("|".join([*sorted(tickers), start]).encode()).hexdigest()[:16]
    return CACHE_DIR / f"rotation_px_{key}.pkl"


def _cache_load(path: Path, today: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as handle:
            cached = pickle.load(handle)
        return cached["data"] if cached.get("date") == today else None
    except Exception:
        return None


def _cache_save(path: Path, today: str, data: pd.DataFrame) -> None:
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump({"date": today, "data": data}, handle)
    except Exception as exc:
        print(f"WARNING: could not write cache {path}: {exc}", file=sys.stderr)


FEAR_GREED_ARCHIVE = (
    "https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/main/datasets/cnn_fear_greed.csv"
)


def fetch_fear_greed(refresh: bool = False) -> pd.Series:
    """Daily CNN Fear & Greed score (0=extreme fear, 100=extreme greed).

    The archive only starts 2021-02-01 -- five and a half years, not the 27
    years the price series has. Any component built on this can only be
    checked against the recent regime, not the pre-2013 half, so it does not
    get the same both-halves confidence as the price-based components. Treat
    conclusions from it accordingly (see china_rotation's index for the same
    caveat on a short history)."""
    import csv

    today = _today_str()
    path = CACHE_DIR / "fear_greed.pkl"
    if not refresh:
        cached = _cache_load(path, today)
        if cached is not None:
            print(f"  -> using today's cached Fear & Greed history ({path.name}; --refresh to refetch)")
            return cached["fg"] if isinstance(cached, dict) else cached

    import requests

    print("  fetching CNN Fear & Greed history ...")
    resp = requests.get(FEAR_GREED_ARCHIVE, timeout=30)
    resp.raise_for_status()
    rows = {}
    for row in csv.DictReader(resp.text.splitlines()):
        try:
            rows[row["Date"]] = float(row["Fear Greed"])
        except (KeyError, ValueError):
            continue
    fg = pd.Series(rows, dtype=float)
    fg.index = pd.to_datetime(fg.index)
    fg = fg.sort_index()
    fg = fg[~fg.index.duplicated(keep="last")]

    _cache_save(path, today, fg)
    return fg


def fetch_prices(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START,
    refresh: bool = False,
) -> pd.DataFrame:
    """Split/dividend-adjusted daily closes, one column per ticker.

    Columns start ragged (HYG only lists in 2007) and stay that way — the index
    is built to run on whatever is available on each date rather than throwing
    away the 1999-2007 history to keep a rectangle."""
    tickers = tickers or TICKERS
    today = _today_str()
    path = _cache_path(tickers, start)
    if not refresh:
        cached = _cache_load(path, today)
        if cached is not None:
            print(f"  -> using today's cached prices ({path.name}; --refresh to refetch)")
            return cached

    import yfinance as yf

    print(f"  fetching {len(tickers)} ETFs from yfinance since {start} ...")
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    px = px.reindex(columns=tickers)
    px.index = pd.to_datetime(px.index)
    px = px.dropna(how="all").sort_index()

    missing = [t for t in tickers if px[t].notna().sum() == 0]
    if missing:
        raise SystemExit(f"no data returned for: {', '.join(missing)}")

    _cache_save(path, today, px)
    return px
