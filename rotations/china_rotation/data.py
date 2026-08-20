"""Price data for the Kechuang50-vs-Hongli-Dibo rotation index.

The two legs actually traded:
  KC50    588000.SS   STAR50 ETF (科创50) — Shanghai's Nasdaq-style growth/tech board
  DIBO    512890.SS   CSI Dividend Low Volatility 100 ETF (红利低波) — the steady leg

Both are onshore A-share ETFs; yfinance carries them under their Shanghai/
Shenzhen tickers (.SS / .SZ), unlike the US module which uses SEC-registered
ETFs.  Neither has deep history: 512890 lists Dec-2018, 588000 lists Sep-2020 —
the STAR board itself didn't exist before mid-2019.  There is no 27-year test
available here; every reading in this module should be discounted for that.

Cached per calendar day in .cache/, same convention as the rest of the project.
"""
from __future__ import annotations

import hashlib
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(".cache")

KC50 = "588000.SS"      # STAR50 ETF — traded growth/tech leg
DIBO = "512890.SS"      # Dividend Low Volatility 100 ETF — traded steady leg

# Longer-history proxies used only to warm up the z-scores before the traded
# ETFs existed — never held.  Same role ^NDX/^GSPC play in tech_rotation/.
KC50_PROXY = "159915.SZ"   # ChiNext ETF (创业板) — Shenzhen's growth board, since 2015.
                           # Not the same index as STAR50, but the same role: an
                           # onshore growth/tech board versus the blue-chip market.
CSI300 = "510300.SS"       # CSI300 ETF — broad market regime gate, since 2015
SEMIS = "512480.SS"        # China semiconductor ETF, since 2019 — leads the tech tape
BONDS = "511010.SS"        # China treasury bond ETF, since 2015 — rate-pressure proxy
SSE50 = "510050.SS"        # SSE50 (mega-cap) ETF, since 2015 — concentration proxy

TICKERS = [KC50, DIBO, KC50_PROXY, CSI300, SEMIS, BONDS, SSE50]

# Early enough that the proxy pair (ChiNext/CSI300) is warm well before STAR50
# lists in Sep-2020.
DEFAULT_START = "2015-01-01"


def _today_str() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _cache_path(tickers: list[str], start: str) -> Path:
    key = hashlib.sha1("|".join([*sorted(tickers), start]).encode()).hexdigest()[:16]
    return CACHE_DIR / f"china_rotation_px_{key}.pkl"


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


def fetch_prices(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START,
    refresh: bool = False,
) -> pd.DataFrame:
    """Split/dividend-adjusted daily closes, one column per ticker, ragged at the start."""
    tickers = tickers or TICKERS
    today = _today_str()
    path = _cache_path(tickers, start)
    if not refresh:
        cached = _cache_load(path, today)
        if cached is not None:
            print(f"  -> using today's cached prices ({path.name}; --refresh to refetch)")
            return cached

    import yfinance as yf

    print(f"  fetching {len(tickers)} China ETFs from yfinance since {start} ...")
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
