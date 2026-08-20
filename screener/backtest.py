#!/usr/bin/env python3
"""Backtest the 10-week swing screen over historical price data.

Mirrors short_term_screener_alpaca.py: quality names in confirmed long-term
uptrends, entered on consolidation rather than a spike, held ~50 trading days,
with new entries suppressed while SPY is below its 200-day SMA.

Price data comes from Alpaca (IEX feed, free tier) for fast, rate-limit-free
bulk daily bars; fundamentals (market cap, ROE, FCF) still come from yfinance,
which Alpaca does not provide.

Prerequisites:
    pip install alpaca-py yfinance pandas numpy
    export ALPACA_API_KEY="your_key"
    export ALPACA_SECRET_KEY="your_secret"
    (or pass --alpaca-key / --alpaca-secret)

LIMITATIONS:
- Fundamentals (market cap, ROE, FCF) are fetched *once* as of today and
  treated as static, so results carry survivorship and fundamental-lookahead
  bias — read any headline "alpha" figure with heavy skepticism.
- The news-sentiment gate and the earnings blackout are NOT simulated: free
  Yahoo data has no point-in-time headline or earnings-date history.  This
  script tests the technical + liquidity + quality-backstop rules only.
- Entries fill at the NEXT session's open after the screen fires, so there is
  no same-bar lookahead; slippage is approximated and commissions, partial
  fills, and market-impact are not modelled.

Usage example:
    python3 backtest.py --tickers sp500.txt --start 2022-01-01 --end 2024-12-31 \
        --hold-days 20 --rebalance-freq 10 --initial-cash 100000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_report as rr


def finite_number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def read_tickers(path: Path) -> list[str]:
    tickers = []
    for line in path.read_text().splitlines():
        ticker = line.split("#", 1)[0].strip().upper()
        if ticker:
            tickers.append(ticker)
    return list(dict.fromkeys(tickers))


# ---------------------------------------------------------------------------
# Alpaca data client + bulk price fetch (IEX feed, free tier)
# ---------------------------------------------------------------------------
def _load_dotenv(path: str = ".env") -> None:
    """Populate os.environ from a .env file for any keys not already set.

    Already-exported real environment variables win, so this only fills gaps.
    No dependency on python-dotenv; missing file is a silent no-op."""
    try:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


def _get_alpaca_client(api_key: str | None, secret_key: str | None):
    from alpaca.data.historical.stock import StockHistoricalDataClient
    _load_dotenv()
    key = api_key or os.environ.get("ALPACA_API_KEY")
    secret = secret_key or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit(
            "Alpaca API key and secret required. Set ALPACA_API_KEY / ALPACA_SECRET_KEY "
            "(env vars or a .env file), or pass --alpaca-key / --alpaca-secret."
        )
    return StockHistoricalDataClient(key, secret)


_OHLCV_RENAME = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}


def _latest_contiguous_segment(index: pd.DatetimeIndex, max_gap_days: int = 20) -> pd.DatetimeIndex:
    """Keep only the most recent run of bars with no gap larger than
    max_gap_days between consecutive sessions.

    A symbol later reused for an unrelated company (e.g. LB: L Brands until
    its August 2021 rename to BBWI, then reassigned to a different listing)
    shows up as a years-long gap in an otherwise near-daily series.  Without
    this, a "20 trading day" hold measured by row position can silently span
    two unrelated companies and years of calendar time.  20 calendar days is
    comfortably above any real market closure (long weekends/holidays run
    3-5 days) and comfortably below a delisting-driven gap."""
    if len(index) < 2:
        return index
    gaps = index.to_series().diff().dt.days
    breaks = gaps[gaps > max_gap_days].index
    return index[index >= breaks[-1]] if len(breaks) else index


def _collect_bars(df: pd.DataFrame, syms: list[str], result: dict[str, pd.DataFrame]) -> int:
    """Pull each symbol's frame out of a multi-indexed (symbol, timestamp) result.
    Returns how many symbols were trimmed for a symbol-reuse gap."""
    trimmed = 0
    for sym in syms:
        try:
            sym_df = df.loc[sym].copy()
        except KeyError:
            continue
        if sym_df.empty or len(sym_df) < 30:
            continue
        idx = pd.to_datetime(sym_df.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        sym_df.index = idx.normalize()
        kept = _latest_contiguous_segment(sym_df.index)
        if len(kept) < len(sym_df.index):
            sym_df = sym_df.loc[kept]
            trimmed += 1
        if len(sym_df) < 30:
            continue
        result[sym] = sym_df.rename(columns=_OHLCV_RENAME)
    return trimmed


def _fetch_bars_salvage(client, syms: list[str], start, end, result: dict[str, pd.DataFrame],
                        trimmed: list[int]) -> int:
    """Fetch one batch; if Alpaca rejects it for a bad symbol, split and retry so
    a single invalid symbol (share class, warrant, unit) cannot sink the batch.
    Recurses down to single symbols.  Returns the count of symbols dropped;
    accumulates the symbol-reuse-gap trim count into trimmed[0]."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    req = StockBarsRequest(
        symbol_or_symbols=syms,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
        adjustment="all",
    )
    try:
        df = client.get_stock_bars(req).df
        if df is not None and not df.empty:
            trimmed[0] += _collect_bars(df, syms, result)
        return 0
    except Exception:
        if len(syms) == 1:
            return 1
        mid = len(syms) // 2
        time.sleep(0.1)
        dropped = _fetch_bars_salvage(client, syms[:mid], start, end, result, trimmed)
        time.sleep(0.1)
        dropped += _fetch_bars_salvage(client, syms[mid:], start, end, result, trimmed)
        return dropped


# ---------------------------------------------------------------------------
# Day-scoped local cache — shared by every tool in this pipeline (backtest.py,
# short_term_screener_alpaca.py, roi_tester.py, alpha_lab.py, win_rate_tester.py)
# so Alpaca bars and fundamentals fetched once today are reused by whichever
# tool runs next, instead of every tool refetching independently.  Keyed by
# date + inputs, so a different universe/range on the same day still refetches;
# the key naturally expires at midnight UTC since the date is part of it.
# ---------------------------------------------------------------------------
CACHE_DIR = Path(".cache")


def _today_str() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _cache_key(*parts: Any) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _cache_load(path: Path, today: str):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as handle:
            cached = pickle.load(handle)
        return cached["data"] if cached.get("date") == today else None
    except Exception:
        return None


def prune_cache(today: str) -> int:
    """Delete cache entries from earlier days.  Returns how many were removed.

    A full-universe bar cache is 100-300MB, and the key includes the exact
    ticker list, so every universe refresh or date-range change mints a new
    file.  Without pruning the directory grows without bound — it reached
    ~1GB in a single day of experimentation."""
    removed = 0
    if not CACHE_DIR.exists():
        return 0
    for path in CACHE_DIR.glob("*.pkl"):
        try:
            with open(path, "rb") as handle:
                stale = pickle.load(handle).get("date") != today
        except Exception:
            stale = True          # unreadable/corrupt entry is not worth keeping
        if stale:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def _cache_save(path: Path, today: str, data: Any) -> None:
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        removed = prune_cache(today)
        if removed:
            print(f"  -> pruned {removed} cache entries from earlier days")
        with open(path, "wb") as handle:
            pickle.dump({"date": today, "data": data}, handle)
    except Exception as exc:
        print(f"WARNING: could not write cache {path}: {exc}", file=sys.stderr)


def fetch_price_data_cached(tickers: list[str], client, start: datetime, end: datetime,
                            refresh: bool = False) -> dict[str, pd.DataFrame]:
    """fetch_price_data, reused for the rest of today once fetched."""
    today = _today_str()
    # "ohlcv-cap" distinguishes this from the raw lowercase-column cache used
    # by short_term_screener_alpaca.py's fetch_alpaca_bars_cached — same bars,
    # different column casing, so they must not collide on the same cache key.
    key = _cache_key("bars", "ohlcv-cap", today, sorted(tickers), start.date(), end.date() if end else None)
    path = CACHE_DIR / f"bars_{key}.pkl"
    if not refresh:
        cached = _cache_load(path, today)
        if cached is not None:
            print(f"  -> using today's cached Alpaca bars for {len(cached)} tickers "
                  f"({path.name}; pass --refresh-cache to force a refetch)")
            return cached
    data = fetch_price_data(tickers, client, start, end)
    _cache_save(path, today, data)
    return data


def fetch_price_data(tickers: list[str], client, start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    """Bulk-fetch daily OHLCV from Alpaca (IEX feed), chunked ~100 symbols/call.
    A batch containing an invalid symbol is retried in halves so the other
    ~99 valid symbols are not lost with it.  A symbol whose bars contain a
    years-long gap (reused after an unrelated delisting) is trimmed to its
    most recent contiguous listing — see _latest_contiguous_segment."""
    print(f"Fetching Alpaca daily bars for {len(tickers)} symbols …")
    chunk_size = 100
    result: dict[str, pd.DataFrame] = {}
    dropped = 0
    trimmed = [0]
    total = len(tickers)
    show_progress = total > chunk_size

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        dropped += _fetch_bars_salvage(client, chunk, start, end, result, trimmed)
        time.sleep(0.3)
        if show_progress:
            done = min(i + chunk_size, total)
            print(f"  Alpaca bars {done}/{total} ({done * 100 // total}%)", flush=True)

    print(f"  -> received usable data for {len(result)} symbols"
          + (f" ({dropped} dropped as invalid for Alpaca)" if dropped else "")
          + (f" ({trimmed[0]} trimmed for a symbol-reuse gap)" if trimmed[0] else ""))
    return result


def _fetch_single_info(ticker: str) -> tuple[str, dict[str, Any] | None, Exception | None]:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).get_info()
        # Same payload shape as the screener's _fetch_yf_info so the shared
        # fundamentals_cache.json entries are interchangeable between tools.
        return ticker, {
            "exchange": str(info.get("exchange") or "").upper(),
            "market_cap": finite_number(info.get("marketCap")),
            "return_on_equity": finite_number(info.get("returnOnEquity")),
            "free_cash_flow": finite_number(info.get("freeCashflow")),
            "trailing_pe": finite_number(info.get("trailingPE")),
            "next_earnings_ts": finite_number(
                info.get("earningsTimestampStart") or info.get("earningsTimestamp")
            ),
        }, None
    except Exception as exc:
        return ticker, None, exc


def fetch_info_map(tickers: list[str], max_workers: int = 10,
                   cache_path: Path | None = Path("fundamentals_cache.json"),
                   max_age_days: float = 1.0) -> dict[str, dict[str, Any]]:
    """Fetch current fundamentals once (static for the whole backtest).

    Shares fundamentals_cache.json with the screener: entries younger than
    max_age_days skip Yahoo entirely, and stale entries are still used when a
    fresh fetch fails, so a rate-limited Yahoo day cannot zero the backtest.
    Default max_age_days=1 refreshes once per calendar day, same cadence as
    the Alpaca bars cache."""
    now = datetime.now(timezone.utc)
    cache: dict[str, dict[str, Any]] = {}
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    out: dict[str, dict[str, Any]] = {}
    to_fetch: list[str] = []
    for t in tickers:
        entry = cache.get(t)
        if entry and max_age_days > 0:
            try:
                age = (now - datetime.fromisoformat(entry["cached_at"])).total_seconds() / 86400
            except Exception:
                age = None
            if age is not None and age <= max_age_days:
                out[t] = entry["data"]
                continue
        to_fetch.append(t)
    if out:
        print(f"  -> {len(out)} fundamentals from cache (≤{max_age_days:.0f}d old)")

    total = len(to_fetch)
    if total:
        print(f"Fetching fundamentals for {total} tickers (yfinance, {max_workers} workers, static) …")
        done = 0
        step = max(1, total // 100)  # report roughly every 1%
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_fetch_single_info, t): t for t in to_fetch}
            for fut in as_completed(futs):
                t, data, err = fut.result()
                if data is not None:
                    out[t] = data
                    cache[t] = {"cached_at": now.isoformat(), "data": data}
                done += 1
                if done % step == 0 or done == total:
                    print(f"  fundamentals {done}/{total} ({done * 100 // total}%) — {len(out)} usable",
                          flush=True)
        stale_used = 0
        for t in to_fetch:
            if t not in out and t in cache:
                out[t] = cache[t]["data"]
                stale_used += 1
        if stale_used:
            print(f"  -> reused {stale_used} stale cache entries for tickers Yahoo failed on")

    if cache_path:
        try:
            cache_path.write_text(json.dumps(cache))
        except Exception as exc:
            print(f"WARNING: could not write {cache_path}: {exc}", file=sys.stderr)

    print(f"  -> fundamentals for {len(out)} tickers")
    return out


# ---------------------------------------------------------------------------
# Technical indicators (vectorised, once per ticker)
# ---------------------------------------------------------------------------
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period).mean()
    return atr


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
                 ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Standard MACD(12,26,9) -> (line, signal, histogram), in price units.

    Callers normalise by close before comparing across the cross-section: raw
    MACD scales with price level, so an unnormalised histogram ranks a $500
    stock above a $15 one regardless of relative momentum."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical columns needed by the screen."""
    df = df.copy()
    df["rsi_14"] = compute_rsi(df["Close"], 14)
    df["rsi_2"] = compute_rsi(df["Close"], 2)
    # MACD(12,26,9).  macd_hist_pct is the histogram as a %% of price — the
    # cross-sectionally comparable form, and the one alpha_lab measured a
    # positive (if modest) IC on.
    macd_line, macd_signal, macd_hist = compute_macd(df["Close"])
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    df["macd_hist_pct"] = macd_hist / df["Close"] * 100
    df["macd_line_pct"] = macd_line / df["Close"] * 100
    df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["atr_14"] = compute_atr(df["High"], df["Low"], df["Close"], 14)
    df["atr_pct"] = df["atr_14"] / df["Close"] * 100
    # Exclude the current session from the volume baseline (matches main screener).
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean().shift(1)
    df["dollar_volume_20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["sma_200"] = df["Close"].rolling(200).mean()
    df["sma200_prev20"] = df["sma_200"].shift(20)
    df["5d_return"] = df["Close"].pct_change(5) * 100
    # 6-month momentum skipping the most recent month (matches main screener).
    df["6m_return"] = (df["Close"].shift(21) / df["Close"].shift(126) - 1) * 100
    # 12-1 momentum (Jegadeesh & Titman): alpha_lab measures ICIR +0.178 for
    # this vs -0.028 for the 6-month version on this universe, so it is the
    # stronger horizon here.  Selectable via --momentum-window.
    df["12m_return"] = (df["Close"].shift(21) / df["Close"].shift(252) - 1) * 100
    df["ext_vs_sma50"] = (df["Close"] / df["sma_50"] - 1) * 100
    high_20 = df["Close"].rolling(20).max()
    df["price_vs_20d_high"] = (df["Close"] / high_20 - 1) * 100
    # How stretched price is above its 10-day EMA (matches main screener).
    df["ext_vs_ema10"] = (df["Close"] / df["ema_10"] - 1) * 100
    # Where the close sits in the day's range: 1.0 = closed at the high.
    day_range = df["High"] - df["Low"]
    df["close_loc"] = ((df["Close"] - df["Low"]) / day_range).where(day_range > 0, 0.5)

    # --- Guotai Junan Alpha-191 factors (mirrors the screener's add_alpha191;
    # normalised so they are comparable across a US price/volume range) ------
    rng = (df["High"] - df["Low"]).replace(0, float("nan"))
    clv = (((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng).astype(float)
    df["alpha002"] = -clv.diff(1)
    money_flow = df["Volume"] * clv
    df["alpha111"] = ((money_flow.ewm(alpha=2 / 11, adjust=False).mean()
                       - money_flow.ewm(alpha=2 / 4, adjust=False).mean())
                      / df["Volume"].rolling(20).mean())
    body = (df["Close"] - df["Open"]) / df["Close"]
    df["alpha054"] = -(body.abs().rolling(10).std() + body
                       + df["Close"].rolling(10).corr(df["Open"]))
    return df


# ---------------------------------------------------------------------------
# Screening logic — mirrors the main screener's technical, liquidity and
# quality rules.  The earnings blackout and news gate are omitted because
# neither is point-in-time backtestable with free Yahoo data.
# ---------------------------------------------------------------------------
def passes_screen(row: pd.Series, info: dict[str, Any], args: argparse.Namespace,
                  spy_6m: float | None = None) -> bool:
    fundamentals_available = getattr(args, "fundamentals_available", True)
    if fundamentals_available:
        exchange = info.get("exchange", "")
        if exchange not in {"NMS", "NGM", "NCM", "NAS", "NYQ", "NYS"}:
            return False

        market_cap = info.get("market_cap")
        if market_cap is None or market_cap <= args.min_market_cap:
            return False

    price = finite_number(row.get("Close"))
    if price is None or price < args.min_price:
        return False

    dollar_volume = finite_number(row.get("dollar_volume_20"))
    if dollar_volume is None or dollar_volume < args.min_dollar_volume:
        return False

    # Long-term trend
    sma_200 = finite_number(row.get("sma_200"))
    if args.above_200d_sma:
        if sma_200 is None or price <= sma_200:
            return False
    sma_50 = finite_number(row.get("sma_50"))
    if args.require_golden_cross:
        if sma_50 is None or sma_200 is None or sma_50 <= sma_200:
            return False
    if args.require_rising_200d:
        sma_200_prev = finite_number(row.get("sma200_prev20"))
        if sma_200 is None or sma_200_prev is None or sma_200 <= sma_200_prev:
            return False

    # Intermediate momentum (absolute floor + relative strength vs SPY).
    # --momentum-window picks which horizon carries the rule; the benchmark
    # series passed in as spy_6m is computed on the same window by the caller.
    mom_col = "12m_return" if getattr(args, "momentum_window", 6) == 12 else "6m_return"
    ret6m = finite_number(row.get(mom_col))
    if ret6m is None or ret6m < args.min_6m_return:
        return False
    if spy_6m is not None and (ret6m - spy_6m) < args.min_rs_6m:
        return False

    # Reversal guards: not stretched above the 50-day, not broken below it
    ext50 = finite_number(row.get("ext_vs_sma50"))
    if ext50 is None or ext50 > args.max_ext_sma50 or ext50 < args.min_ext_sma50:
        return False
    ret5 = finite_number(row.get("5d_return"))
    if ret5 is None or ret5 > args.max_5d_return:
        return False

    # Entry zone
    rsi = row.get("rsi_14")
    if rsi is None or not (args.min_rsi <= rsi <= args.max_rsi):
        return False

    atr_pct = finite_number(row.get("atr_pct"))
    if atr_pct is None or atr_pct < args.min_atr_pct or atr_pct > args.max_atr_pct:
        return False

    # --- Optional RSI(2) / MACD gates (all off by default) -----------------
    # Connors RSI(2): buy a deeply oversold 2-period RSI inside an uptrend.
    # The 200-day-SMA precondition Connors requires is already enforced above.
    max_rsi2 = getattr(args, "max_rsi2", None)
    if max_rsi2 is not None and max_rsi2 > 0:
        rsi2 = finite_number(row.get("rsi_2"))
        if rsi2 is None or rsi2 > max_rsi2:
            return False
    min_rsi2 = getattr(args, "min_rsi2", None)
    if min_rsi2 is not None and min_rsi2 > 0:
        rsi2 = finite_number(row.get("rsi_2"))
        if rsi2 is None or rsi2 < min_rsi2:
            return False
    # MACD: require the histogram (MACD - signal) above a floor.  0 = the
    # classic "MACD line above its signal line" bullish-momentum condition.
    min_macd_hist = getattr(args, "min_macd_hist_pct", None)
    if min_macd_hist is not None:
        hist = finite_number(row.get("macd_hist_pct"))
        if hist is None or hist < min_macd_hist:
            return False
    # MACD line above zero — the longer-term "12-EMA above 26-EMA" regime.
    if getattr(args, "require_macd_positive", False):
        line = finite_number(row.get("macd_line_pct"))
        if line is None or line <= 0:
            return False

    # Optional extras (off by default)
    if args.min_volume_ratio > 0:
        vol_ratio = finite_number(row.get("vol_ratio"))
        if vol_ratio is None or vol_ratio < args.min_volume_ratio:
            return False
    if args.min_close_strength > 0:
        close_loc = finite_number(row.get("close_loc"))
        if close_loc is None or close_loc < args.min_close_strength:
            return False

    if fundamentals_available:
        if args.require_positive_fcf:
            fcf = info.get("free_cash_flow")
            if fcf is None or fcf <= 0:
                return False
        if args.require_roe is not None and args.require_roe > 0:
            roe = info.get("return_on_equity")
            if roe is None or roe < args.require_roe:
                return False

    return True


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    ticker: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: float
    return_pct: float
    days_held: int
    exit_reason: str = "TIME"


def resolve_exit(df: pd.DataFrame, entry_idx: int, time_exit_idx: int, entry_price: float,
                 stop_pct: float, take_pct: float, trail_pct: float,
                 slippage: float) -> tuple[int, float, str]:
    """Walk the holding window bar by bar and return (exit_idx, exit_price, reason).

    Realistic fill assumptions:
      - a gap through the stop fills at the OPEN, not the stop price
      - if a bar touches both the stop and the target, the STOP is assumed
        first (conservative — intraday order is unknowable from daily bars)
    Returns the time-based exit when nothing is breached.
    """
    peak = entry_price
    hard_stop = entry_price * (1 - stop_pct / 100) if stop_pct > 0 else None
    target = entry_price * (1 + take_pct / 100) if take_pct > 0 else None

    for i in range(entry_idx + 1, min(time_exit_idx, len(df.index) - 1) + 1):
        bar = df.iloc[i]
        high, low, open_ = float(bar["High"]), float(bar["Low"]), float(bar["Open"])
        # Trail off the peak established by PRIOR bars: using this bar's own
        # high would assume the high printed before the low, which daily bars
        # cannot establish.  Peak is updated after the breach check.
        stop = hard_stop
        if trail_pct > 0:
            trail = peak * (1 - trail_pct / 100)
            stop = trail if stop is None else max(stop, trail)

        if stop is not None and low <= stop:
            fill = min(open_, stop)          # gap-down fills at the open
            return i, fill * (1 - slippage / 100), "STOP"
        if target is not None and high >= target:
            fill = max(open_, target)        # gap-up fills at the open
            return i, fill * (1 - slippage / 100), "TARGET"
        peak = max(peak, high)

    idx = min(time_exit_idx, len(df.index) - 1)
    return idx, float(df.iloc[idx]["Close"]) * (1 - slippage / 100), "TIME"


def get_nth_trading_day_after(dates: list[datetime], current: datetime, n: int) -> datetime | None:
    try:
        idx = dates.index(current)
    except ValueError:
        # fallback to nearest future date
        future = [d for d in dates if d > current]
        if not future:
            return None
        current = future[0]
        idx = dates.index(current)
    if idx + n < len(dates):
        return dates[idx + n]
    return None


def run_backtest(
    hist_data: dict[str, pd.DataFrame],
    info_map: dict[str, dict[str, Any]],
    spy_df: pd.DataFrame | None,
    args: argparse.Namespace,
) -> tuple[list[Trade], pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        trades: list of Trade
        equity_curve: DataFrame(date, equity)
        benchmark: DataFrame(date, spy_equity)
    """
    # --- Master trading calendar anchored to SPY ---------------------------
    # Anchoring to SPY (rather than intersecting every ticker's dates) keeps a
    # single sparse or short-history name from collapsing the whole window.
    # Trading begins at --start; everything before it is warm-up used only so
    # the 200-day SMA / 6-month momentum are already defined on day one.
    start_dt = pd.Timestamp(args.start)

    spy_6m_by_date: dict = {}
    if spy_df is not None and not spy_df.empty:
        spy_full = spy_df.copy()
        # Regime is computed on the FULL series (needs the warm-up) and only
        # then trimmed, so the gate is live from the first traded session.
        # Risk-off = below the 200-day OR the 200-day is falling (bear-rally trap).
        spy_sma200 = spy_full["Close"].rolling(200).mean()
        spy_risk_off_full = (spy_full["Close"] <= spy_sma200) | (spy_sma200 <= spy_sma200.shift(20))
        # SPY 6-month return per date, for the relative-strength rule.
        spy_6m_series = (spy_full["Close"].shift(21) / spy_full["Close"].shift(126) - 1) * 100
        spy_6m_by_date = {d: v for d, v in spy_6m_series.items() if pd.notna(v)}
        spy = spy_full[spy_full.index >= start_dt].copy()
        all_dates = sorted(spy.index)
        spy["spy_return"] = spy["Close"].pct_change()
        spy["spy_equity"] = (1 + spy["spy_return"].fillna(0)).cumprod() * args.initial_cash
    else:
        # Fallback: union (never intersection) of every ticker's trading days.
        print("WARNING: no SPY benchmark available; using union of ticker dates", file=sys.stderr)
        union: set[datetime] = set()
        for df in hist_data.values():
            union |= set(df.index)
        all_dates = sorted(d for d in union if d >= start_dt)
        spy = pd.DataFrame(index=all_dates)
        spy["spy_equity"] = args.initial_cash
        spy_risk_off_full = None

    if len(all_dates) < 50:
        raise RuntimeError("Too few trading dates in the backtest window.")

    # Point-in-time market-regime gate: no new entries while SPY sits at/below
    # its 200-day SMA.
    risk_off_dates: set = set()
    if args.market_regime and spy_risk_off_full is not None:
        risk_off_dates = set(spy_risk_off_full.index[spy_risk_off_full.fillna(False)])
        in_window = sum(1 for d in all_dates if d in risk_off_dates)
        print(f"Market-regime gate: {in_window} of {len(all_dates)} traded sessions risk-off")

    rebalance_dates = {all_dates[i] for i in range(0, len(all_dates), args.rebalance_freq)}

    cash = float(args.initial_cash)
    positions: dict[str, dict[str, Any]] = {}
    pending_entries: dict[datetime, list[str]] = {}  # execution date -> tickers screened the prior bar
    trades: list[Trade] = []
    equity_records: list[dict[str, Any]] = []

    for date in all_dates:
        # --- 1. Execute entries screened on the previous rebalance bar -------
        scheduled = pending_entries.pop(date, None)
        if scheduled:
            available = [t for t in scheduled
                         if t not in positions and t in hist_data and date in hist_data[t].index]
            # Respect the concurrent-position cap so a single name can never take
            # an outsized share of the book.
            room = max(0, args.max_positions - len(positions))
            available = available[:room]
            if available:
                # Size each new position at 1/max_positions of current equity
                # rather than splitting all cash across however few names passed.
                equity_now = cash + sum(p["shares"] * p["last_price"] for p in positions.values())
                allocation = min(equity_now / args.max_positions, cash / len(available))
                if allocation >= 1:
                    for ticker in available:
                        bar = hist_data[ticker].loc[date]
                        fill = finite_number(bar.get("Open")) or finite_number(bar.get("Close"))
                        if not fill or fill <= 0:
                            continue
                        exit_date = get_nth_trading_day_after(all_dates, date, args.hold_days)
                        if exit_date is None:
                            continue
                        buy_price = fill * (1 + args.slippage / 100)
                        shares = allocation / buy_price
                        positions[ticker] = {
                            "shares": shares,
                            "entry_price": buy_price,
                            "entry_date": date,
                            "exit_date": exit_date,
                            "last_price": buy_price,
                            "peak": buy_price,
                        }
                        cash -= shares * buy_price

        # --- 2. Mark to market & exits (forward-fill price across data gaps) --
        mtm = cash
        for ticker, pos in list(positions.items()):
            bar = None
            if ticker in hist_data and date in hist_data[ticker].index:
                bar = hist_data[ticker].loc[date]
                pos["last_price"] = float(bar["Close"])
            price = pos["last_price"]
            mtm += pos["shares"] * price

            # Risk exits are checked intraday BEFORE the time exit: a stop that
            # was breached today gets you out today, not on the hold's last day.
            exit_reason, sell_price = None, None
            if bar is not None and date > pos["entry_date"]:
                high, low, open_ = float(bar["High"]), float(bar["Low"]), float(bar["Open"])
                # Trail off PRIOR bars' peak (see resolve_exit); update after.
                stop = pos["entry_price"] * (1 - args.stop_loss_pct / 100) if args.stop_loss_pct > 0 else None
                if args.trailing_stop_pct > 0:
                    trail = pos["peak"] * (1 - args.trailing_stop_pct / 100)
                    stop = trail if stop is None else max(stop, trail)
                target = pos["entry_price"] * (1 + args.take_profit_pct / 100) if args.take_profit_pct > 0 else None
                if stop is not None and low <= stop:
                    exit_reason, sell_price = "STOP", min(open_, stop) * (1 - args.slippage / 100)
                elif target is not None and high >= target:
                    exit_reason, sell_price = "TARGET", max(open_, target) * (1 - args.slippage / 100)
                pos["peak"] = max(pos["peak"], high)

            if exit_reason is None and date >= pos["exit_date"]:
                exit_reason, sell_price = "TIME", price * (1 - args.slippage / 100)

            if exit_reason is not None:
                cash += pos["shares"] * sell_price
                trades.append(Trade(
                    ticker=ticker,
                    entry_date=pos["entry_date"],
                    exit_date=date,
                    entry_price=pos["entry_price"],
                    exit_price=sell_price,
                    shares=pos["shares"],
                    return_pct=(sell_price / pos["entry_price"] - 1) * 100,
                    days_held=(date - pos["entry_date"]).days,
                    exit_reason=exit_reason,
                ))
                del positions[ticker]

        equity_records.append({"date": date, "equity": mtm})

        # --- 3. Rebalance: screen on this bar, queue entries for the next -----
        if date not in rebalance_dates:
            continue
        if date in risk_off_dates:
            continue
        next_day = get_nth_trading_day_after(all_dates, date, 1)
        if next_day is None:
            continue

        picks = []
        for ticker, df in hist_data.items():
            if ticker not in info_map or date not in df.index:
                continue
            if passes_screen(df.loc[date], info_map[ticker], args, spy_6m_by_date.get(date)):
                picks.append(ticker)
        if picks:
            pending_entries[next_day] = picks

    equity_curve = pd.DataFrame(equity_records).set_index("date")
    benchmark = spy[["spy_equity"]].reindex(equity_curve.index, method="ffill")
    return trades, equity_curve, benchmark


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(trades: list[Trade], equity: pd.DataFrame, benchmark: pd.DataFrame, args: argparse.Namespace) -> None:
    if not trades:
        print("\nNo trades generated.")
        return

    df = pd.DataFrame([{
        "ticker": t.ticker,
        "entry_date": t.entry_date.strftime("%Y-%m-%d"),
        "exit_date": t.exit_date.strftime("%Y-%m-%d"),
        "entry_price": round(t.entry_price, 2),
        "exit_price": round(t.exit_price, 2),
        "return_pct": round(t.return_pct, 2),
        "days_held": t.days_held,
        "exit_reason": t.exit_reason,
    } for t in trades])

    out_path = Path(args.output_prefix + "_trades.csv")
    df.to_csv(out_path, index=False)
    print(f"\nTrades written to {out_path}")

    # Summary stats
    rets = df["return_pct"].astype(float)
    wins = rets > 0

    equity["returns"] = equity["equity"].pct_change()
    equity["cummax"] = equity["equity"].cummax()
    equity["drawdown"] = (equity["equity"] - equity["cummax"]) / equity["cummax"] * 100

    total_ret = (equity["equity"].iloc[-1] / args.initial_cash - 1) * 100
    spy_total_ret = (benchmark["spy_equity"].iloc[-1] / args.initial_cash - 1) * 100 if "spy_equity" in benchmark else 0

    ann_factor = 252 / len(equity)
    strat_ann = ((1 + total_ret / 100) ** ann_factor - 1) * 100
    spy_ann = ((1 + spy_total_ret / 100) ** ann_factor - 1) * 100 if "spy_equity" in benchmark else 0

    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    BACKTEST SUMMARY                          ║
╠══════════════════════════════════════════════════════════════╣
  Backtest period      : {equity.index[0].strftime('%Y-%m-%d')} → {equity.index[-1].strftime('%Y-%m-%d')}
  Initial cash         : ${args.initial_cash:,.0f}
  Hold days            : {args.hold_days}
  Rebalance freq       : every {args.rebalance_freq} trading days
  Slippage             : {args.slippage}%

  --- Strategy ---
  Total trades         : {len(df)}
  Win rate             : {wins.mean()*100:.1f}%
  Avg return / trade   : {rets.mean():.2f}%
  Median return        : {rets.median():.2f}%
  Best trade           : {rets.max():.2f}%
  Worst trade          : {rets.min():.2f}%
  Total return         : {total_ret:.2f}%
  Annualised return    : {strat_ann:.2f}%
  Max drawdown         : {equity['drawdown'].min():.2f}%

  --- Benchmark (SPY buy & hold) ---
  Total return         : {spy_total_ret:.2f}%
  Annualised return    : {spy_ann:.2f}%

  --- Alpha ---
  Excess return        : {total_ret - spy_total_ret:+.2f}%
╚══════════════════════════════════════════════════════════════╝
"""
    print(summary)

    # Save equity curve
    combined = equity.copy()
    if "spy_equity" in benchmark:
        combined["spy_equity"] = benchmark["spy_equity"]
    combined.to_csv(args.output_prefix + "_equity.csv")
    print(f"Equity curve written to {args.output_prefix}_equity.csv")

    rr.metrics("backtest", "headline", {
        "period_start": equity.index[0].strftime("%Y-%m-%d"),
        "period_end": equity.index[-1].strftime("%Y-%m-%d"),
        "initial_cash": args.initial_cash, "hold_days": args.hold_days,
        "rebalance_freq": args.rebalance_freq, "slippage_pct": args.slippage,
        "total_trades": len(df),
        "win_rate_pct": round(wins.mean() * 100, 2),
        "avg_return_per_trade_pct": round(rets.mean(), 4),
        "median_return_pct": round(rets.median(), 4),
        "best_trade_pct": round(rets.max(), 4),
        "worst_trade_pct": round(rets.min(), 4),
        "total_return_pct": round(total_ret, 4),
        "annualised_return_pct": round(strat_ann, 4),
        "max_drawdown_pct": round(equity["drawdown"].min(), 4),
        "spy_total_return_pct": round(spy_total_ret, 4),
        "spy_annualised_return_pct": round(spy_ann, 4),
        "excess_return_pct": round(total_ret - spy_total_ret, 4),
    })
    if "exit_reason" in df.columns:
        rr.rows("backtest", "by_exit_reason", [
            {"exit_reason": r, "trades": len(g),
             "win_pct": round((g["return_pct"] > 0).mean() * 100, 2),
             "avg_return_pct": round(g["return_pct"].mean(), 3),
             "worst_return_pct": round(g["return_pct"].min(), 3)}
            for r, g in df.groupby("exit_reason")])
    rr.note("backtest", "caveats",
            "Fundamentals are today's values (survivorship + lookahead bias); the news gate "
            "and earnings blackout are not simulated. Treat headline alpha with skepticism.")

    # Plot (optional, if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(combined.index, combined["equity"], label="Strategy", linewidth=1.5)
        if "spy_equity" in combined:
            ax.plot(combined.index, combined["spy_equity"], label="SPY", linewidth=1.5, alpha=0.7)
        ax.set_title("10-Week Swing Screen Backtest")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = args.output_prefix + "_equity.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", type=Path, default=Path("screener_universe.txt"),
                        help="Ticker file, one symbol per line (default: screener_universe.txt, "
                             "written by short_term_screener_alpaca.py)")
    parser.add_argument("--start", type=str, required=True,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--alpaca-key", type=str, default=None,
                        help="Alpaca API key (or set ALPACA_API_KEY env var)")
    parser.add_argument("--alpaca-secret", type=str, default=None,
                        help="Alpaca API secret (or set ALPACA_SECRET_KEY env var)")
    parser.add_argument("--initial-cash", type=float, default=100_000)
    parser.add_argument("--hold-days", type=int, default=50,
                        help="Trading days to hold each position, ~10 weeks (default: 50)")
    parser.add_argument("--max-positions", type=int, default=5,
                        help="Maximum concurrent positions; caps per-name weight at "
                             "1/N of equity so one loser cannot sink the book (default: 5)")
    parser.add_argument("--rebalance-freq", type=int, default=10,
                        help="Trade every N trading days (default: 10)")
    parser.add_argument("--warmup-days", type=int, default=400,
                        help="Extra calendar days of history fetched before --start so the "
                             "200-day SMA and 6-month momentum are defined (default: 400)")
    parser.add_argument("--stop-loss-pct", type=float, default=15.0,
                        help="Hard stop below entry, %%; caps catastrophic single-position "
                             "losses (0 disables; default: 15). A wider 25%% stop looked "
                             "better in ab_sweep's per-pick avg ROI but cut THIS backtest's "
                             "excess from +38%% to +23%%, so 15 stands.")
    parser.add_argument("--trailing-stop-pct", type=float, default=0.0,
                        help="Trailing stop below the running peak, %% (0 disables)")
    parser.add_argument("--take-profit-pct", type=float, default=0.0,
                        help="Take-profit above entry, %% (0 disables)")
    parser.add_argument("--slippage", type=float, default=0.05,
                        help="One-way slippage %% (default: 0.05)")

    # Screen parameters (must match main screener)
    parser.add_argument("--above-200d-sma", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-golden-cross", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-rising-200d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-6m-return", type=float, default=5.0)
    parser.add_argument("--min-rs-6m", type=float, default=0.0)
    parser.add_argument("--momentum-window", type=int, choices=[6, 12], default=6,
                        help="Momentum horizon in months for the trend rule and the ranking: "
                             "6 (default, current behaviour) or 12. alpha_lab measures a much "
                             "stronger ICIR for the 12-month version on this universe.")
    parser.add_argument("--max-ext-sma50", type=float, default=15.0)  # 25 hurt backtest excess (-2.6pp)
    parser.add_argument("--min-ext-sma50", type=float, default=-3.0)  # -8 hurt backtest excess (-3pp)
    parser.add_argument("--max-5d-return", type=float, default=8.0)
    parser.add_argument("--min-rsi", type=float, default=40.0)
    parser.add_argument("--max-rsi", type=float, default=65.0)
    parser.add_argument("--min-atr-pct", type=float, default=1.0)
    parser.add_argument("--max-atr-pct", type=float, default=6.0)
    parser.add_argument("--min-price", type=float, default=10.0)
    parser.add_argument("--min-dollar-volume", type=float, default=10_000_000)
    parser.add_argument("--min-volume-ratio", type=float, default=0.0)
    parser.add_argument("--min-close-strength", type=float, default=0.0)
    # --- RSI(2) / MACD gates (off by default; see alpha_report_rsi_macd.csv) --
    parser.add_argument("--max-rsi2", type=float, default=None,
                        help="Connors RSI(2) ceiling — buy only deeply oversold names inside "
                             "the uptrend (classic values 5 or 10; default: off)")
    parser.add_argument("--min-rsi2", type=float, default=None,
                        help="RSI(2) floor, the momentum-side inverse of --max-rsi2 (default: off)")
    parser.add_argument("--min-macd-hist-pct", type=float, default=None,
                        help="Minimum MACD histogram as %% of price; 0 = require the MACD line "
                             "above its signal line (default: off)")
    parser.add_argument("--require-macd-positive", action=argparse.BooleanOptionalAction, default=False,
                        help="Require the MACD line above zero (12-EMA above 26-EMA; default: off)")
    parser.add_argument("--market-regime", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip new entries when SPY is at/below its 200-day SMA "
                             "(point-in-time; disable with --no-market-regime)")
    parser.add_argument("--min-market-cap", type=float, default=2_000_000_000)
    parser.add_argument("--require-positive-fcf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-roe", type=float, default=0.10)

    parser.add_argument("--output-prefix", type=str, default="backtest")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Ignore today's cached Alpaca bars / fundamentals and refetch "
                             "(cache normally makes a same-day rerun skip both network calls)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rr.begin("backtest", args)
    if not args.tickers.exists():
        raise SystemExit(
            f"Ticker file '{args.tickers}' not found. Run short_term_screener_alpaca.py "
            f"first (it writes this file), or pass --tickers <file>."
        )
    tickers = read_tickers(args.tickers)
    if not tickers:
        raise SystemExit("No tickers found in file.")

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    # Fetch a warm-up year before --start so the 200-day SMA and 6-month
    # momentum are already defined on the first session actually traded.
    fetch_start = start - timedelta(days=args.warmup_days)

    # 1. Fetch prices + SPY benchmark from Alpaca
    client = _get_alpaca_client(args.alpaca_key, args.alpaca_secret)
    hist_data = fetch_price_data_cached(tickers, client, fetch_start, end, refresh=args.refresh_cache)
    if not hist_data:
        raise SystemExit("No price data returned from Alpaca.")
    spy_df = fetch_price_data_cached(["SPY"], client, fetch_start, end, refresh=args.refresh_cache).get("SPY")

    # 2. Compute signals
    print("Computing technical signals …")
    too_short = 0
    for ticker in list(hist_data.keys()):
        # A name with fewer than 200 bars in the whole window can never have a
        # 200-day SMA, so it can never pass the screen — drop it before the
        # slow fundamentals phase.
        if len(hist_data[ticker]) < 200:
            del hist_data[ticker]
            too_short += 1
            continue
        hist_data[ticker] = compute_signals(hist_data[ticker])
    if too_short:
        print(f"  -> skipped {too_short} tickers with <200 bars (cannot pass the trend rules)")
    if not hist_data:
        raise SystemExit("No tickers with enough history to screen.")

    # 3. Fetch static fundamentals (yfinance — Alpaca has none)
    info_map = fetch_info_map(list(hist_data.keys()), max_workers=args.max_workers,
                              max_age_days=0 if args.refresh_cache else 1.0)
    args.fundamentals_available = bool(info_map)
    if not args.fundamentals_available:
        print("WARNING: fundamentals unavailable for EVERY ticker (Yahoo outage/rate limit?) — "
              "screening on technicals only; exchange, market-cap, FCF and ROE gates skipped.",
              file=sys.stderr)
        info_map = {t: {} for t in hist_data}

    # 4. Run backtest
    print("Running backtest …")
    trades, equity, benchmark = run_backtest(hist_data, info_map, spy_df, args)

    # 5. Report
    report(trades, equity, benchmark, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
