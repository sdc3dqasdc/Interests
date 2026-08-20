"""Backtest for NTI-core with the steady leg swapped: QQQ vs a flat 3.35%/yr
product instead of SPY, switch rule (ladder), 20pt no-trade band.

Same signal and point-in-time conventions as backtest.py (trailing z-scores
only, traded at t+1's close), but three things differ: the non-tech sleeve pays
a flat rate instead of SPY's return, trades are skipped unless the target moved
at least `--band`, and costs default to 1%/side rather than 5bp.

That last one is not a detail. At 1% the fee decides the answer: it makes slower
rebalancing strictly better, and it is what sinks the fine-grained continuous
dial, which pays the spread on every wobble.

    .venv/bin/python -m tech_rotation.backtest_product
    .venv/bin/python -m tech_rotation.backtest_product --sweep --risk-off 0
    .venv/bin/python -m tech_rotation.backtest_product --cost-bps 5   # institutional
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import data as dt
from . import index as ix
from .backtest import TRADING_DAYS, COST_BPS, OOS_SPLIT, _metrics, _fmt, _table, _period_table, Result

# 1% per side, not backtest.py's 5bp. That module models an ETF pair traded at
# institutional spreads; this one models a retail product whose round trip costs
# ~2%, and at that level the fee is not a rounding error -- it is what decides
# the rebalance cadence and kills the fine-grained continuous dial outright.
PRODUCT_COST_BPS = 100.0


def _table_clean(results: list[Result], span: tuple[str | None, str | None] = (None, None)) -> pd.DataFrame:
    """_table, with Sharpe/Calmar blanked for the flat product leg (zero vol
    makes those ratios blow up to meaningless numbers)."""
    df = _table(results, span)
    near_zero_vol = df["Vol"].abs() < 1e-9
    df.loc[near_zero_vol, ["Sharpe", "Calmar"]] = np.nan
    return df


def run_banded(
    px: pd.DataFrame,
    nti: pd.Series,
    product_return: float,
    band: float = 0.20,
    risk_off: float = ix.RISK_OFF_WEIGHT,
    rebalance: int = 5,
    phase: int = 0,
    lag: int = 1,
    cost_bps: float = PRODUCT_COST_BPS,
) -> Result:
    """Ladder target, traded only when it has drifted `band` from what's held.

    `phase` shifts which sessions the rebalance grid lands on. It should not
    matter: a real cadence effect is about how OFTEN you trade, not which
    particular Tuesdays you picked. Sweeping it is the cheapest way to tell a
    frequency effect from the luck of a grid that happened to straddle a
    selloff.

    The product leg is a flat annualised return compounded daily -- there is no
    price series for it, so this is exactly as synthetic as the number you give
    it and carries none of SPY's own volatility or drawdowns.

    `risk_off` is what the defensive state holds in QQQ; 0.0 goes fully to the
    product.  Note the interaction with `band`: the ladder emits two weights, so
    every move is (risk_on - risk_off) points, and any band smaller than that
    gap never blocks a trade."""
    tech_ret = px[dt.TECH].pct_change()
    product_daily = (1 + product_return) ** (1 / TRADING_DAYS) - 1
    product_ret = pd.Series(product_daily, index=px.index)

    ladder = ix.ladder_weight(nti, risk_off=risk_off)
    traded = ladder.shift(lag)
    schedule = pd.Series(np.arange(len(traded)) % rebalance == phase % rebalance,
                         index=traded.index)
    candidate = traded.where(schedule).ffill()

    held = pd.Series(np.nan, index=candidate.index)
    current = np.nan
    for date, val in candidate.items():
        if pd.notna(val):
            if pd.isna(current) or abs(val - current) >= band:
                current = val
        held[date] = current

    valid = held.notna() & tech_ret.notna()
    held, tech_ret, product_ret = held[valid], tech_ret[valid], product_ret[valid]

    gross = held * tech_ret + (1 - held) * product_ret
    turn = held.diff().abs().fillna(held.iloc[0]) * 2
    net = gross - turn * cost_bps / 10_000.0

    equity = (1 + net).cumprod()
    turnover_yr = turn.sum() / (len(turn) / TRADING_DAYS)
    return Result(f"NTI-core -> product (off={risk_off:.0%}, band={band:.0%}, rb={rebalance}d)",
                  equity, held, turnover_yr)


def run_continuous(
    px: pd.DataFrame,
    nti: pd.Series,
    product_return: float,
    k: float = 0.25,
    band: float = 0.0,
    rebalance: int = 5,
    lag: int = 1,
    cost_bps: float = PRODUCT_COST_BPS,
) -> Result:
    """Continuous 0-100% tilt, traded only when the target has drifted `band`
    from what is held.

    Unlike the ladder, the band is NOT inert here: the ladder emits two weights
    so every move is 80pt and no band below that ever blocks a trade, whereas
    the continuous dial moves a few points at a time and a band is the only
    thing standing between it and paying the fee on every wobble."""
    tech_ret = px[dt.TECH].pct_change()
    product_daily = (1 + product_return) ** (1 / TRADING_DAYS) - 1
    product_ret = pd.Series(product_daily, index=px.index)

    target = ix.target_weight(nti, k=k, floor=0.0, cap=1.0)
    traded = target.shift(lag)
    schedule = pd.Series(np.arange(len(traded)) % rebalance == 0, index=traded.index)
    candidate = traded.where(schedule).ffill()

    held = pd.Series(np.nan, index=candidate.index)
    current = np.nan
    for date, val in candidate.items():
        if pd.notna(val):
            if pd.isna(current) or abs(val - current) >= band:
                current = val
        held[date] = current

    valid = held.notna() & tech_ret.notna()
    held, tech_ret, product_ret = held[valid], tech_ret[valid], product_ret[valid]

    gross = held * tech_ret + (1 - held) * product_ret
    turn = held.diff().abs().fillna(held.iloc[0]) * 2
    net = gross - turn * cost_bps / 10_000.0

    equity = (1 + net).cumprod()
    turnover_yr = turn.sum() / (len(turn) / TRADING_DAYS)
    return Result(f"NTI-core -> product (cont, band={band:.0%}, rb={rebalance}d)",
                  equity, held, turnover_yr)


def run_hold_qqq(px: pd.DataFrame, index: pd.Index, cost_bps: float = PRODUCT_COST_BPS) -> Result:
    tech_ret = px[dt.TECH].pct_change().reindex(index)
    equity = (1 + tech_ret.dropna()).cumprod()
    return Result(f"hold {dt.TECH}", equity, pd.Series(1.0, index=index), 0.0)


def outperformance(strat: Result, bench: Result) -> dict[str, float]:
    """How much the strategy outran holding the Nasdaq outright."""
    s, b = strat.equity.align(bench.equity, join="inner")
    s, b = s / s.iloc[0], b / b.iloc[0]
    years = len(s) / TRADING_DAYS
    strat_cagr = s.iloc[-1] ** (1 / years) - 1
    bench_cagr = b.iloc[-1] ** (1 / years) - 1

    strat_annual = s.resample("YE").last().pct_change().dropna()
    bench_annual = b.resample("YE").last().pct_change().dropna()
    yrs = strat_annual.index.intersection(bench_annual.index)

    return {
        "strategy total return": s.iloc[-1] - 1,
        "hold QQQ total return": b.iloc[-1] - 1,
        "total return edge": (s.iloc[-1] - 1) - (b.iloc[-1] - 1),
        "strategy CAGR": strat_cagr,
        "hold QQQ CAGR": bench_cagr,
        "CAGR edge": strat_cagr - bench_cagr,
        "strategy avg annual ROI": strat_annual.mean(),
        "hold QQQ avg annual ROI": bench_annual.mean(),
        "years strategy beat QQQ": (strat_annual.loc[yrs] > bench_annual.loc[yrs]).mean(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=dt.DEFAULT_START)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--band", type=float, default=0.20, help="no-trade band, in weight (0.20 = 20pt)")
    ap.add_argument("--product-return", type=float, default=0.0335,
                    help="flat annualised return of the steady-leg product (default 3.35%%)")
    ap.add_argument("--rebalance", type=int, default=5)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--cost-bps", type=float, default=PRODUCT_COST_BPS,
                    help=f"cost per unit of turnover, one side, in bp "
                         f"(default {PRODUCT_COST_BPS:.0f} = {PRODUCT_COST_BPS/100:.0f}%%)")
    ap.add_argument("--continuous", action="store_true",
                    help="0-100%% continuous tilt instead of the ladder switch (no band)")
    ap.add_argument("--k", type=float, default=0.25, help="tilt per sigma, continuous mode only")
    ap.add_argument("--risk-off", type=float, default=ix.RISK_OFF_WEIGHT,
                    help="fraction of QQQ held in the defensive state, ladder mode "
                         f"(default {ix.RISK_OFF_WEIGHT:.2f}; pass 0 to go fully to the product)")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep rebalance over 1/2/3 weeks (5/10/15 sessions) instead of one --rebalance")
    args = ap.parse_args()

    px = dt.fetch_prices(start=args.start, refresh=args.refresh)
    comps = ix.build_components(px)
    core_nti = ix.composite(comps[list(ix.CORE_WEIGHTS)], ix.CORE_WEIGHTS)

    rebalances = (5, 10, 15) if args.sweep else (args.rebalance,)
    strats = []
    for rb in rebalances:
        if args.continuous:
            s = run_continuous(px, core_nti, args.product_return, k=args.k, band=args.band,
                               rebalance=rb, lag=args.lag, cost_bps=args.cost_bps)
        else:
            s = run_banded(px, core_nti, args.product_return, band=args.band,
                           risk_off=args.risk_off, rebalance=rb, lag=args.lag,
                           cost_bps=args.cost_bps)
        strats.append(s)
    strat = strats[0]
    idx = strat.equity.index
    qqq = run_hold_qqq(px, idx, args.cost_bps)
    product_equity = (1 + args.product_return) ** (np.arange(len(idx)) / TRADING_DAYS)
    product = Result(f"hold product ({args.product_return:.2%}/yr)",
                     pd.Series(product_equity, index=idx), pd.Series(0.0, index=idx), 0.0)

    results = [*strats, qqq, product]

    span = f"{idx[0].date()} -> {idx[-1].date()}"
    mode = (f"continuous k={args.k}, band={args.band:.0%}" if args.continuous
            else f"ladder off={args.risk_off:.0%}, band={args.band:.0%}")
    print(f"\n=== Full sample  {span}  ({len(idx) / TRADING_DAYS:.1f}y) "
         f"[{mode}, cost={args.cost_bps/100:.2f}%/side, product={args.product_return:.2%}/yr] ===")
    print(_fmt(_table_clean(results)))

    print(f"\n=== In sample  {idx[0].date()} -> {OOS_SPLIT} ===")
    print(_fmt(_table_clean(results, (None, OOS_SPLIT))))
    print(f"\n=== Out of sample  {OOS_SPLIT} -> {idx[-1].date()} ===")
    print(_fmt(_table_clean(results, (OOS_SPLIT, None))))

    for s in strats:
        for label, span2 in (("Full sample", (None, None)),
                             ("In sample", (None, OOS_SPLIT)),
                             ("Out of sample", (OOS_SPLIT, None))):
            s_eq = s.equity.loc[span2[0]:span2[1]]
            b_eq = qqq.equity.loc[span2[0]:span2[1]]
            if len(s_eq) < TRADING_DAYS // 2 or len(b_eq) < TRADING_DAYS // 2:
                continue
            s_r = Result(s.name, s_eq, s.weights, s.turnover)
            b_r = Result(qqq.name, b_eq, qqq.weights, qqq.turnover)
            stats = outperformance(s_r, b_r)
            print(f"\n=== {label}: {s.name} vs hold {dt.TECH} ===")
            for k, v in stats.items():
                pct = "%" if "return" in k or "CAGR" in k or "ROI" in k or "years" in k else ""
                print(f"  {k:<28} {v * 100:+7.2f}{pct}" if pct else f"  {k:<28} {v:.3f}")

    print("\n=== Total return by episode ===")
    episodes = {
        "dot-com bust 00-02": ("2000-03-01", "2002-10-09"),
        "GFC 07-09": ("2007-10-09", "2009-03-09"),
        "tech run 13-21": ("2013-01-01", "2021-12-31"),
        "rate shock 2022": ("2022-01-01", "2022-12-31"),
        "AI run 23-now": ("2023-01-01", None),
    }
    ep = _period_table(results, episodes)
    print((ep * 100).round(1).to_string() + "   (%)")

    print(f"\nAverage {dt.TECH} weight {strat.weights.mean() * 100:.1f}% "
         f"(min {strat.weights.min() * 100:.0f}%, max {strat.weights.max() * 100:.0f}%), "
         f"turnover {strat.turnover:.2f}x/yr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
