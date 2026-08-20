"""Backtest for the Kechuang-Dibo Index (KC50 vs Dividend Low Volatility).

Same harness rules as tech_rotation/backtest.py: reading at close t, traded at
t+1's close; weekly rebalance; 5bp/side cost; every z-score trailing-only.
`--selftest` proves it.

The one difference that matters: **the traded history here is ~5 years, not
27.**  STAR50 (588000.SS) lists 2020-09-28.  Splitting that into "in-sample" /
"out-of-sample" halves gives ~2.5 years each -- one bull leg and one mixed leg,
not two independent market regimes.  Component selection done the same way as
the US module (keep only what has the same sign in both halves) is therefore
a much weaker filter here: with this little data, a coin flip has a real chance
of looking consistent by luck.  Read every result in this file as provisional.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data as dt
from . import index as ix

TRADING_DAYS = 252
COST_BPS = 5.0
TRADED_START = "2020-09-28"          # STAR50 ETF inception -- nothing tradeable before this
OOS_SPLIT = "2023-06-01"             # roughly bisects the traded window


@dataclass
class Result:
    name: str
    equity: pd.Series
    weights: pd.Series
    turnover: float


def _metrics(equity: pd.Series, turnover: float = float("nan")) -> dict[str, float]:
    ret = equity.pct_change(fill_method=None).dropna()
    if ret.empty:
        return {}
    years = len(ret) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    dd = (equity / equity.cummax() - 1).min()
    return {
        "CAGR": cagr, "Vol": vol,
        "Sharpe": (ret.mean() * TRADING_DAYS) / vol if vol else np.nan,
        "MaxDD": dd, "Calmar": cagr / abs(dd) if dd else np.nan,
        "Turnover/yr": turnover,
    }


def run_strategy(px: pd.DataFrame, kdi: pd.Series, name: str, k: float = 0.25,
                 rebalance: int = 5, lag: int = 1, binary: bool = False,
                 ladder: bool = False, cost_bps: float = COST_BPS) -> Result:
    growth_ret = px[dt.KC50].pct_change(fill_method=None)
    steady_ret = px[dt.DIBO].pct_change(fill_method=None)

    if ladder:
        target = ix.ladder_weight(kdi)
    elif binary:
        target = (kdi > 0).astype(float).where(kdi.notna())
    else:
        target = ix.target_weight(kdi, k=k)

    traded = target.shift(lag)
    schedule = pd.Series(np.arange(len(traded)) % rebalance == 0, index=traded.index)
    weights = traded.where(schedule).ffill()

    valid = weights.notna() & growth_ret.notna() & steady_ret.notna()
    weights, growth_ret, steady_ret = weights[valid], growth_ret[valid], steady_ret[valid]

    gross = weights * growth_ret + (1 - weights) * steady_ret
    turn = weights.diff().abs().fillna(weights.iloc[0]) * 2
    net = gross - turn * cost_bps / 10_000.0

    equity = (1 + net).cumprod()
    turnover_yr = turn.sum() / (len(turn) / TRADING_DAYS)
    return Result(name, equity, weights, turnover_yr)


def run_benchmark(px: pd.DataFrame, name: str, w_growth: float, index: pd.Index,
                  rebalance: int = 5, cost_bps: float = COST_BPS) -> Result:
    const = pd.Series(w_growth, index=px.index)
    growth_ret, steady_ret = px[dt.KC50].pct_change(fill_method=None), px[dt.DIBO].pct_change(fill_method=None)
    weights = const.reindex(index)
    growth_ret, steady_ret = growth_ret.reindex(index), steady_ret.reindex(index)
    gross = weights * growth_ret + (1 - weights) * steady_ret
    turn = weights.diff().abs().fillna(weights.iloc[0]) * 2
    if 0.0 < w_growth < 1.0:
        drift = (growth_ret - steady_ret).abs() * w_growth * (1 - w_growth)
        turn = turn + drift.where(pd.Series(np.arange(len(index)) % rebalance == 0, index=index), 0.0)
    net = gross - turn * cost_bps / 10_000.0
    equity = (1 + net.dropna()).cumprod()
    return Result(name, equity, weights, turn.sum() / (len(index) / TRADING_DAYS))


def timing_stats(px: pd.DataFrame, weights: pd.Series,
                 span: tuple[str | None, str | None] = (None, None)) -> dict[str, float]:
    spread = (px[dt.KC50].pct_change(fill_method=None) - px[dt.DIBO].pct_change(fill_method=None)).reindex(weights.index)
    w = weights.loc[span[0]:span[1]]
    sp = spread.loc[span[0]:span[1]]
    active = ((w - w.mean()) * sp).dropna()
    if len(active) < 60:
        return {}
    mean_ann = active.mean() * TRADING_DAYS
    vol_ann = active.std() * np.sqrt(TRADING_DAYS)
    tstat = active.mean() / (active.std() / np.sqrt(len(active))) if active.std() else np.nan
    fwd = sp[::-1].rolling(21).sum()[::-1].shift(-1)
    ic = w.rank().corr(fwd.rank())
    return {"Timing CAGR": mean_ann,
            "Timing IR": mean_ann / vol_ann if vol_ann else np.nan,
            "t-stat": tstat, "IC(21d)": ic,
            "Hit rate": float((np.sign(w - w.mean()) == np.sign(sp)).reindex(active.index).mean())}


def _table(results: list[Result], span: tuple[str | None, str | None] = (None, None)) -> pd.DataFrame:
    rows = {}
    for r in results:
        eq = r.equity.loc[span[0]:span[1]]
        if len(eq) < 40:
            continue
        eq = eq / eq.iloc[0]
        rows[r.name] = _metrics(eq, r.turnover)
    return pd.DataFrame(rows).T


def _fmt(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in ("CAGR", "Vol", "MaxDD"):
        if col in out:
            out[col] = (out[col] * 100).map(lambda v: f"{v:6.2f}%")
    for col in ("Sharpe", "Calmar", "Turnover/yr"):
        if col in out:
            out[col] = out[col].map(lambda v: f"{v:6.2f}")
    return out.to_string()


def ablate(px: pd.DataFrame, comps: pd.DataFrame, since: str | None = None, solo: bool = True,
          **kw) -> pd.DataFrame:
    def measure(result: Result) -> dict[str, float]:
        eq = result.equity.loc[since:] if since else result.equity
        eq = eq / eq.iloc[0] if len(eq) else eq
        row = {"From": eq.index[0].date().isoformat() if len(eq) else "-"}
        row.update(_metrics(eq, result.turnover))
        return row

    rows = {"(full index)": measure(run_strategy(px, ix.composite(comps), "full", **kw))}
    for name in ix.DEFAULT_WEIGHTS:
        if name not in comps.columns:
            continue
        drop_w = {c: w for c, w in ix.DEFAULT_WEIGHTS.items() if c != name}
        rows[f"drop {name}"] = measure(
            run_strategy(px, ix.composite(comps[list(drop_w)], drop_w), f"drop {name}", **kw))
        if solo:
            rows[f"only {name}"] = measure(
                run_strategy(px, ix.composite(comps[[name]], {name: 1.0}), f"only {name}", **kw))
    return pd.DataFrame(rows).T


def ladder_table(px: pd.DataFrame, kdi: pd.Series,
                 bands: tuple[float, ...] = (-1.5, -0.5, 0.5, 1.5)) -> pd.DataFrame:
    spread = px[dt.KC50].pct_change(fill_method=None) - px[dt.DIBO].pct_change(fill_method=None)
    fwd21 = spread[::-1].rolling(21).sum()[::-1].shift(-1)
    df = pd.DataFrame({"nti": kdi, "f21": fwd21}).dropna()
    edges = [-np.inf, *bands, np.inf]
    labels = ([f"< {bands[0]:+.1f}"] + [f"{a:+.1f} .. {b:+.1f}" for a, b in zip(bands, bands[1:])]
             + [f"> {bands[-1]:+.1f}"])
    grp = df.groupby(pd.cut(df.nti, edges, labels=labels), observed=False)
    out = pd.DataFrame({
        "days": grp.size(),
        "% time": (grp.size() / len(df) * 100).round(1),
        "fwd21 %": (grp.f21.mean() * 100).round(2),
    })
    out["target %KC50"] = [ix.RISK_OFF_WEIGHT * 100 if float(lbl.split()[-1]) <= ix.BUY_THRESHOLD
                           else ix.RISK_ON_WEIGHT * 100 for lbl in labels]
    return out


def selftest(px: pd.DataFrame, dates: list[str] | None = None) -> bool:
    full, _ = ix.build_index(px)
    dates = dates or ["2021-06-15", "2022-09-30", "2023-11-15", "2025-03-01", "2026-06-01"]
    ok = True
    for date in dates:
        truncated, _ = ix.build_index(px.loc[:date])
        a, b = full.loc[:date].dropna(), truncated.dropna()
        if a.empty or b.empty:
            print(f"  {date}: no reading (warm-up)")
            continue
        diff = abs(a.iloc[-1] - b.iloc[-1])
        status = "ok" if diff < 1e-10 else "LEAK"
        ok &= diff < 1e-10
        print(f"  {date}: full={a.iloc[-1]:+.6f}  point-in-time={b.iloc[-1]:+.6f}  {status}")
    return ok


def today_reading(kdi: pd.Series, comps: pd.DataFrame, k: float,
                  weights: dict[str, float], label: str) -> str:
    date = kdi.dropna().index[-1]
    value = kdi.loc[date]
    w = float(ix.target_weight(pd.Series([value]), k=k).iloc[0])
    lines = [f"{label} reading {date.date()}: {value:+.2f} sigma",
             f"  -> target {w * 100:.0f}% {dt.KC50} (KC50) / {(1 - w) * 100:.0f}% {dt.DIBO} (DIBO)",
             "", "  component            z      weight  contribution"]
    row = comps.loc[date]
    total_w = sum(weights[c] for c in comps.columns if c in weights and pd.notna(row[c]))
    for name in weights:
        if name not in comps.columns or pd.isna(row[name]):
            continue
        wt = weights[name]
        lines.append(f"  {name:<16} {row[name]:+6.2f}    {wt:4.2f}      {row[name] * wt / total_w:+6.3f}")
    return "\n".join(lines)


def _plot_profit(results: list[Result], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for r in results:
        eq = r.equity / r.equity.iloc[0]
        lw = 1.8 if "core" in r.name else 1.0
        ax1.plot(eq.index, (eq - 1) * 100, lw=lw, label=f"{r.name}   {(eq.iloc[-1] - 1) * 100:+,.0f}%")
        ax2.plot(eq.index, (eq / eq.cummax() - 1) * 100, lw=lw)
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_ylabel("cumulative profit %")
    ax1.set_title("KC50 vs Dividend-Low-Vol rotation — total profit since Sep 2020, net of costs")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.set_ylabel("drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=dt.DEFAULT_START)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--k", type=float, default=0.25)
    ap.add_argument("--rebalance", type=int, default=5)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--cost-bps", type=float, default=COST_BPS)
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--profit", metavar="PATH", nargs="?", const="china_rotation/kdi_profit.png")
    ap.add_argument("--csv", metavar="PATH", nargs="?", const="china_rotation/kdi_history.csv")
    args = ap.parse_args()

    px = dt.fetch_prices(start=args.start, refresh=args.refresh)

    if args.selftest:
        print("\n=== Point-in-time selftest (no lookahead) ===")
        return 0 if selftest(px) else 1

    kdi, comps = ix.build_index(px)
    kw = dict(k=args.k, rebalance=args.rebalance, lag=args.lag, binary=args.binary, cost_bps=args.cost_bps)

    strat = run_strategy(px, kdi, "KDI (all 8)", **kw)
    core_kdi = ix.composite(comps[list(ix.CORE_WEIGHTS)], ix.CORE_WEIGHTS)
    core = run_strategy(px, core_kdi, "KDI-core (4 factors)", **kw)
    core_ladder = run_strategy(px, core_kdi, "KDI-core 2-state", **{**kw, "ladder": True})

    idx = strat.equity.loc[TRADED_START:].index
    results = [strat, core, core_ladder,
              run_benchmark(px, "hold DIBO", 0.0, idx, args.rebalance, args.cost_bps),
              run_benchmark(px, "hold KC50", 1.0, idx, args.rebalance, args.cost_bps),
              run_benchmark(px, "50/50 rebalanced", 0.5, idx, args.rebalance, args.cost_bps)]
    avg_w = float(strat.weights.loc[TRADED_START:].mean())
    results.append(run_benchmark(px, f"static {avg_w * 100:.0f}% KC50", avg_w, idx,
                                 args.rebalance, args.cost_bps))
    results.append(run_strategy(px, ix.composite(comps[["valuation"]], {"valuation": 1.0}),
                                "ratio only (video)", **kw))

    for r in results:
        r.equity = r.equity.loc[TRADED_START:]
        r.equity = r.equity / r.equity.iloc[0]

    print(f"\n*** CAVEAT: traded history is {TRADED_START} -> {idx[-1].date()}, "
          f"~{len(idx) / TRADING_DAYS:.1f} years. Component selection here is far less "
          f"reliable than in tech_rotation/ (27y). See docstring. ***")

    print(f"\n=== Full traded sample  {TRADED_START} -> {idx[-1].date()} ===")
    print(_fmt(_table(results)))
    print(f"\n=== First half  {TRADED_START} -> {OOS_SPLIT} ===")
    print(_fmt(_table(results, (TRADED_START, OOS_SPLIT))))
    print(f"\n=== Second half  {OOS_SPLIT} -> {idx[-1].date()} ===")
    print(_fmt(_table(results, (OOS_SPLIT, None))))

    print("\n=== Timing only (average tilt removed) ===")
    for label, weights in (("KDI", strat.weights), ("KDI-core", core.weights)):
        timing = pd.DataFrame({
            "full": timing_stats(px, weights, (TRADED_START, None)),
            "1st half": timing_stats(px, weights, (TRADED_START, OOS_SPLIT)),
            "2nd half": timing_stats(px, weights, (OOS_SPLIT, None)),
        }).T
        print(f"-- {label}")
        print(timing.round(3).to_string())

    print(f"\nAverage KC50 weight {avg_w * 100:.1f}%, turnover {strat.turnover:.2f}x/yr")
    print("\n" + today_reading(kdi, comps, args.k, ix.DEFAULT_WEIGHTS, "KDI (all 8)"))
    print("\n" + today_reading(core_kdi, comps[list(ix.CORE_WEIGHTS)], args.k,
                               ix.CORE_WEIGHTS, "KDI-core (provisional)"))

    if args.ladder:
        print("\n=== What each reading level paid (KC50 minus DIBO, forward) ===")
        print(ladder_table(px, core_kdi).to_string())

    if args.ablate:
        print(f"\n=== Component ablation, drop-one, full traded sample ===")
        print(_fmt(ablate(px, comps, **kw)))
        print(f"\n=== Component ablation, drop-one, since {OOS_SPLIT} ===")
        print(_fmt(ablate(px, comps, since=OOS_SPLIT, solo=False, **kw)))

    if args.csv:
        out = comps.loc[TRADED_START:].copy()
        out["KDI"] = kdi
        out["KDI_core"] = core_kdi
        out["target_kc50"] = ix.target_weight(kdi, k=args.k)
        out.dropna(subset=["KDI"]).to_csv(args.csv)
        print(f"\nwrote {args.csv}")

    if args.profit:
        _plot_profit(results, args.profit)
        print(f"wrote {args.profit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
