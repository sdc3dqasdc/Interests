"""Backtest for the Nasdaq Tilt Index.

Rules of the harness, so the numbers mean something:

  * The reading for day t uses only bars up to and including t, and is traded
    at day t+1's close.  Every z-score inside the index is a trailing window.
  * Rebalance weekly (default).  Costs are charged on turnover at 5bp per side.
  * The comparison is not "did it make money" -- QQQ made money.  It is whether
    it beat the three things you could have done without an index: hold SPY,
    hold QQQ, or hold a rebalanced 50/50.
  * The split matters more than the headline.  1999-2012 covers a bust and a
    crash; 2013-onward covers the run where nothing beat just holding QQQ.  A
    rotation rule that only works in the first half is a rule fitted to the
    bust.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data as dt
from . import index as ix

TRADING_DAYS = 252
COST_BPS = 5.0          # per unit of turnover, one side
OOS_SPLIT = "2013-01-01"
# The first date on which every component is live and warmed up (HYG lists in
# 2007, plus two years of z warm-up and one of re-scaling).
ALL_LIVE = "2010-01-01"


@dataclass
class Result:
    name: str
    equity: pd.Series
    weights: pd.Series
    turnover: float

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()


def _metrics(equity: pd.Series, turnover: float = float("nan")) -> dict[str, float]:
    ret = equity.pct_change().dropna()
    if ret.empty:
        return {}
    years = len(ret) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    dd = (equity / equity.cummax() - 1).min()
    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": (ret.mean() * TRADING_DAYS) / vol if vol else np.nan,
        "MaxDD": dd,
        "Calmar": cagr / abs(dd) if dd else np.nan,
        "Turnover/yr": turnover,
    }


def run_strategy(
    px: pd.DataFrame,
    nti: pd.Series,
    name: str,
    k: float = 0.25,
    rebalance: int = 5,
    lag: int = 1,
    binary: bool = False,
    ladder: bool = False,
    cost_bps: float = COST_BPS,
) -> Result:
    """Hold `w` in QQQ and `1-w` in SPY, where w comes from the index.

    `lag` is the number of sessions between the reading and the trade; 1 is
    "read tonight's close, trade tomorrow's".  Higher values are a robustness
    check, not a realistic alternative."""
    tech_ret = px[dt.TECH].pct_change()
    steady_ret = px[dt.STEADY].pct_change()

    if ladder:
        target = ix.ladder_weight(nti)
    elif binary:
        target = (nti > 0).astype(float).where(nti.notna())
    else:
        target = ix.target_weight(nti, k=k)

    # Trade only on rebalance dates, and only on information that was already
    # public: shift by `lag` sessions before anything is acted on.
    traded = target.shift(lag)
    schedule = pd.Series(np.arange(len(traded)) % rebalance == 0, index=traded.index)
    weights = traded.where(schedule).ffill()

    valid = weights.notna() & tech_ret.notna() & steady_ret.notna()
    weights, tech_ret, steady_ret = weights[valid], tech_ret[valid], steady_ret[valid]

    gross = weights * tech_ret + (1 - weights) * steady_ret
    # Turnover is the weight actually moved; both legs move, hence the 2x.
    turn = weights.diff().abs().fillna(weights.iloc[0]) * 2
    net = gross - turn * cost_bps / 10_000.0

    equity = (1 + net).cumprod()
    turnover_yr = turn.sum() / (len(turn) / TRADING_DAYS)
    return Result(name, equity, weights, turnover_yr)


def run_benchmark(px: pd.DataFrame, name: str, w_tech: float, index: pd.Index,
                  rebalance: int = 5, cost_bps: float = COST_BPS) -> Result:
    """Static blend, rebalanced on the same schedule so the comparison is fair."""
    const = pd.Series(w_tech, index=px.index)
    tech_ret, steady_ret = px[dt.TECH].pct_change(), px[dt.STEADY].pct_change()
    weights = const.reindex(index)
    tech_ret, steady_ret = tech_ret.reindex(index), steady_ret.reindex(index)
    gross = weights * tech_ret + (1 - weights) * steady_ret
    # A 0/1 blend never trades after entry; a 50/50 pays drift back to target.
    turn = (weights.diff().abs().fillna(weights.iloc[0]) * 2)
    if 0.0 < w_tech < 1.0:
        drift = (tech_ret - steady_ret).abs() * w_tech * (1 - w_tech)
        turn = turn + drift.where(pd.Series(np.arange(len(index)) % rebalance == 0,
                                            index=index), 0.0)
    net = gross - turn * cost_bps / 10_000.0
    equity = (1 + net.dropna()).cumprod()
    return Result(name, equity, weights, turn.sum() / (len(index) / TRADING_DAYS))


def timing_stats(px: pd.DataFrame, weights: pd.Series,
                 span: tuple[str | None, str | None] = (None, None)) -> dict[str, float]:
    """Strip out the average tilt and measure only the timing.

    A rule that sits at 70% QQQ on average will beat SPY in any period QQQ won,
    with no skill involved.  The timing P&L is the part that survives removing
    the average: sum over days of (w_t - w_bar) * (QQQ - SPY) return.  Its
    t-statistic is the question "is this distinguishable from luck".

    IC is the rank correlation between the reading and the next 21 sessions of
    QQQ-minus-SPY return -- the same question asked without any position sizing."""
    spread = (px[dt.TECH].pct_change() - px[dt.STEADY].pct_change()).reindex(weights.index)
    w = weights.loc[span[0]:span[1]]
    sp = spread.loc[span[0]:span[1]]
    active = (w - w.mean()) * sp
    active = active.dropna()
    if len(active) < TRADING_DAYS:
        return {}
    mean_ann = active.mean() * TRADING_DAYS
    vol_ann = active.std() * np.sqrt(TRADING_DAYS)
    tstat = active.mean() / (active.std() / np.sqrt(len(active))) if active.std() else np.nan

    fwd = sp[::-1].rolling(21).sum()[::-1].shift(-1)
    # rank correlation, computed without scipy
    ic = w.rank().corr(fwd.rank())
    return {
        "Timing CAGR": mean_ann,
        "Timing IR": mean_ann / vol_ann if vol_ann else np.nan,
        "t-stat": tstat,
        "IC(21d)": ic,
        "Hit rate": float((np.sign(w - w.mean()) == np.sign(sp)).reindex(active.index).mean()),
    }


def _table(results: list[Result], span: tuple[str | None, str | None] = (None, None)) -> pd.DataFrame:
    rows = {}
    for r in results:
        eq = r.equity.loc[span[0]:span[1]]
        if len(eq) < TRADING_DAYS // 2:
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


def _period_table(results: list[Result], periods: dict[str, tuple[str, str]]) -> pd.DataFrame:
    rows = {}
    for label, (start, end) in periods.items():
        for r in results:
            eq = r.equity.loc[start:end]
            if len(eq) < 40:
                continue
            rows.setdefault(label, {})[r.name] = eq.iloc[-1] / eq.iloc[0] - 1
    return pd.DataFrame(rows).T


def ablate(px: pd.DataFrame, comps: pd.DataFrame, since: str | None = None,
           solo: bool = True, **kw) -> pd.DataFrame:
    """What each component is worth: drop it, and run it alone.

    A component whose drop-one Sharpe is no worse than the full index is not
    paying for its complexity.  `since` re-bases every curve to that date so the
    same test can be read out of sample.

    Every curve is measured over the SAME window, which matters more here than
    it looks: HYG lists in 2007 and RSP in 2003, so a credit-only or
    concentration-only variant measured from its own inception simply skips the
    dot-com bust and posts a flattering number that has nothing to do with the
    component.  The window used is printed as `From`."""
    def measure(result: Result) -> dict[str, float]:
        eq = _rebase(result.equity, since) if since else result.equity
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


def ladder_table(px: pd.DataFrame, nti: pd.Series,
                 bands: tuple[float, ...] = (-1.5, -0.5, 0.5, 1.5)) -> pd.DataFrame:
    """What each reading level actually paid: bucket every historical reading and
    measure the NEXT 21 and 63 sessions of QQQ-minus-SPY return.

    Split in and out of sample, because a band that only pays in one half is the
    dot-com bust wearing a threshold as a disguise."""
    spread = px[dt.TECH].pct_change() - px[dt.STEADY].pct_change()
    fwd21 = spread[::-1].rolling(21).sum()[::-1].shift(-1)
    fwd63 = spread[::-1].rolling(63).sum()[::-1].shift(-1)
    df = pd.DataFrame({"nti": nti, "f21": fwd21, "f63": fwd63}).dropna()

    edges = [-np.inf, *bands, np.inf]
    labels = ([f"< {bands[0]:+.1f}"]
              + [f"{a:+.1f} .. {b:+.1f}" for a, b in zip(bands, bands[1:])]
              + [f"> {bands[-1]:+.1f}"])
    rows = {}
    for name, sub in (("full", df), ("IS", df.loc[:OOS_SPLIT]), ("OOS", df.loc[OOS_SPLIT:])):
        grp = sub.groupby(pd.cut(sub.nti, edges, labels=labels), observed=False)
        for band in labels:
            rows.setdefault(band, {})["% time"] = round(len(sub[sub.nti.between(
                edges[labels.index(band)], edges[labels.index(band) + 1])]) / len(sub) * 100, 1) \
                if name == "full" else rows[band]["% time"]
            rows[band][f"{name} fwd21%"] = round(grp.f21.mean().get(band, np.nan) * 100, 2)
            rows[band][f"{name} fwd63%"] = round(grp.f63.mean().get(band, np.nan) * 100, 2)
    out = pd.DataFrame(rows).T.loc[labels]
    out["target %QQQ"] = [int(ix.RISK_OFF_WEIGHT * 100) if float(lbl.split()[-1]) <= ix.BUY_THRESHOLD
                          else int(ix.RISK_ON_WEIGHT * 100) for lbl in labels]
    return out


def sensitivity(px: pd.DataFrame, nti: pd.Series) -> pd.DataFrame:
    """Does the result survive the arbitrary choices? (tilt size, rebalance
    frequency, execution lag)."""
    rows = {}
    for k in (0.10, 0.25, 0.40, 0.50):
        r = run_strategy(px, nti, f"k={k}", k=k)
        rows[f"tilt k={k:.2f}"] = _metrics(r.equity, r.turnover)
    for rb in (1, 5, 10, 21):
        r = run_strategy(px, nti, f"rb={rb}", rebalance=rb)
        rows[f"rebalance={rb}d"] = _metrics(r.equity, r.turnover)
    for lag in (1, 2, 5, 10):
        r = run_strategy(px, nti, f"lag={lag}", lag=lag)
        rows[f"lag={lag}d"] = _metrics(r.equity, r.turnover)
    r = run_strategy(px, nti, "binary", binary=True)
    rows["binary switch"] = _metrics(r.equity, r.turnover)
    return pd.DataFrame(rows).T


def today_reading(nti: pd.Series, comps: pd.DataFrame, k: float,
                  weights: dict[str, float], label: str) -> str:
    date = nti.dropna().index[-1]
    value = nti.loc[date]
    w = float(ix.target_weight(pd.Series([value]), k=k).iloc[0])
    lines = [
        f"{label} reading {date.date()}: {value:+.2f} sigma",
        f"  -> target {w * 100:.0f}% {dt.TECH} / {(1 - w) * 100:.0f}% {dt.STEADY}",
        "",
        "  component            z      weight  contribution",
    ]
    row = comps.loc[date]
    total_w = sum(weights[c] for c in comps.columns if c in weights and pd.notna(row[c]))
    for name in weights:
        if name not in comps.columns or pd.isna(row[name]):
            continue
        wt = weights[name]
        lines.append(f"  {name:<16} {row[name]:+6.2f}    {wt:4.2f}      {row[name] * wt / total_w:+6.3f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=dt.DEFAULT_START)
    ap.add_argument("--refresh", action="store_true", help="refetch prices, ignore today's cache")
    ap.add_argument("--k", type=float, default=0.25, help="tilt per sigma (0.25 = all-in at 2 sigma)")
    ap.add_argument("--rebalance", type=int, default=5, help="sessions between rebalances")
    ap.add_argument("--lag", type=int, default=1, help="sessions between reading and trade")
    ap.add_argument("--cost-bps", type=float, default=COST_BPS)
    ap.add_argument("--binary", action="store_true", help="switch fully instead of tilting")
    ap.add_argument("--ablate", action="store_true", help="drop-one / only-one component tests")
    ap.add_argument("--sensitivity", action="store_true", help="parameter robustness grid")
    ap.add_argument("--selftest", action="store_true",
                    help="point-in-time check that the index uses no future data")
    ap.add_argument("--plot", metavar="PATH", nargs="?", const="tech_rotation/nti_equity.png")
    ap.add_argument("--profit", metavar="PATH", nargs="?", const="tech_rotation/nti_profit.png",
                    help="chart cumulative profit %% and drawdown")
    ap.add_argument("--ladder", action="store_true",
                    help="what each reading level paid, in and out of sample")
    ap.add_argument("--csv", metavar="PATH", nargs="?", const="tech_rotation/nti_history.csv")
    args = ap.parse_args()

    px = dt.fetch_prices(start=args.start, refresh=args.refresh)

    if args.selftest:
        print("\n=== Point-in-time selftest (no lookahead) ===")
        return 0 if selftest(px) else 1

    nti, comps = ix.build_index(px)

    kw = dict(k=args.k, rebalance=args.rebalance, lag=args.lag,
              binary=args.binary, cost_bps=args.cost_bps)
    strat = run_strategy(px, nti, "NTI rotation", **kw)
    idx = strat.equity.index
    results = [
        strat,
        run_benchmark(px, f"hold {dt.STEADY}", 0.0, idx, args.rebalance, args.cost_bps),
        run_benchmark(px, f"hold {dt.TECH}", 1.0, idx, args.rebalance, args.cost_bps),
        run_benchmark(px, "50/50 rebalanced", 0.5, idx, args.rebalance, args.cost_bps),
        # The control that actually matters: the same average Nasdaq exposure,
        # held blindly.  Anything the index earns over this line is timing;
        # anything below it was just beta.
        run_benchmark(px, f"static {strat.weights.mean() * 100:.0f}% {dt.TECH}",
                      float(strat.weights.mean()), idx, args.rebalance, args.cost_bps),
    ]
    # The trimmed index, and the video's ratio on its own as the honest control.
    core_nti = ix.composite(comps[list(ix.CORE_WEIGHTS)], ix.CORE_WEIGHTS)
    core = run_strategy(px, core_nti, "NTI-core (4 factors)", **kw)
    results.append(core)
    results.append(run_strategy(px, core_nti, "NTI-core 2-state", **{**kw, "ladder": True}))
    results.append(run_strategy(px, ix.composite(comps[["valuation"]], {"valuation": 1.0}),
                                "ratio only (video)", **kw))

    span = f"{idx[0].date()} -> {idx[-1].date()}"
    print(f"\n=== Full sample  {span}  ({len(idx) / TRADING_DAYS:.1f}y) ===")
    print(_fmt(_table(results)))

    print(f"\n=== In sample  {idx[0].date()} -> {OOS_SPLIT} ===")
    print(_fmt(_table(results, (None, OOS_SPLIT))))
    print(f"\n=== Out of sample  {OOS_SPLIT} -> {idx[-1].date()} ===")
    print(_fmt(_table(results, (OOS_SPLIT, None))))

    print("\n=== Total return by episode ===")
    episodes = {
        "dot-com bust 00-02": ("2000-03-01", "2002-10-09"),
        "GFC 07-09": ("2007-10-09", "2009-03-09"),
        "tech run 13-21": ("2013-01-01", "2021-12-31"),
        "rate shock 2022": ("2022-01-01", "2022-12-31"),
        "AI run 23-now": ("2023-01-01", None),
    }
    ep = _period_table(results, {k: v for k, v in episodes.items()})
    print((ep * 100).round(1).to_string() + "   (%)")

    print("\n=== Timing only (average tilt removed) ===")
    for label, weights in (("NTI", strat.weights), ("NTI-core", core.weights)):
        timing = pd.DataFrame({
            "full sample": timing_stats(px, weights),
            "in sample": timing_stats(px, weights, (None, OOS_SPLIT)),
            "out of sample": timing_stats(px, weights, (OOS_SPLIT, None)),
        }).T
        print(f"-- {label}")
        print(timing.round(3).to_string())

    exposure = strat.weights
    print(f"\nAverage {dt.TECH} weight {exposure.mean() * 100:.1f}% "
          f"(min {exposure.min() * 100:.0f}%, max {exposure.max() * 100:.0f}%), "
          f"turnover {strat.turnover:.2f}x/yr")

    print("\n" + today_reading(nti, comps, args.k, ix.DEFAULT_WEIGHTS, "NTI (all 9)"))
    print("\n" + today_reading(core_nti, comps[list(ix.CORE_WEIGHTS)], args.k,
                               ix.CORE_WEIGHTS, "NTI-core (recommended)"))

    if args.ablate:
        # Every component exists and is warm by 2010, so this is the only window
        # in which the one-component variants can be compared with each other.
        print(f"\n=== Component ablation, common window since {ALL_LIVE} ===")
        print(_fmt(ablate(px, comps, since=ALL_LIVE, **kw)))
        print("\n=== Component ablation, drop-one, full sample ===")
        print(_fmt(ablate(px, comps, solo=False, **kw)))
        print(f"\n=== Component ablation, drop-one, since {OOS_SPLIT} ===")
        print(_fmt(ablate(px, comps, since=OOS_SPLIT, solo=False, **kw)))

    if args.ladder:
        print("\n=== What each reading level paid (QQQ minus SPY, forward) ===")
        print(ladder_table(px, core_nti).to_string())
        print(f"\n  buy threshold {ix.BUY_THRESHOLD:+.1f} sigma: "
              f"{ix.RISK_ON_WEIGHT:.0%} {dt.TECH} above it, {ix.RISK_OFF_WEIGHT:.0%} below.")

    if args.sensitivity:
        print("\n=== Parameter sensitivity, NTI (full sample) ===")
        print(_fmt(sensitivity(px, nti)))
        print("\n=== Parameter sensitivity, NTI-core (full sample) ===")
        print(_fmt(sensitivity(px, core_nti)))

    if args.csv:
        out = comps.copy()
        out["NTI"] = nti
        out["NTI_core"] = core_nti
        out["target_qqq"] = ix.target_weight(nti, k=args.k)
        out["equity_nti"] = strat.equity
        out.dropna(subset=["NTI"]).to_csv(args.csv)
        print(f"\nwrote {args.csv}")

    if args.plot:
        _plot(results, nti, args.plot)
        print(f"wrote {args.plot}")

    if args.profit:
        _plot_profit(results, args.profit)
        print(f"wrote {args.profit}")
    return 0


def selftest(px: pd.DataFrame, dates: list[str] | None = None) -> bool:
    """Point-in-time check: rebuild the index from a truncated price history and
    confirm the reading for that date is identical to the one the full-history
    run produces.

    This is the test that catches lookahead.  Any centred window, any full-sample
    mean or standard deviation, any bfill would make the truncated value differ
    from the full one -- and a backtest built on the full-history version would
    be quietly using tomorrow's data."""
    full, _ = ix.build_index(px)
    dates = dates or ["2005-06-15", "2011-09-30", "2018-03-15", "2022-06-30", "2024-11-01"]
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


def _rebase(equity: pd.Series, start: str) -> pd.Series:
    eq = equity.loc[start:]
    return eq / eq.iloc[0]


def _plot_profit(results: list[Result], path: str) -> None:
    """Cumulative profit in per cent, with the drawdown underneath.

    Log scale on the profit panel: over 27 years the lines span +740% to +2000%,
    and on a linear axis the last five years would be the only visible part."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for r in results:
        eq = r.equity / r.equity.iloc[0]
        lw = 1.8 if "core" in r.name else 1.0
        ax1.plot(eq.index, (eq - 1) * 100, lw=lw,
                 label=f"{r.name}   {(eq.iloc[-1] - 1) * 100:+,.0f}%")
        ax2.plot(eq.index, (eq / eq.cummax() - 1) * 100, lw=lw)

    ax1.set_yscale("symlog", linthresh=100)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}%"))
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_ylabel("cumulative profit")
    ax1.set_title("Total profit since March 1999, net of costs")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3, which="both")

    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.set_ylabel("drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def _plot(results: list[Result], nti: pd.Series, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for r in results:
        ax1.plot(r.equity.index, r.equity, lw=1.6 if "NTI" in r.name else 1.0,
                 label=f"{r.name}  x{r.equity.iloc[-1]:.1f}")
    ax1.set_yscale("log")
    ax1.set_ylabel("growth of $1 (log)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_title("Nasdaq Tilt Index — QQQ/SPY rotation vs holding either")

    # Show the signal only over the traded span; the earlier warm-up years exist
    # but had no QQQ to trade.
    nti = nti.loc[results[0].equity.index[0]:results[0].equity.index[-1]]
    ax2.plot(nti.index, nti, lw=0.8, color="#444")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.fill_between(nti.index, 0, nti.where(nti > 0), color="tab:green", alpha=0.35)
    ax2.fill_between(nti.index, 0, nti.where(nti < 0), color="tab:red", alpha=0.35)
    ax2.set_ylabel("NTI (sigma)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    raise SystemExit(main())
