"""Today's reading and today's decision — the morning command.

The backtest module answers "does this work". This one answers "what do I hold",
without recomputing 27 years of equity curves.  Same index, same point-in-time
rules; it just reads the last row.

    .venv/bin/python -m tech_rotation.daily
    .venv/bin/python -m tech_rotation.daily --current 0.70   # what I hold now
    .venv/bin/python -m tech_rotation.daily --plot           # last year's chart

A word on cadence: the index is a slow allocation dial, not a day-trade signal.
Check it daily if you like; act on it weekly at most.

At the 5bp costs backtest.py assumes, the result is flat across rebalance
frequencies from daily to monthly, so cadence is a free choice.  At the 1%/side
a retail product actually charges (backtest_product.py) it stops being free:
phase-averaged, slower is strictly better, and weekly is the worst of 1/2/3
weeks.  Weekly is kept here as the decision cadence because it matches how
often the numbers get looked at, not because it tested best — if you are paying
1% a side, acting fortnightly or every three weeks is the better trade.
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import data as dt
from . import index as ix


def _sparkline(values: pd.Series) -> str:
    """Rough trajectory of the reading, oldest to newest."""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = -2.5, 2.5
    return "".join(blocks[min(len(blocks) - 1,
                              max(0, int((v - lo) / (hi - lo) * len(blocks))))]
                   for v in values)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--current", type=float, default=None, metavar="W",
                    help="the QQQ weight you hold right now (0-1), to get a trade or a hold")
    ap.add_argument("--band", type=float, default=0.25,
                    help="no-trade band: act only if the target moved more than this (default 0.25). "
                         "Only bites in --tilt mode; the switch rule's moves are always the full "
                         "risk-on/risk-off gap, which no band below that gap can block")
    ap.add_argument("--risk-off", type=float, default=ix.RISK_OFF_WEIGHT,
                    help="fraction of QQQ held in the defensive state "
                         f"(default {ix.RISK_OFF_WEIGHT:.2f}; pass 0 to go fully to the steady leg)")
    ap.add_argument("--k", type=float, default=0.25, help="tilt per sigma")
    ap.add_argument("--tilt", action="store_true",
                    help="size the trade off the continuous tilt instead of the switch rule")
    ap.add_argument("--full", action="store_true",
                    help="also show the 9-factor index (the 4-factor core is the tested one)")
    ap.add_argument("--refresh", action="store_true", help="refetch prices, ignore today's cache")
    ap.add_argument("--plot", metavar="PATH", nargs="?", const="tech_rotation/nti_today.png",
                    help="chart the last year of the reading and the target weight")
    ap.add_argument("--history", type=int, default=252, help="sessions to chart")
    args = ap.parse_args()

    px = dt.fetch_prices(refresh=args.refresh)
    comps = ix.build_components(px)
    core = ix.composite(comps[list(ix.CORE_WEIGHTS)], ix.CORE_WEIGHTS).dropna()
    date = core.index[-1]
    value = float(core.iloc[-1])
    target = float(ix.target_weight(pd.Series([value]), k=args.k).iloc[0])

    ladder = float(ix.ladder_weight(pd.Series([value]), risk_off=args.risk_off).iloc[0])
    side = "RISK-ON" if value > ix.BUY_THRESHOLD else "DEFENSIVE"

    print(f"\nNTI-core  {date.date()}   {value:+.2f} sigma   [{side}]")
    print(f"  buy:     {ladder * 100:3.0f}% {dt.TECH}  /  {(1 - ladder) * 100:3.0f}% {dt.STEADY}"
          f"   (switch rule, threshold {ix.BUY_THRESHOLD:+.1f})")
    print(f"  or:      {target * 100:3.0f}% {dt.TECH}  /  {(1 - target) * 100:3.0f}% {dt.STEADY}"
          f"   (continuous tilt — same Sharpe at 5bp, worse at 1%/side)")

    week, month = core.iloc[-6] if len(core) > 6 else value, core.iloc[-22] if len(core) > 22 else value
    print(f"  trend:   {value - week:+.2f} vs a week ago, {value - month:+.2f} vs a month ago")
    tail = core.iloc[-args.history:]
    print(f"  last {len(tail)} sessions: {_sparkline(tail.iloc[::max(1, len(tail) // 60)])}")

    print("\n  component          z    weight   contribution")
    row = comps.loc[date]
    total = sum(w for c, w in ix.CORE_WEIGHTS.items() if pd.notna(row[c]))
    for name, w in ix.CORE_WEIGHTS.items():
        if pd.isna(row[name]):
            continue
        print(f"  {name:<14} {row[name]:+6.2f}   {w:4.2f}       {row[name] * w / total:+6.3f}")

    if args.full:
        nti = ix.composite(comps).dropna()
        v = float(nti.iloc[-1])
        w = float(ix.target_weight(pd.Series([v]), k=args.k).iloc[0])
        print(f"\n  (untrimmed 9-factor index: {v:+.2f} sigma -> {w * 100:.0f}% {dt.TECH}; "
              f"five of its factors failed the both-halves test — see README)")

    if args.current is not None:
        move = (target if args.tilt else ladder) - args.current
        print(f"\n  you hold: {args.current * 100:3.0f}% {dt.TECH}")
        if abs(move) < args.band:
            print(f"  ACTION:   hold — target is {move * +100:+.0f}pt away, inside the "
                  f"{args.band * 100:.0f}pt no-trade band")
        else:
            verb = "buy" if move > 0 else "sell"
            print(f"  ACTION:   {verb} {abs(move) * 100:.0f}pt of {dt.TECH} "
                  f"({args.current * 100:.0f}% -> "
                  f"{(target if args.tilt else ladder) * 100:.0f}%), funded from {dt.STEADY}")

    print("\n  reminder: t-stat on the timing edge is ~1.4 over 27 years. This sizes "
          "exposure;\n  it does not predict tomorrow.")

    if args.plot:
        _plot(core.iloc[-args.history:], args.k, args.plot)
        print(f"\n  wrote {args.plot}")
    return 0


def _plot(core: pd.Series, k: float, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weight = ix.target_weight(core, k=k)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(core.index, core, color="#222", lw=1.2)
    ax1.axhline(0, color="k", lw=0.6)
    ax1.fill_between(core.index, 0, core.where(core > 0), color="tab:green", alpha=0.3)
    ax1.fill_between(core.index, 0, core.where(core < 0), color="tab:red", alpha=0.3)
    ax1.set_ylabel("NTI-core (sigma)")
    ax1.set_title(f"NTI-core through {core.index[-1].date()} — "
                  f"{core.iloc[-1]:+.2f}σ, target {weight.iloc[-1] * 100:.0f}% {dt.TECH}")
    ax1.grid(alpha=0.3)

    ax2.plot(weight.index, weight * 100, color="tab:blue", lw=1.4)
    ax2.axhline(50, color="k", lw=0.6, ls="--")
    ax2.set_ylim(-5, 105)
    ax2.set_ylabel(f"% {dt.TECH}")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    raise SystemExit(main())
