"""Today's reading and today's decision for KC50 (STAR50) vs Dividend Low Vol.

    .venv/bin/python -m china_rotation.daily
    .venv/bin/python -m china_rotation.daily --current 0.50

Unlike tech_rotation.daily, this does NOT offer a two-state switch rule.  The
buy-ladder bucketing (see README) came out non-monotonic on this data -- the
most bearish readings paid positive forward, the most bullish paid negative --
which is a five-year sample doing what small samples do, not a real pattern.
So the only recommendation here is the continuous tilt off the full 8-factor
index, which is also the version that measured best in the backtest.
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import data as dt
from . import index as ix


def _sparkline(values: pd.Series) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = -2.5, 2.5
    return "".join(blocks[min(len(blocks) - 1,
                              max(0, int((v - lo) / (hi - lo) * len(blocks))))]
                   for v in values)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--current", type=float, default=None, metavar="W",
                    help="the KC50 weight you hold right now (0-1)")
    ap.add_argument("--band", type=float, default=0.10)
    ap.add_argument("--k", type=float, default=0.25)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--plot", metavar="PATH", nargs="?", const="china_rotation/kdi_today.png")
    ap.add_argument("--history", type=int, default=252)
    args = ap.parse_args()

    px = dt.fetch_prices(refresh=args.refresh)
    comps = ix.build_components(px)
    kdi = ix.composite(comps).dropna()   # full 8-factor -- the version that tested best here
    date = kdi.index[-1]
    value = float(kdi.iloc[-1])
    target = float(ix.target_weight(pd.Series([value]), k=args.k).iloc[0])

    print(f"\nKDI  {date.date()}   {value:+.2f} sigma   (Kechuang50 vs Hongli-Dibo)")
    print(f"  target:  {target * 100:3.0f}% {dt.KC50} (STAR50)  /  {(1 - target) * 100:3.0f}% "
          f"{dt.DIBO} (Div. Low Vol)")
    print("  *** only ~5.5 years of traded history behind this reading -- see README ***")

    week = kdi.iloc[-6] if len(kdi) > 6 else value
    month = kdi.iloc[-22] if len(kdi) > 22 else value
    print(f"  trend:   {value - week:+.2f} vs a week ago, {value - month:+.2f} vs a month ago")
    tail = kdi.iloc[-args.history:]
    print(f"  last {len(tail)} sessions: {_sparkline(tail.iloc[::max(1, len(tail) // 60)])}")

    print("\n  component          z    weight   contribution")
    row = comps.loc[date]
    total = sum(w for c, w in ix.DEFAULT_WEIGHTS.items() if pd.notna(row[c]))
    for name, w in ix.DEFAULT_WEIGHTS.items():
        if pd.isna(row[name]):
            continue
        print(f"  {name:<14} {row[name]:+6.2f}   {w:4.2f}       {row[name] * w / total:+6.3f}")

    if args.current is not None:
        move = target - args.current
        print(f"\n  you hold: {args.current * 100:3.0f}% {dt.KC50}")
        if abs(move) < args.band:
            print(f"  ACTION:   hold — target is {move * 100:+.0f}pt away, inside the "
                  f"{args.band * 100:.0f}pt no-trade band")
        else:
            verb = "buy" if move > 0 else "sell"
            print(f"  ACTION:   {verb} {abs(move) * 100:.0f}pt of {dt.KC50} "
                  f"({args.current * 100:.0f}% -> {target * 100:.0f}%), funded from {dt.DIBO}")

    print("\n  reminder: ~5.5 years of data, one dominant regime shift (2021-22 selloff,\n"
          "  2023-25 AI-driven recovery). Treat this as a much weaker read than the US\n"
          "  version of this model, which has 27 years and passed a lookahead selftest.")

    if args.plot:
        _plot(kdi.iloc[-args.history:], args.k, args.plot)
        print(f"\n  wrote {args.plot}")
    return 0


def _plot(kdi: pd.Series, k: float, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weight = ix.target_weight(kdi, k=k)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(kdi.index, kdi, color="#222", lw=1.2)
    ax1.axhline(0, color="k", lw=0.6)
    ax1.fill_between(kdi.index, 0, kdi.where(kdi > 0), color="tab:green", alpha=0.3)
    ax1.fill_between(kdi.index, 0, kdi.where(kdi < 0), color="tab:red", alpha=0.3)
    ax1.set_ylabel("KDI (sigma)")
    ax1.set_title(f"KDI through {kdi.index[-1].date()} — {kdi.iloc[-1]:+.2f}σ, "
                  f"target {weight.iloc[-1] * 100:.0f}% STAR50")
    ax1.grid(alpha=0.3)

    ax2.plot(weight.index, weight * 100, color="tab:blue", lw=1.4)
    ax2.axhline(50, color="k", lw=0.6, ls="--")
    ax2.set_ylim(-5, 105)
    ax2.set_ylabel("% STAR50")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    raise SystemExit(main())
