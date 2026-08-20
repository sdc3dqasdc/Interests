#!/usr/bin/env python3
"""A/B many screen/exit configurations against ONE load of the market data.

win_rate_tester.py answers "how did this configuration do?".  Running it once
per variant re-fetches bars, recomputes every technical signal and re-fetches
fundamentals each time — minutes of identical work to change one threshold.
This tool loads that data once (via win_rate_tester.load_market_data) and then
replays win_rate_tester.run_picks against each variant, so a 14-way comparison
costs one setup instead of fourteen.

It exists to answer questions like "does the market-regime gate actually help?"
with a measurement rather than an assumption.  Several rules in this pipeline
have never been tested at all — alpha_lab only ever measured price/volume
factors, so it has nothing to say about the ROE/FCF quality gates or the SPY
regime gate.  "Untested" is not "shown not to work", and those justify opposite
actions; this is how you tell them apart.

Usage:
    python3 ab_sweep.py --start 2022-01-01 --end 2026-02-01
    python3 ab_sweep.py --start 2022-01-01 --end 2026-02-01 --only baseline mom_12
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Any

import pandas as pd

import run_report as rr
import win_rate_tester as wrt


# Each variant is a set of overrides applied to the baseline args namespace.
# Keep baseline first: everything else is reported as a delta against it.
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline":        {},

    # ================== READ THIS BEFORE TRUSTING A WINNER ==================
    # This sweep ranks by per-pick avg ROI / excess over OVERLAPPING windows.
    # That metric repeatedly disagreed with the portfolio backtest, which
    # models position limits, rebalancing and compounding:
    #   - the sweep loved a wider stop (25) and wider ext band (25/-8); the
    #     backtest showed BOTH cut portfolio excess (+38% -> +23%). Reverted.
    #   - hold_30 was the sweep's TOP variant (+5.91 excess); in the backtest
    #     it COLLAPSED to +2.76% excess.
    # Lesson: treat a sweep winner as a hypothesis and confirm it in backtest.py
    # (a one-off `--hold-days`/`--stop-loss-pct`/... run) before adopting it.
    # baseline now runs the ORIGINAL tight defaults (stop 15, ext 15/-3), which
    # the backtest showed are best.
    # =======================================================================

    # --- exit mechanics ------------------------------------------------------
    "noStop":          {"stop_loss_pct": 0.0},            # sweep-neutral; backtest-negative
    "stop_20":         {"stop_loss_pct": 20.0},

    # --- entry/universe knobs (sweep-exploratory; verify any winner) ---------
    "regime_off":      {"market_regime": False},          # more picks, dilutes excess
    "rsi_30_70":       {"min_rsi": 30.0, "max_rsi": 70.0},
    "ext_wide":        {"max_ext_sma50": 25.0, "min_ext_sma50": -8.0},

    # dropped: rs_min5 & mom_win9 (byte-identical no-ops vs baseline last run),
    # dv_50m (-2.5pp dud), hold_30 (sweep +5.91 excess but backtest +2.76% —
    # the clearest example of why sweep winners need backtest confirmation),
    # tp_35 (sweep-positive but take-profits cap winners; not worth the churn).
}


def summarize(picks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Headline stats for one variant's picks."""
    if not picks:
        return {"variant": label, "picks": 0}
    d = pd.DataFrame(picks)
    n = len(d)
    p = d["win"].mean()
    se = math.sqrt(p * (1 - p) / n) if n else 0.0
    ex = d["excess_pct"].dropna()
    stops = d[d["exit_reason"] == "STOP"] if "exit_reason" in d else d.iloc[0:0]
    return {
        "variant": label, "picks": n,
        "win_pct": round(p * 100, 1),
        "ci_low": round(max(0.0, p - 1.96 * se) * 100, 1),
        "ci_high": round(min(1.0, p + 1.96 * se) * 100, 1),
        "avg_roi": round(d["roi_pct"].mean(), 2),
        "median_roi": round(d["roi_pct"].median(), 2),
        "excess_spy": round(ex.mean(), 2) if len(ex) else None,
        "beat_spy_pct": round((ex > 0).mean() * 100, 1) if len(ex) else None,
        "worst": round(d["roi_pct"].min(), 2),
        "stopped": len(stops),
    }


def rsi_shape(picks: list[dict[str, Any]]) -> pd.DataFrame | None:
    """In-band RSI vs realised return, for the rsi_zone scoring question.

    These picks already passed the screen's RSI filter, so this is the shape
    the scorer actually sees — which is the only shape it should be tuned on.
    A universe-wide IC includes RSI values the screen never admits."""
    d = pd.DataFrame(picks)
    if "rsi_14" not in d or d["rsi_14"].isna().all():
        return None
    d = d.dropna(subset=["rsi_14"])
    d["bucket"] = pd.cut(d["rsi_14"], [0, 40, 45, 50, 55, 60, 65, 100])
    g = d.groupby("bucket", observed=True).agg(
        picks=("roi_pct", "size"), avg_roi=("roi_pct", "mean"),
        win_pct=("win", lambda s: s.mean() * 100),
        dist_from_50=("rsi_14", lambda s: (s - 50).abs().mean()))
    return g.round(2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--only", nargs="*", default=None,
                   help="Run only these variants (default: all)")
    p.add_argument("--output", default="ab_sweep.csv")
    return p.parse_args()


def main() -> int:
    cli = parse_args()
    rr.begin("ab_sweep", cli)

    # Build the baseline namespace by calling win_rate_tester's OWN parser, so
    # every default (thresholds, hold, slippage) comes from one place and the
    # sweep cannot silently drift from what the single-run tool would do.
    argv = sys.argv
    sys.argv = ["win_rate_tester.py", "--start", cli.start, "--end", cli.end]
    try:
        base = wrt.parse_args()
    finally:
        sys.argv = argv

    print(f"Loading market data once for {len(VARIANTS)} variants …")
    hist, spy, info_map = wrt.load_market_data(base)
    base.fundamentals_available = bool(info_map) and any(info_map.values())

    names = [n for n in (cli.only or list(VARIANTS)) if n in VARIANTS]
    for bad in set(cli.only or []) - set(VARIANTS):
        print(f"  skipping unknown variant '{bad}'")

    all_picks: dict[str, list[dict[str, Any]]] = {}
    for i, name in enumerate(names, 1):
        print(f"  [{i}/{len(names)}] {name} …", flush=True)
        args = argparse.Namespace(**vars(base))
        for k, v in VARIANTS[name].items():
            setattr(args, k, v)
        all_picks[name] = wrt.run_picks(hist, spy, info_map, args, verbose=False)

    rows = [summarize(all_picks[name], name) for name in names]
    rep = pd.DataFrame(rows)
    rep.to_csv(cli.output, index=False)

    base_row = rep[rep["variant"] == "baseline"]
    b_ex = float(base_row["excess_spy"].iloc[0]) if len(base_row) else None
    b_lo = float(base_row["ci_low"].iloc[0]) if len(base_row) else None
    b_hi = float(base_row["ci_high"].iloc[0]) if len(base_row) else None

    print(f"\n{'variant':<14}{'picks':>6}{'win%':>7}{'95% CI':>14}{'avgROI':>8}"
          f"{'excess':>8}{'worst':>8}{'stops':>7}   vs baseline")
    print("-" * 88)
    for _, r in rep.sort_values("excess_spy", ascending=False, na_position="last").iterrows():
        if not r["picks"]:
            print(f"{r['variant']:<14}{'0':>6}   (no completed picks)")
            continue
        ci = f"{r['ci_low']:.1f}-{r['ci_high']:.1f}"
        verdict = ""
        if b_ex is not None and r["variant"] != "baseline" and r["excess_spy"] is not None:
            d = r["excess_spy"] - b_ex
            # Inside baseline's CI = not distinguishable from it on this sample.
            inside = b_lo is not None and b_lo <= r["win_pct"] <= b_hi
            verdict = f"{d:+.2f}pp {'(within CI)' if inside else '(outside CI)'}"
        print(f"{r['variant']:<14}{r['picks']:>6}{r['win_pct']:>7.1f}{ci:>14}"
              f"{r['avg_roi']:>8.2f}{r['excess_spy'] if r['excess_spy'] is not None else 0:>8.2f}"
              f"{r['worst']:>8.1f}{r['stopped']:>7}   {verdict}")

    rr.rows("ab_sweep", "variants", rep.to_dict("records"))

    if "baseline" in all_picks:
        shape = rsi_shape(all_picks["baseline"])
        if shape is not None:
            print("\n--- RSI(14) at entry vs realised return, INSIDE the screen band ---")
            print(shape.to_string())
            rr.rows("ab_sweep", "rsi_shape_in_band",
                    shape.reset_index().astype({"bucket": str}).to_dict("records"))

    rr.note("ab_sweep", "caveats",
            "Pick windows overlap, so picks are correlated and every CI here is optimistic. "
            "Variants whose win rate falls inside the baseline CI are not distinguishable "
            "from baseline on this sample.")
    print(f"\nComparison written to {cli.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())