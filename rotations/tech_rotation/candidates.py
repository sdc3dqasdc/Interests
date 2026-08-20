"""Test the two new candidate components (sentiment, trend_50) against the
same bar CORE_WEIGHTS had to clear: does adding it to NTI-core help, and does
its own contribution have the same sign in both halves of the sample.

`sentiment` (CNN Fear & Greed) only has data since 2021-02-01, so it cannot be
checked against the pre-2013 half at all -- there is a third table for it,
"since 2021", instead of a real both-halves pass. Treat its result as weaker
evidence than everything else in this package, not as equivalent.

    .venv/bin/python -m tech_rotation.candidates
"""
from __future__ import annotations

import pandas as pd

from . import data as dt
from . import index as ix
from .backtest import (TRADING_DAYS, COST_BPS, OOS_SPLIT, _metrics, _fmt, _table, Result,
                       run_strategy, timing_stats)

FG_START = "2021-02-01"


def main() -> int:
    px = dt.fetch_prices()
    fg = dt.fetch_fear_greed()
    comps = ix.build_components(px, fear_greed=fg)

    core_nti = ix.composite(comps[list(ix.CORE_WEIGHTS)], ix.CORE_WEIGHTS)
    core = run_strategy(px, core_nti, "NTI-core (baseline, 4 factors)")

    variants = {}
    for cand in ("sentiment", "trend_50"):
        if cand not in comps.columns:
            print(f"  (skipping {cand}: no data)")
            continue
        weights = {**ix.CORE_WEIGHTS, cand: ix.CANDIDATE_WEIGHTS[cand]}
        nti = ix.composite(comps[list(weights)], weights)
        variants[f"core + {cand}"] = run_strategy(px, nti, f"core + {cand}")
        solo_nti = ix.composite(comps[[cand]], {cand: 1.0})
        variants[f"only {cand}"] = run_strategy(px, solo_nti, f"only {cand}")

    idx = core.equity.index
    results = [core, *variants.values()]

    print(f"\n=== Full sample {idx[0].date()} -> {idx[-1].date()} ===")
    print(_fmt(_table(results)))
    print(f"\n=== In sample -> {OOS_SPLIT} ===")
    print(_fmt(_table(results, (None, OOS_SPLIT))))
    print(f"\n=== Out of sample {OOS_SPLIT} -> ===")
    print(_fmt(_table(results, (OOS_SPLIT, None))))

    if "sentiment" in comps.columns:
        print(f"\n=== sentiment only has data since {FG_START} -- this window is the only fair test ===")
        print(_fmt(_table([r for r in results if "sentiment" in r.name or r.name == core.name],
                          (FG_START, None))))

    print("\n=== Timing only (average tilt removed) -- t-stat is the question that matters ===")
    for name, result in [("NTI-core", core), *variants.items()]:
        span = (FG_START, None) if "sentiment" in name else (None, None)
        stats = timing_stats(px, result.weights, span)
        if not stats:
            print(f"  {name}: not enough data in its fair window")
            continue
        print(f"  {name:<22} window={span[0] or idx[0].date()}->{span[1] or 'now'}  "
             f"CAGR={stats['Timing CAGR']*100:+.2f}%  IR={stats['Timing IR']:.2f}  "
             f"t={stats['t-stat']:.2f}  IC={stats['IC(21d)']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
