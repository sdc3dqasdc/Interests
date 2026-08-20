"""Is rb=10d's out-of-sample advantage a frequency effect or grid luck?"""
import numpy as np, pandas as pd
from tech_rotation import data as dt, index as ix
from tech_rotation.backtest import TRADING_DAYS, OOS_SPLIT
from tech_rotation.backtest_product import run_banded, run_hold_qqq, PRODUCT_COST_BPS

px = dt.fetch_prices()
comps = ix.build_components(px)
core = ix.composite(comps[list(ix.CORE_WEIGHTS)], ix.CORE_WEIGHTS)

def cagr(eq):
    eq = eq / eq.iloc[0]
    return eq.iloc[-1] ** (TRADING_DAYS / len(eq)) - 1

rows = []
for rb in (5, 10, 15):
    for phase in range(rb):
        s = run_banded(px, core, 0.0335, band=0.20, risk_off=0.0,
                       rebalance=rb, phase=phase)
        q = run_hold_qqq(px, s.equity.index)
        oos_s = s.equity.loc[OOS_SPLIT:]
        oos_q = q.equity.loc[OOS_SPLIT:]
        e2022 = s.equity.loc["2022-01-01":"2022-12-31"]
        rows.append({
            "rb": rb, "phase": phase,
            "full_cagr": cagr(s.equity) * 100,
            "oos_edge": (cagr(oos_s) - cagr(oos_q)) * 100,
            "y2022": (e2022.iloc[-1] / e2022.iloc[0] - 1) * 100,
        })

df = pd.DataFrame(rows)
for rb, g in df.groupby("rb"):
    print(f"\n=== rb={rb}d, all {rb} phase offsets ===")
    print(g[["phase", "full_cagr", "oos_edge", "y2022"]].round(2).to_string(index=False))
    print(f"  OOS edge: mean {g.oos_edge.mean():+.2f}  min {g.oos_edge.min():+.2f}  "
          f"max {g.oos_edge.max():+.2f}  spread {g.oos_edge.max()-g.oos_edge.min():.2f}pt")
    print(f"  2022:     mean {g.y2022.mean():+.2f}%  min {g.y2022.min():+.2f}%  max {g.y2022.max():+.2f}%")
