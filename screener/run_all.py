#!/usr/bin/env python3
"""Run the whole pipeline as ONE test, into one consolidated report.

Wipes run_report.csv / run_report.md, then runs each tool in dependency order
with RUN_REPORT_SESSION set so they all append to the same report instead of
each wiping it.  The result is a single document describing the entire test:
every tool's parameters, headline metrics and key tables, in one place.

Stages (skip any with --skip):
    screener        the live screen; writes screener_universe.txt + candidates
    select_top15    ranks those candidates into top_candidates.csv
    roi_tester      tracks how the current picks have actually done
    alpha_lab       measures factor IC/ICIR on your universe
    win_rate_tester replays screen -> rank -> top-N picks historically
    backtest        portfolio simulation vs SPY

The screener needs sentiment input, so it is skipped unless you pass
--with-screener (it is the only stage that changes the universe file).

Usage:
    python3 run_all.py --start 2022-01-01 --end 2026-02-01
    python3 run_all.py --start 2022-01-01 --end 2026-02-01 --skip alpha_lab
    python3 run_all.py --start 2022-01-01 --end 2026-02-01 --with-screener
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import run_report as rr

STAGES = ["screener", "select_top15", "roi_tester", "alpha_lab", "win_rate_tester", "backtest"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="Start date for the historical stages")
    p.add_argument("--end", required=True, help="End date for the historical stages")
    p.add_argument("--top", type=int, default=5, help="Picks per date for win_rate_tester")
    p.add_argument("--hold-days", type=int, default=50)  # 10-week hold
    p.add_argument("--skip", nargs="*", default=[], choices=STAGES,
                   help="Stages to skip")
    p.add_argument("--with-screener", action="store_true",
                   help="Also run the live screener (rewrites screener_universe.txt)")
    p.add_argument("--refresh-cache", action="store_true",
                   help="Force every stage to refetch instead of reusing today's cache")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra flags passed through to win_rate_tester and backtest")
    return p.parse_args()


def build_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    py = sys.executable
    refresh = ["--refresh-cache"] if args.refresh_cache else []
    window = ["--start", args.start, "--end", args.end]
    cmds: list[tuple[str, list[str]]] = [
        ("screener", [py, "short_term_screener_alpaca.py", "--universe", "nasdaq-nyse",
                      "--cnn-sentiment"] + refresh),
        ("select_top15", [py, "select_top15.py"]),
        ("roi_tester", [py, "roi_tester.py", "--top", str(args.top),
                        "--hold-days", str(args.hold_days)] + refresh),
        ("alpha_lab", [py, "alpha_lab.py"] + window + ["--horizon", str(args.hold_days)] + refresh),
        ("win_rate_tester", [py, "win_rate_tester.py"] + window
         + ["--top", str(args.top), "--hold-days", str(args.hold_days)]
         + refresh + list(args.extra)),
        ("backtest", [py, "backtest.py"] + window
         + ["--hold-days", str(args.hold_days)] + refresh + list(args.extra)),
    ]
    skip = set(args.skip)
    if not args.with_screener:
        skip.add("screener")
    return [(name, cmd) for name, cmd in cmds if name not in skip]


def main() -> int:
    args = parse_args()
    session = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-pipeline"

    # Wipe first, then hand the session id to every child so they append.
    rr.reset(session)
    env = dict(os.environ, RUN_REPORT_SESSION=session)

    commands = build_commands(args)
    print(f"Test run {session}: {len(commands)} stages -> {rr.REPORT_MD}\n")
    rr.metrics("run_all", "test", {
        "session": session, "start": args.start, "end": args.end,
        "top": args.top, "hold_days": args.hold_days,
        "stages": ", ".join(name for name, _ in commands),
        "refresh_cache": args.refresh_cache,
        "extra_flags": " ".join(args.extra) if args.extra else "",
    })

    results = []
    for i, (name, cmd) in enumerate(commands, 1):
        print(f"[{i}/{len(commands)}] {name}: {' '.join(cmd[1:])}", flush=True)
        proc = subprocess.run(cmd, env=env)
        status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        results.append({"stage": name, "status": status, "exit_code": proc.returncode})
        print(f"    -> {status}\n", flush=True)

    rr.rows("run_all", "stages", results)
    failed = [r for r in results if r["exit_code"] != 0]
    if failed:
        rr.note("run_all", "test",
                f"{len(failed)} stage(s) failed: {', '.join(r['stage'] for r in failed)}. "
                f"The report covers only the stages that completed.")

    print(f"Consolidated report -> {rr.REPORT_MD} (and {rr.REPORT_CSV})")
    if failed:
        print(f"WARNING: {len(failed)} stage(s) failed: "
              f"{', '.join(r['stage'] for r in failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
