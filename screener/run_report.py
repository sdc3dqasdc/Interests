#!/usr/bin/env python3
"""One consolidated report for a whole test run, shared by every tool.

Every script in this pipeline (screener -> select_top15 -> backtest ->
win_rate_tester -> roi_tester -> alpha_lab) writes its parameters, headline
metrics and key tables into a SINGLE pair of files:

    run_report.csv   long-format, machine-readable, one fact per row
    run_report.md    the same content rendered for reading

WHY LONG FORMAT: the six tools emit wildly different shapes — the screener has
candidate counts, the backtest has an equity curve summary, alpha_lab has a
factor table.  A wide CSV would be mostly empty columns.  One fact per row
(run_id, tool, section, kind, name, value) holds all of them without either
losing information or inventing a schema per tool.

WIPE SEMANTICS ("wiped for each new test"):
  - Running any tool on its own starts a NEW test: the report is wiped first,
    so it only ever describes the run you just did.
  - run_all.py exports RUN_REPORT_SESSION so the whole pipeline counts as ONE
    test: the first tool wipes, every later tool appends to the same report.

So the report always covers exactly one test, whether that test is a single
script or the full pipeline.  Set RUN_REPORT_SESSION yourself to group any
other set of commands into one test.
"""

from __future__ import annotations

import csv
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

REPORT_CSV = Path("run_report.csv")
REPORT_MD = Path("run_report.md")

_FIELDS = ["run_id", "timestamp_utc", "tool", "section", "kind", "name", "value"]

# Env var run_all.py sets so a multi-script pipeline counts as a single test.
_SESSION_ENV = "RUN_REPORT_SESSION"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fmt(value: Any) -> str:
    """Render one value for the CSV, keeping floats readable."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        return f"{value:.6g}"
    if isinstance(value, Path):
        return str(value)
    return str(value)


# Cached so repeated appends in one process do not re-read the file.
_RUN_ID: str | None = None


def _current_run_id() -> str | None:
    """The run_id already in the report, or None when there is no report."""
    global _RUN_ID
    if _RUN_ID is not None:
        return _RUN_ID
    if not REPORT_CSV.exists():
        return None
    try:
        with REPORT_CSV.open(newline="") as handle:
            for row in csv.DictReader(handle):
                _RUN_ID = row.get("run_id")
                return _RUN_ID
    except Exception:
        return None
    return None


def reset(run_id: str) -> None:
    """Wipe the report and start a fresh one under run_id.

    Writes a seed meta row carrying the run_id: a header-only CSV has no row
    to read the id back from, so a second process could not tell whether it
    was joining this run or starting a new one."""
    global _RUN_ID
    try:
        with REPORT_CSV.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_FIELDS)
            writer.writeheader()
            writer.writerow({
                "run_id": run_id, "timestamp_utc": _now(), "tool": "report",
                "section": "run", "kind": "meta", "name": "run_id", "value": run_id,
            })
        _RUN_ID = run_id
        REPORT_MD.write_text(f"# Test run `{run_id}`\n\n_started {_now()}_\n")
    except Exception as exc:
        print(f"WARNING: could not reset {REPORT_CSV}: {exc}")


def _append(tool: str, section: str, kind: str, pairs: Iterable[tuple[str, Any]]) -> None:
    run_id = _current_run_id() or "unknown"
    try:
        with REPORT_CSV.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_FIELDS)
            stamp = _now()
            for name, value in pairs:
                writer.writerow({
                    "run_id": run_id, "timestamp_utc": stamp, "tool": tool,
                    "section": section, "kind": kind, "name": name, "value": _fmt(value),
                })
    except Exception as exc:
        print(f"WARNING: could not write {REPORT_CSV}: {exc}")
        return
    render_md()


def begin(tool: str, args: Any = None, *, skip: Iterable[str] = ()) -> None:
    """Open this tool's section, wiping the report unless we're in a session.

    Call once at the top of main().  Passing the argparse Namespace records
    every parameter, so the report is self-describing — you can tell from the
    report alone exactly which flags produced these numbers."""
    session = os.environ.get(_SESSION_ENV)
    want = session or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    if _current_run_id() != want:
        reset(want)

    _append(tool, "run", "meta", [("started_utc", _now())])
    if args is not None:
        hidden = {"alpaca_key", "alpaca_secret"} | set(skip)
        params = vars(args) if hasattr(args, "__dict__") else dict(args)
        _append(tool, "parameters", "param",
                sorted((k, v) for k, v in params.items() if k not in hidden))


def metrics(tool: str, section: str, mapping: Mapping[str, Any]) -> None:
    """Record headline numbers, e.g. {'win_rate_pct': 57.7, 'picks': 685}."""
    _append(tool, section, "metric", list(mapping.items()))


def rows(tool: str, section: str, records: Iterable[Mapping[str, Any]],
         limit: int | None = None) -> None:
    """Record a small table (picks, factor stats) as one fact per cell."""
    out: list[tuple[str, Any]] = []
    for i, rec in enumerate(records):
        if limit is not None and i >= limit:
            break
        for col, val in rec.items():
            out.append((f"{i}.{col}", val))
    if out:
        _append(tool, section, "row", out)


def note(tool: str, section: str, text: str) -> None:
    """Record a free-text caveat or warning alongside the numbers."""
    _append(tool, section, "note", [("note", text)])


# ---------------------------------------------------------------------------
# Markdown rendering — regenerated after every append so the .md stays current
# even if a later tool in the pipeline crashes.
# ---------------------------------------------------------------------------
def render_md() -> None:
    if not REPORT_CSV.exists():
        return
    try:
        with REPORT_CSV.open(newline="") as handle:
            data = list(csv.DictReader(handle))
    except Exception:
        return
    if not data:
        return

    run_id = data[0].get("run_id", "unknown")
    lines = [f"# Test run `{run_id}`", "",
             f"_{len(data)} facts recorded; rendered {_now()}_", ""]

    # Preserve first-seen order of tools and of sections within each tool.
    tools: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in data:
        tools.setdefault(row["tool"], {}).setdefault(row["section"], []).append(row)

    for tool, sections in tools.items():
        # Skip tools that contributed only bookkeeping (the run_id seed row).
        if all(e["kind"] == "meta" for entries in sections.values() for e in entries):
            continue
        lines += [f"## {tool}", ""]
        for section, entries in sections.items():
            kind = entries[0]["kind"]
            if kind == "meta":
                continue
            lines += [f"### {section}", ""]
            if kind == "row":
                lines += _render_table(entries)
            elif kind == "note":
                lines += [f"> {e['value']}" for e in entries] + [""]
            else:
                lines += ["| field | value |", "| --- | --- |"]
                lines += [f"| {e['name']} | {e['value']} |" for e in entries]
                lines += [""]
    try:
        REPORT_MD.write_text("\n".join(lines))
    except Exception:
        pass


def _render_table(entries: list[dict[str, str]]) -> list[str]:
    """Rebuild a table from its flattened '<row>.<col>' cells."""
    table: dict[int, dict[str, str]] = {}
    cols: list[str] = []
    for e in entries:
        idx, _, col = e["name"].partition(".")
        try:
            i = int(idx)
        except ValueError:
            continue
        table.setdefault(i, {})[col] = e["value"]
        if col not in cols:
            cols.append(col)
    if not cols:
        return [""]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for i in sorted(table):
        out.append("| " + " | ".join(table[i].get(c, "") for c in cols) + " |")
    return out + [""]
