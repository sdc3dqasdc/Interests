#!/usr/bin/env python3
"""Execute the pipeline's picks on an Alpaca PAPER account.

Reads top_candidates.csv (written by select_top15.py) and reconciles the paper
account against it with the same mechanics backtest.py measures:

  - at most --max-positions names, each sized 1/N of equity, scaled by the
    screener's sentiment-derived ``position_weight``
  - a hard --stop-loss-pct stop, placed as a real GTC stop order so it works
    even when this script is not running
  - a fixed --hold-days holding period, after which the position is sold

Nothing is sent to Alpaca unless you pass --live: the default is a dry run that
prints the exact orders it would place.  Even with --live, this only ever talks
to the paper endpoint (https://paper-api.alpaca.markets) unless you deliberately
pass --real, which requires typing a confirmation.

Entry dates are tracked locally in paper_positions.json, since Alpaca's position
objects do not carry one.  Delete that file to forget the hold clock.

Usage:
    .venv/bin/python paper_trader.py                 # dry run, show the plan
    .venv/bin/python paper_trader.py --live          # submit to the paper account
    .venv/bin/python paper_trader.py --live --hold-days 50 --max-positions 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt  # reuse _load_dotenv so .env works exactly as elsewhere

LEDGER_PATH = Path("paper_positions.json")
PAPER_URL = "https://paper-api.alpaca.markets"


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
def get_trading_client(api_key: str | None, secret_key: str | None, paper: bool = True):
    """Trading client for the paper (or, if you insist, live) endpoint.

    The same ALPACA_API_KEY / ALPACA_SECRET_KEY the rest of the pipeline uses —
    but note that paper and live accounts have SEPARATE keys.  Keys generated on
    the paper dashboard only work with paper=True."""
    from alpaca.trading.client import TradingClient
    import os

    bt._load_dotenv()
    key = api_key or os.environ.get("ALPACA_API_KEY")
    secret = secret_key or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit(
            "Alpaca API key and secret required. Set ALPACA_API_KEY / ALPACA_SECRET_KEY "
            "(env vars or a .env file), or pass --alpaca-key / --alpaca-secret."
        )
    return TradingClient(key, secret, paper=paper)


def latest_prices(symbols: list[str], api_key: str | None, secret_key: str | None) -> dict[str, float]:
    """Last IEX trade per symbol; falls back to {} so callers can use the CSV price."""
    if not symbols:
        return {}
    from alpaca.data.requests import StockLatestTradeRequest
    from alpaca.data.enums import DataFeed

    client = bt._get_alpaca_client(api_key, secret_key)
    try:
        trades = client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=symbols, feed=DataFeed.IEX)
        )
    except Exception as exc:  # data outage should not block the reconcile
        print(f"  ! latest-trade lookup failed ({exc}); falling back to CSV prices")
        return {}
    return {sym: float(t.price) for sym, t in trades.items() if t and t.price}


# ---------------------------------------------------------------------------
# Local entry-date ledger
# ---------------------------------------------------------------------------
def load_ledger() -> dict:
    try:
        return json.loads(LEDGER_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_ledger(ledger: dict) -> None:
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2, sort_keys=True))


def days_held(entry_iso: str | None) -> int | None:
    """Trading days since entry, or None when the entry date is unknown."""
    if not entry_iso:
        return None
    try:
        entry = date.fromisoformat(entry_iso[:10])
    except ValueError:
        return None
    return int(np.busday_count(entry, date.today()))


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
def build_plan(picks: pd.DataFrame, positions: list, equity: float, prices: dict[str, float],
               args: argparse.Namespace, ledger: dict) -> tuple[list[dict], list[dict]]:
    """Return (sells, buys). Sells run first so their cash funds the buys."""
    held = {p.symbol: p for p in positions}
    wanted = list(picks["ticker"])

    sells: list[dict] = []
    for symbol, pos in held.items():
        entry = ledger.get(symbol, {}).get("entry_date")
        age = days_held(entry)
        if age is not None and age >= args.hold_days:
            reason = f"held {age} trading days >= --hold-days {args.hold_days}"
        elif args.exit_on_drop and symbol not in wanted:
            reason = "no longer in top_candidates.csv"
        else:
            continue
        sells.append({"symbol": symbol, "qty": abs(float(pos.qty)), "reason": reason})

    sold = {s["symbol"] for s in sells}
    room = args.max_positions - (len(held) - len(sold))
    if room <= 0:
        return sells, []

    per_name = equity / args.max_positions

    buys: list[dict] = []
    for _, row in picks.iterrows():
        if len(buys) >= room:
            break
        symbol = row["ticker"]
        if symbol in held and symbol not in sold:
            continue  # already own it; do not average in
        weight = float(row.get("position_weight") or 1.0)
        price = prices.get(symbol) or float(row["price"])
        if price <= 0:
            continue
        qty = int((per_name * weight) // price)
        if qty < 1:
            print(f"  ! {symbol}: 1/{args.max_positions} of equity (${per_name * weight:,.0f}) "
                  f"is below one share at ${price:,.2f} — skipped")
            continue
        buys.append({
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "notional": qty * price,
            "weight": weight,
            "stop": round(price * (1 - args.stop_loss_pct / 100), 2) if args.stop_loss_pct > 0 else None,
        })
    return sells, buys


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def cancel_open_orders(client, symbol: str) -> None:
    """Cancel resting orders (e.g. the protective stop) so a sell is not blocked
    by shares already reserved."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
    for order in orders:
        client.cancel_order_by_id(order.id)
        print(f"    cancelled resting {order.side.value} order {order.id}")


def submit_sell(client, symbol: str, qty: float):
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    return client.submit_order(MarketOrderRequest(
        symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY))


def submit_buy(client, symbol: str, qty: int):
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    return client.submit_order(MarketOrderRequest(
        symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))


def submit_stop(client, symbol: str, qty: float, stop_price: float):
    from alpaca.trading.requests import StopOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    return client.submit_order(StopOrderRequest(
        symbol=symbol, qty=qty, side=OrderSide.SELL,
        stop_price=stop_price, time_in_force=TimeInForce.GTC))


def ensure_stops(client, args, ledger: dict, live: bool) -> None:
    """Every open position should carry a GTC stop. Repairs anything missing —
    e.g. a stop that could not be placed because the buy filled after hours."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    if args.stop_loss_pct <= 0:
        return
    open_orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    protected = {o.symbol for o in open_orders if o.side.value == "sell"}
    for pos in client.get_all_positions():
        if pos.symbol in protected:
            continue
        basis = float(ledger.get(pos.symbol, {}).get("entry_price") or pos.avg_entry_price)
        stop_price = round(basis * (1 - args.stop_loss_pct / 100), 2)
        qty = abs(float(pos.qty))
        print(f"  STOP  {pos.symbol:<6} {qty:g} sh @ ${stop_price:,.2f} "
              f"(-{args.stop_loss_pct:g}% from ${basis:,.2f})")
        if live:
            try:
                submit_stop(client, pos.symbol, qty, stop_price)
            except Exception as exc:
                print(f"    ! stop rejected for {pos.symbol}: {exc}")


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("top_candidates.csv"),
                        help="Ranked picks from select_top15.py (default: top_candidates.csv)")
    parser.add_argument("--live", action="store_true",
                        help="Actually submit the orders. Without this, prints the plan only.")
    parser.add_argument("--real", action="store_true",
                        help="Use the LIVE money endpoint instead of paper. Requires typing a "
                             "confirmation. Don't.")
    parser.add_argument("--max-positions", type=int, default=5,
                        help="Concurrent positions; each sized 1/N of equity (default: 5)")
    parser.add_argument("--hold-days", type=int, default=50,
                        help="Trading days to hold before selling (default: 50, matches backtest)")
    parser.add_argument("--stop-loss-pct", type=float, default=15.0,
                        help="GTC stop below entry, %% (0 disables; default: 15)")
    parser.add_argument("--exit-on-drop", action="store_true",
                        help="Also sell any holding that has fallen out of top_candidates.csv "
                             "(the backtest does NOT do this — it holds the full period)")
    parser.add_argument("--allow-closed", action="store_true",
                        help="Submit even when the market is closed (orders queue for the next "
                             "session and fill at an unknown open price)")
    parser.add_argument("--alpaca-key", type=str, default=None)
    parser.add_argument("--alpaca-secret", type=str, default=None)
    parser.add_argument("--cash-override", type=float, default=None,
                        help="Size positions off this equity figure instead of the account's")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.real:
        print("You asked to trade REAL MONEY. This script is written for paper trading.")
        if input('Type "I understand" to continue: ').strip() != "I understand":
            print("Aborted.")
            return 1

    if not args.input.exists():
        raise SystemExit(f"{args.input} not found — run select_top15.py first.")
    picks = pd.read_csv(args.input)
    if "ticker" not in picks.columns:
        raise SystemExit(f"{args.input} has no 'ticker' column.")

    client = get_trading_client(args.alpaca_key, args.alpaca_secret, paper=not args.real)
    account = client.get_account()
    clock = client.get_clock()
    equity = float(args.cash_override if args.cash_override is not None else account.equity)

    mode = "LIVE MONEY" if args.real else "PAPER"
    print(f"{mode} account {account.account_number}  equity ${float(account.equity):,.2f}  "
          f"cash ${float(account.cash):,.2f}")
    print(f"Market is {'OPEN' if clock.is_open else 'CLOSED'} "
          f"(next open {clock.next_open:%Y-%m-%d %H:%M %Z})")
    if account.trading_blocked:
        raise SystemExit("Account is flagged trading_blocked — nothing to do.")
    if not clock.is_open and args.live and not args.allow_closed:
        raise SystemExit("Market closed. Re-run during regular hours, or pass --allow-closed "
                         "to queue the orders for the next session.")

    positions = client.get_all_positions()
    ledger = load_ledger()
    prices = latest_prices(list(picks["ticker"]), args.alpaca_key, args.alpaca_secret)

    print(f"\nHolding {len(positions)} position(s):")
    for pos in positions:
        age = days_held(ledger.get(pos.symbol, {}).get("entry_date"))
        age_str = f"{age}d held" if age is not None else "entry date unknown"
        print(f"  {pos.symbol:<6} {float(pos.qty):g} sh  "
              f"P/L {float(pos.unrealized_plpc) * 100:+.1f}%  ({age_str})")

    sells, buys = build_plan(picks, positions, equity, prices, args, ledger)

    print(f"\nPlan ({'SUBMITTING' if args.live else 'dry run — nothing will be sent'}):")
    if not sells and not buys:
        print("  nothing to do")
    for s in sells:
        print(f"  SELL  {s['symbol']:<6} {s['qty']:g} sh   ({s['reason']})")
    for b in buys:
        stop = f", stop ${b['stop']:,.2f}" if b["stop"] else ""
        print(f"  BUY   {b['symbol']:<6} {b['qty']:>4} sh @ ~${b['price']:,.2f} "
              f"= ${b['notional']:,.0f} (weight {b['weight']:g}{stop})")

    if not args.live:
        print("\nDry run complete. Re-run with --live to submit these to the paper account.")
        return 0

    for s in sells:
        cancel_open_orders(client, s["symbol"])
        try:
            order = submit_sell(client, s["symbol"], s["qty"])
            print(f"  sold {s['symbol']} — order {order.id}")
            ledger.pop(s["symbol"], None)
        except Exception as exc:
            print(f"  ! sell rejected for {s['symbol']}: {exc}")

    for b in buys:
        try:
            order = submit_buy(client, b["symbol"], b["qty"])
            print(f"  bought {b['symbol']} — order {order.id}")
            ledger[b["symbol"]] = {
                "entry_date": date.today().isoformat(),
                "entry_price": b["price"],
                "qty": b["qty"],
                "submitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        except Exception as exc:
            print(f"  ! buy rejected for {b['symbol']}: {exc}")

    save_ledger(ledger)

    # Protective stops, for the new fills and any older position missing one.
    print("\nProtective stops:")
    ensure_stops(client, args, ledger, live=True)

    print(f"\nDone. Ledger written to {LEDGER_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
