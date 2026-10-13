"""Pure backtest logic: portfolio simulation, metrics, JSON helpers.

Nothing in this module touches the database or an LLM, so it can be tested on
its own.

Trading model
-------------
* Long-only, all-in: ``buy`` while flat invests all cash, ``sell`` while long
  liquidates the whole position. ``buy`` while long and ``sell`` while flat do
  nothing (``hold`` always does nothing). Iranian retail traders cannot short.
* Fractional shares are allowed.
* Execution timing (``execution``):
    - ``next_open``  : signal at the close of day D, filled at the open of the
                       next trading day (default, no look-ahead).
    - ``next_close`` : signal at the close of day D, filled at the close of
                       the next trading day.
    - ``same_close`` : signal and fill at the close of day D. Optimistic: the
                       agents see D's close and trade at it.
  A signal on the very last day of the range cannot be filled (there is no
  next day inside the range) and is reported as ``pending_order_at_end``.
"""

import datetime as dt
import math
import statistics
from collections import Counter
from decimal import Decimal
from typing import Any

EXECUTION_MODES = ("next_open", "next_close", "same_close")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_date(value: Any) -> dt.date:
    """Normalise date / datetime / 'YYYY-MM-DD...' strings to a date."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def to_jsonable(obj: Any) -> Any:
    """Recursively convert agent outputs to something JSON/JSONB accepts.

    Handles pydantic models, Decimal, dates, numpy/pandas scalars and
    NaN/inf (Postgres JSON rejects NaN, so they become null).
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return _finite(obj)
    if isinstance(obj, Decimal):
        return _finite(float(obj))
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):  # pydantic v2
        return to_jsonable(obj.model_dump(mode="json"))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if type(obj).__name__ in ("NAType", "NaTType"):  # pd.NA / pd.NaT
        return None
    if hasattr(obj, "item"):  # numpy scalar
        try:
            return to_jsonable(obj.item())
        except Exception:
            pass
    return str(obj)


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class Portfolio:
    """Single-instrument, long-only, all-in portfolio."""

    def __init__(
        self,
        initial_capital: float,
        fee_buy: float = 0.0,
        fee_sell: float = 0.0,
    ):
        self.initial_capital = float(initial_capital)
        self.fee_buy = float(fee_buy)
        self.fee_sell = float(fee_sell)

        self.cash = float(initial_capital)
        self.shares = 0.0

        self._cost_basis = 0.0
        self.entry_date: dt.date | None = None
        self.entry_price: float | None = None
        self.closed_trades: list[dict] = []

    @property
    def in_position(self) -> bool:
        return self.shares > 0

    def equity(self, price: float) -> float:
        return self.cash + self.shares * price

    def execute(
        self,
        action: str,
        price: float | None,
        date: dt.date,
    ) -> dict | None:
        """Execute buy/sell at ``price``. Returns a fill dict or None."""
        if price is None or not math.isfinite(price) or price <= 0:
            return None

        if action == "buy" and not self.in_position and self.cash > 0:
            spend = self.cash
            fee = spend - spend / (1 + self.fee_buy)
            self.shares = (spend - fee) / price
            self.cash = 0.0
            self._cost_basis = spend
            self.entry_date = date
            self.entry_price = price
            return {
                "action": "buy",
                "price": price,
                "shares": self.shares,
                "fee": fee,
            }

        if action == "sell" and self.in_position:
            shares = self.shares
            gross = shares * price
            fee = gross * self.fee_sell
            proceeds = gross - fee
            self.closed_trades.append({
                "entry_date": self.entry_date.isoformat(),
                "entry_price": self.entry_price,
                "exit_date": date.isoformat(),
                "exit_price": price,
                "return": proceeds / self._cost_basis - 1,
            })
            self.cash = proceeds
            self.shares = 0.0
            self._cost_basis = 0.0
            return {
                "action": "sell",
                "price": price,
                "shares": shares,
                "fee": fee,
            }

        return None


# ---------------------------------------------------------------------------
# Day-by-day simulator
# ---------------------------------------------------------------------------

class Simulator:
    """Drives a Portfolio one trading day at a time.

    Per day call ``begin_day`` (fills the order from the previous signal),
    obtain the agents' decision, then call ``end_day`` with the final action.
    """

    def __init__(
        self,
        *,
        initial_capital: float,
        execution: str = "next_open",
        fee_buy: float = 0.0,
        fee_sell: float = 0.0,
    ):
        if execution not in EXECUTION_MODES:
            raise ValueError(
                f"execution must be one of {EXECUTION_MODES}, got {execution!r}"
            )

        self.execution = execution
        self.portfolio = Portfolio(initial_capital, fee_buy, fee_sell)

        self.pending_action: str | None = None
        self._pending_signal_date: dt.date | None = None

        self._executed: dict | None = None
        self._position_before = "flat"

        self._prev_equity: float | None = None
        self._prev_close: float | None = None
        self._peak: float | None = None
        self._bh_base: float | None = None

    @property
    def unfilled_order(self) -> str | None:
        """Last-day signal that would still change the position, if any."""
        if self.pending_action == "buy" and not self.portfolio.in_position:
            return "buy"
        if self.pending_action == "sell" and self.portfolio.in_position:
            return "sell"
        return None

    def begin_day(self, day: dict) -> str:
        """Fill yesterday's order. Returns the position before today's signal."""
        executed = None

        if self.pending_action and self.execution != "same_close":
            open_price = day.get("open")
            if self.execution == "next_open" and open_price and open_price > 0:
                price = open_price
            else:
                # next_close, or the open is missing for this day
                price = day["close"]

            executed = self.portfolio.execute(
                self.pending_action,
                price,
                day["date"],
            )
            if executed:
                executed["signal_date"] = self._pending_signal_date

        self.pending_action = None
        self._pending_signal_date = None
        self._executed = executed
        self._position_before = "long" if self.portfolio.in_position else "flat"

        return self._position_before

    def end_day(self, day: dict, final_action: str | None) -> dict:
        """Register today's decision and mark the portfolio to market."""
        close = float(day["close"])

        if self._bh_base is None:
            self._bh_base = close

        if final_action in ("buy", "sell"):
            if self.execution == "same_close":
                executed = self.portfolio.execute(
                    final_action,
                    close,
                    day["date"],
                )
                if executed:
                    executed["signal_date"] = day["date"]
                    self._executed = executed
            else:
                self.pending_action = final_action
                self._pending_signal_date = day["date"]

        pf = self.portfolio
        equity = pf.equity(close)

        self._peak = equity if self._peak is None else max(self._peak, equity)
        drawdown = equity / self._peak - 1

        daily_return = (
            0.0
            if self._prev_equity is None
            else equity / self._prev_equity - 1
        )
        bh_daily_return = (
            0.0
            if self._prev_close is None
            else close / self._prev_close - 1
        )

        self._prev_equity = equity
        self._prev_close = close

        ex = self._executed or {}

        return {
            "position_before": self._position_before,
            "position_after": "long" if pf.in_position else "flat",
            "executed_action": ex.get("action"),
            "executed_price": ex.get("price"),
            "executed_shares": ex.get("shares"),
            "executed_fee": ex.get("fee"),
            "executed_signal_date": ex.get("signal_date"),
            "shares": pf.shares,
            "cash": pf.cash,
            "equity": equity,
            "daily_return": daily_return,
            "cumulative_return": equity / pf.initial_capital - 1,
            "drawdown": drawdown,
            "buy_hold_equity": pf.initial_capital * close / self._bh_base,
            "buy_hold_daily_return": bh_daily_return,
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _max_drawdown(equity: list[float]) -> float:
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1)
    return worst


def _sharpe(returns: list[float], periods_per_year: int) -> float | None:
    if len(returns) < 2:
        return None
    sd = statistics.stdev(returns)
    if sd == 0:
        return None
    return statistics.mean(returns) / sd * math.sqrt(periods_per_year)


def compute_metrics(
    rows: list[dict],
    closed_trades: list[dict],
    *,
    initial_capital: float,
    periods_per_year: int = 240,
    pending_order_at_end: str | None = None,
) -> dict:
    """Summary statistics.

    ``rows``: one dict per trading day, in date order, with keys
    equity, buy_hold_equity, daily_return, buy_hold_daily_return,
    position_after, status, proposed_action, final_action, risk_verdict.
    """
    if not rows:
        return {}

    equity = [r["equity"] for r in rows]
    bh_equity = [r["buy_hold_equity"] for r in rows]
    returns = [r["daily_return"] for r in rows[1:]]
    bh_returns = [r["buy_hold_daily_return"] for r in rows[1:]]

    wins = [t for t in closed_trades if t["return"] > 0]

    return {
        "trading_days": len(rows),
        "initial_capital": initial_capital,
        "final_equity": equity[-1],
        "total_return": equity[-1] / initial_capital - 1,
        "buy_hold_return": bh_equity[-1] / initial_capital - 1,
        "excess_return": (equity[-1] - bh_equity[-1]) / initial_capital,
        "max_drawdown": _max_drawdown(equity),
        "buy_hold_max_drawdown": _max_drawdown(bh_equity),
        "daily_volatility": (
            statistics.stdev(returns) if len(returns) > 1 else None
        ),
        "sharpe_annualized": _sharpe(returns, periods_per_year),
        "buy_hold_sharpe_annualized": _sharpe(bh_returns, periods_per_year),
        "periods_per_year": periods_per_year,
        "closed_trades": len(closed_trades),
        "winning_trades": len(wins),
        "win_rate": len(wins) / len(closed_trades) if closed_trades else None,
        "avg_trade_return": (
            statistics.mean(t["return"] for t in closed_trades)
            if closed_trades
            else None
        ),
        "exposure": (
            sum(1 for r in rows if r["position_after"] == "long") / len(rows)
        ),
        "open_position_at_end": rows[-1]["position_after"] == "long",
        "pending_order_at_end": pending_order_at_end,
        "error_days": sum(1 for r in rows if r["status"] != "completed"),
        "proposed_actions": dict(
            Counter(r["proposed_action"] or "none" for r in rows)
        ),
        "final_actions": dict(
            Counter(r["final_action"] or "none" for r in rows)
        ),
        "risk_vetoes": sum(1 for r in rows if r.get("risk_verdict") == "veto"),
        "trades": closed_trades,
    }
