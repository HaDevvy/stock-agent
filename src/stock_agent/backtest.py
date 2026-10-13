"""Backtest the agent graph over a date range and store everything in the DB.

For every trading day D in [start, end] the whole graph (technical, flow,
codal, industry -> decision -> risk) runs with ``data_until = D``. The risk-
checked ``final_decision`` is traded by a simple long-only simulator (see
backtest_sim.py for the exact trading rules).

Stored in the database
----------------------
backtests                  id (UUID), name, symbol, dates, settings, graph
                           (yaml + mermaid + PNG), equity chart PNG, metrics,
                           status, created_at / finished_at
backtest_days              one row per trading day: prices, proposed / risk /
                           final decision, fills, equity, drawdown, ...
backtest_agent_responses   the response of EVERY agent on EVERY day (plus the
                           evidence it received), including failures

Usage (from the project root, where .env lives)
-----------------------------------------------
    export PYTHONPATH=src

    python -m stock_agent.backtest run --symbol خودرو --start 2026-06-01 --end 2026-06-10

    python -m stock_agent.backtest list
    python -m stock_agent.backtest show   <backtest-id>
    python -m stock_agent.backtest export <backtest-id> --out out/
    python -m stock_agent.backtest run --resume <backtest-id>
"""

import argparse
import csv
import datetime as dt
import hashlib
import sys
import threading
import time
import traceback
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from .backtest_charts import render_equity_chart, render_graph_png
from .backtest_models import (
    Backtest,
    BacktestAgentResponse,
    BacktestDay,
    Base,
    utcnow,
)
from .backtest_sim import (
    EXECUTION_MODES,
    Simulator,
    compute_metrics,
    to_date,
    to_jsonable,
)
from .config import settings
from .database import engine, get_trading_days
from .graph import build_graph
from .graph_loader import DEFAULT_GRAPH_PATH, load_graph_config
from .state import AgentState


def init_db() -> None:
    """Create the backtest tables if they do not exist yet."""
    Base.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Recording every agent response
# ---------------------------------------------------------------------------

def _attr(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _summarize(output: dict | None) -> tuple[Any, Any, Any]:
    """(label, score, confidence) for quick querying."""
    if not output:
        return None, None, None

    analyses = output.get("analyses")
    if analyses:
        item = analyses[0]
        return (
            _attr(item, "signal"),
            _attr(item, "score"),
            _attr(item, "confidence"),
        )

    review = output.get("risk_review")
    if review is not None:
        return _attr(review, "verdict"), None, _attr(review, "confidence")

    decision = output.get("decision")
    if decision is not None:
        return _attr(decision, "action"), None, _attr(decision, "confidence")

    return None, None, None


class AgentRecorder:
    """Wraps every graph node and keeps its output (or error).

    Nodes of the same graph run in parallel threads, hence the lock. The graph
    is invoked for one day at a time, so ``reset()`` between days is safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[dict] = []

    def reset(self) -> None:
        with self._lock:
            self._records = []

    def records(self) -> list[dict]:
        with self._lock:
            return sorted(self._records, key=lambda r: r["started_at"])

    def wrap(self, node_name: str, handler: Callable) -> Callable:
        @wraps(handler)
        def wrapped(state: AgentState):
            started_at = utcnow()
            clock = time.perf_counter()

            try:
                output = handler(state)
            except Exception:
                self._add(
                    node_name, started_at, clock, None, traceback.format_exc()
                )
                raise

            self._add(node_name, started_at, clock, output, None)
            return output

        return wrapped

    def _add(
        self,
        node_name: str,
        started_at: dt.datetime,
        clock: float,
        output: dict | None,
        error: str | None,
    ) -> None:
        label = score = confidence = None
        inputs = None
        stored_output = None

        if output is not None:
            label, score, confidence = _summarize(output)

            evidence = output.get("agent_inputs")
            if evidence:
                inputs = to_jsonable(_attr(evidence[0], "inputs"))

            stored_output = to_jsonable(
                {k: v for k, v in output.items() if k != "agent_inputs"}
            )

        record = {
            "agent": node_name,
            "status": "error" if error else "ok",
            "error": error,
            "label": label,
            "score": score,
            "confidence": confidence,
            "output": stored_output,
            "inputs": inputs,
            "started_at": started_at,
            "duration_ms": int((time.perf_counter() - clock) * 1000),
        }

        with self._lock:
            self._records.append(record)


# ---------------------------------------------------------------------------
# One day of agents
# ---------------------------------------------------------------------------

def _run_agents(
    graph,
    recorder: AgentRecorder,
    symbol: str,
    day: dt.date,
    portfolio_state: dict | None,
) -> tuple[dict | None, list[dict], str | None]:
    recorder.reset()

    state: dict[str, Any] = {
        "symbol": symbol,
        "decision_date": day.isoformat(),
        "data_until": day.isoformat(),
    }
    if portfolio_state is not None:
        state["portfolio"] = portfolio_state

    result = None
    error = None

    try:
        result = graph.invoke(state, config={"max_concurrency": 1})
        if not (result.get("final_decision") or result.get("decision")):
            error = "Graph finished without producing a decision."
    except Exception:
        error = traceback.format_exc()

    return result, recorder.records(), error


# ---------------------------------------------------------------------------
# Backtest run
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _apply_decision(row: BacktestDay, result: dict) -> None:
    proposed = result.get("decision")
    review = result.get("risk_review")
    final = result.get("final_decision") or proposed

    row.proposed_action = _attr(proposed, "action")
    row.proposed_confidence = _attr(proposed, "confidence")
    row.proposed_reason = _attr(proposed, "reason")

    row.risk_verdict = _attr(review, "verdict")
    row.risk_level = _attr(review, "risk_level")
    row.risk_confidence = _attr(review, "confidence")
    row.risk_reason = _attr(review, "reason")

    row.final_action = _attr(final, "action")
    row.final_confidence = _attr(final, "confidence")
    row.final_reason = _attr(final, "reason")


def _clear_decision(row: BacktestDay) -> None:
    for column in (
        "proposed_action", "proposed_confidence", "proposed_reason",
        "risk_verdict", "risk_level", "risk_confidence", "risk_reason",
        "final_action", "final_confidence", "final_reason",
    ):
        setattr(row, column, None)


def run_backtest(
    *,
    symbol: str | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    name: str | None = None,
    execution: str = "next_open",
    initial_capital: float = 100_000_000,
    fee_buy: float = 0.0,
    fee_sell: float = 0.0,
    with_portfolio: bool = False,
    periods_per_year: int = 240,
    graph_path: str | Path | None = None,
    resume_id: uuid.UUID | None = None,
    force: bool = False,
) -> uuid.UUID:
    init_db()

    graph_file = Path(graph_path) if graph_path else DEFAULT_GRAPH_PATH
    graph_yaml = graph_file.read_text(encoding="utf-8")
    graph_sha = _sha256(graph_yaml)

    recorder = AgentRecorder()
    graph = build_graph(graph_file, handler_wrapper=recorder.wrap)

    with Session(engine, expire_on_commit=False) as session:
        # -------------------------------------------------- create / resume
        if resume_id is not None:
            bt = session.get(Backtest, resume_id)
            if bt is None:
                raise SystemExit(f"Backtest {resume_id} not found.")

            cfg = bt.settings
            if cfg.get("graph_sha256") != graph_sha and not force:
                raise SystemExit(
                    "The graph file differs from the one this backtest was "
                    "started with. Use --graph with the original file, or "
                    "--force to continue anyway."
                )

            symbol = bt.symbol
            start, end = bt.start_date, bt.end_date
            execution = cfg["execution"]
            initial_capital = cfg["initial_capital"]
            fee_buy, fee_sell = cfg["fee_buy"], cfg["fee_sell"]
            with_portfolio = cfg["with_portfolio"]
            periods_per_year = cfg["periods_per_year"]

            bt.status = "running"
            bt.error = None
            bt.finished_at = None
        else:
            if not (symbol and start and end):
                raise SystemExit("--symbol, --start and --end are required.")
            if execution not in EXECUTION_MODES:
                raise SystemExit(f"--execution must be one of {EXECUTION_MODES}")
            if start > end:
                raise SystemExit("--start must not be after --end.")

            bt = None

        price_rows = get_trading_days(
            symbol, start.isoformat(), end.isoformat()
        )
        days = [
            {
                "date": to_date(r["date"]),
                "open": float(r["open"]) if r["open"] is not None else None,
                "high": float(r["high"]) if r["high"] is not None else None,
                "low": float(r["low"]) if r["low"] is not None else None,
                "close": float(r["close"]),
                "volume": float(r["volume"]) if r["volume"] is not None else None,
            }
            for r in price_rows
        ]
        if not days:
            raise SystemExit(
                f"No trading days found for {symbol} between {start} and {end}."
            )

        if bt is None:
            config = load_graph_config(graph_file)
            bt = Backtest(
                name=name or f"{symbol} {start}..{end}",
                symbol=symbol,
                start_date=start,
                end_date=end,
                status="running",
                settings={
                    "symbol": symbol,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "execution": execution,
                    "initial_capital": initial_capital,
                    "fee_buy": fee_buy,
                    "fee_sell": fee_sell,
                    "with_portfolio": with_portfolio,
                    "periods_per_year": periods_per_year,
                    "graph_path": str(graph_file),
                    "graph_sha256": graph_sha,
                    "model_name": settings.model_name,
                    "technical_lookback_days": settings.technical_lookback_days,
                    "industry_lookback_days": settings.industry_lookback_days,
                    "trading_days": len(days),
                },
                graph_yaml=graph_yaml,
                graph_mermaid=graph.get_graph().draw_mermaid(),
                graph_png=render_graph_png(config),
            )
            session.add(bt)

        session.commit()
        bt_id = bt.id

        print(f"Backtest id: {bt_id}")
        print(
            f"{symbol}: {len(days)} trading days, "
            f"{days[0]['date']} .. {days[-1]['date']}, execution={execution}"
        )

        # ---------------------------------------------------------- the loop
        existing = {
            row.trade_date: row
            for row in session.scalars(
                select(BacktestDay).where(BacktestDay.backtest_id == bt_id)
            )
        }

        sim = Simulator(
            initial_capital=initial_capital,
            execution=execution,
            fee_buy=fee_buy,
            fee_sell=fee_sell,
        )
        series: list[dict] = []

        try:
            for index, day in enumerate(days, start=1):
                d = day["date"]

                sim.begin_day(day)

                row = existing.get(d)
                reuse = row is not None and row.status == "completed"

                if reuse:
                    source = "cached"
                else:
                    pf = sim.portfolio
                    portfolio_state = (
                        {
                            "position": "long" if pf.in_position else "flat",
                            "shares": pf.shares,
                            "cash": pf.cash,
                            "entry_price": pf.entry_price,
                            "entry_date": (
                                pf.entry_date.isoformat()
                                if pf.entry_date else None
                            ),
                        }
                        if with_portfolio
                        else None
                    )

                    result, records, error = _run_agents(
                        graph, recorder, symbol, d, portfolio_state
                    )

                    if row is None:
                        row = BacktestDay(
                            backtest_id=bt_id, trade_date=d, attempts=0
                        )
                        session.add(row)
                        existing[d] = row

                    row.attempts += 1
                    row.open_price = day["open"]
                    row.high_price = day["high"]
                    row.low_price = day["low"]
                    row.close_price = day["close"]
                    row.volume = day["volume"]

                    if error:
                        row.status = "error"
                        row.error = error
                        _clear_decision(row)
                    else:
                        row.status = "completed"
                        row.error = None
                        _apply_decision(row, result)

                    session.flush()  # row.id is needed for the responses

                    for rec in records:
                        session.add(
                            BacktestAgentResponse(
                                backtest_id=bt_id,
                                day_id=row.id,
                                trade_date=d,
                                attempt=row.attempts,
                                **rec,
                            )
                        )
                    source = "agents" if not error else "ERROR"

                # a failed day counts as "hold"
                final_action = (
                    row.final_action if row.status == "completed" else None
                )

                snap = sim.end_day(day, final_action)
                for key, value in snap.items():
                    setattr(row, key, value)

                session.commit()

                series.append({
                    **snap,
                    "trade_date": d,
                    "close": day["close"],
                    "status": row.status,
                    "proposed_action": row.proposed_action,
                    "final_action": row.final_action,
                    "risk_verdict": row.risk_verdict,
                })

                print(
                    f"[{index}/{len(days)}] {d} close={day['close']:.2f} "
                    f"proposed={row.proposed_action} final={row.final_action} "
                    f"fill={snap['executed_action'] or '-'} "
                    f"equity={snap['equity']:,.0f} ({source})",
                    flush=True,
                )

        except KeyboardInterrupt:
            session.rollback()
            bt = session.get(Backtest, bt_id)
            bt.status = "interrupted"
            session.commit()
            print(f"\nInterrupted. Continue with: run --resume {bt_id}")
            raise
        except Exception:
            session.rollback()
            bt = session.get(Backtest, bt_id)
            bt.status = "failed"
            bt.error = traceback.format_exc()
            session.commit()
            raise

        # ------------------------------------------------------- finish up
        metrics = compute_metrics(
            series,
            sim.portfolio.closed_trades,
            initial_capital=initial_capital,
            periods_per_year=periods_per_year,
            pending_order_at_end=sim.unfilled_order,
        )

        bt.metrics = to_jsonable(metrics)
        bt.equity_chart_png = render_equity_chart(
            series,
            title=(
                f"Backtest {str(bt_id)[:8]} | {start} to {end} | "
                f"{execution}"
            ),
        )
        bt.status = "completed"
        bt.finished_at = utcnow()
        session.commit()

    _print_metrics(metrics)

    if metrics["error_days"]:
        print(
            f"\n{metrics['error_days']} day(s) failed and were treated as "
            f"'hold'. Retry them with: run --resume {bt_id}"
        )

    return bt_id


# ---------------------------------------------------------------------------
# Reporting commands
# ---------------------------------------------------------------------------

def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _print_metrics(m: dict) -> None:
    if not m:
        print("No metrics.")
        return

    print("\n=== Result ===")
    print(f"Final equity        : {m['final_equity']:,.0f}")
    print(f"Strategy return     : {_pct(m['total_return'])}")
    print(f"Buy & hold return   : {_pct(m['buy_hold_return'])}")
    print(f"Max drawdown        : {_pct(m['max_drawdown'])} "
          f"(buy & hold {_pct(m['buy_hold_max_drawdown'])})")
    print(f"Sharpe (annualized) : {_num(m['sharpe_annualized'])} "
          f"(buy & hold {_num(m['buy_hold_sharpe_annualized'])})")
    print(f"Closed trades       : {m['closed_trades']} "
          f"(win rate {_pct(m['win_rate'])})")
    print(f"Time in market      : {_pct(m['exposure'])}")
    print(f"Proposed decisions  : {m['proposed_actions']}")
    print(f"Final decisions     : {m['final_actions']} "
          f"(risk vetoes: {m['risk_vetoes']})")
    if m["open_position_at_end"]:
        print("Position still open at the end (marked to the last close).")
    if m["pending_order_at_end"]:
        print(f"Unfilled '{m['pending_order_at_end']}' signal on the last day.")


def cmd_list(limit: int) -> None:
    init_db()
    with Session(engine) as session:
        rows = session.scalars(
            select(Backtest).order_by(Backtest.created_at.desc()).limit(limit)
        ).all()

        for bt in rows:
            m = bt.metrics or {}
            print(
                f"{bt.id}  {bt.status:<11} {bt.symbol:<10} "
                f"{bt.start_date}..{bt.end_date}  "
                f"return={_pct(m.get('total_return'))}  "
                f"b&h={_pct(m.get('buy_hold_return'))}  "
                f"created={bt.created_at:%Y-%m-%d %H:%M}"
            )
        if not rows:
            print("No backtests yet.")


def cmd_show(backtest_id: uuid.UUID) -> None:
    init_db()
    with Session(engine) as session:
        bt = session.get(Backtest, backtest_id)
        if bt is None:
            raise SystemExit(f"Backtest {backtest_id} not found.")

        print(f"id       : {bt.id}")
        print(f"name     : {bt.name}")
        print(f"status   : {bt.status}")
        print(f"created  : {bt.created_at}")
        print(f"settings : {bt.settings}")
        _print_metrics(bt.metrics or {})

        print("\n=== Days ===")
        print(f"{'date':<11}{'close':>10} {'proposed':<9}{'risk':<8}"
              f"{'final':<7}{'fill':<6}{'position':<9}{'equity':>16}")
        for day in bt.days:
            print(
                f"{day.trade_date!s:<11}{(day.close_price or 0):>10.2f} "
                f"{day.proposed_action or '-':<9}{day.risk_verdict or '-':<8}"
                f"{day.final_action or '-':<7}{day.executed_action or '-':<6}"
                f"{day.position_after or '-':<9}{(day.equity or 0):>16,.0f}"
                + ("  ERROR" if day.status == "error" else "")
            )


def cmd_export(backtest_id: uuid.UUID, out: Path) -> None:
    init_db()
    out.mkdir(parents=True, exist_ok=True)

    with Session(engine) as session:
        bt = session.get(Backtest, backtest_id)
        if bt is None:
            raise SystemExit(f"Backtest {backtest_id} not found.")

        (out / "graph.yaml").write_text(bt.graph_yaml, encoding="utf-8")
        (out / "graph.mmd").write_text(bt.graph_mermaid, encoding="utf-8")
        if bt.graph_png:
            (out / "graph.png").write_bytes(bt.graph_png)
        if bt.equity_chart_png:
            (out / "equity.png").write_bytes(bt.equity_chart_png)

        columns = [
            "trade_date", "status", "open_price", "close_price", "volume",
            "proposed_action", "risk_verdict", "final_action",
            "executed_action", "executed_price", "position_after",
            "shares", "cash", "equity", "daily_return", "drawdown",
            "buy_hold_equity",
        ]
        with (out / "days.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for day in bt.days:
                writer.writerow([getattr(day, c) for c in columns])

    print(f"Exported to {out}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m stock_agent.backtest")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run (or resume) a backtest")
    run.add_argument("--symbol")
    run.add_argument("--start", type=_date, help="YYYY-MM-DD")
    run.add_argument("--end", type=_date, help="YYYY-MM-DD")
    run.add_argument("--name")
    run.add_argument(
        "--execution", choices=EXECUTION_MODES, default="next_open",
        help="when signals are filled (default: next_open; same_close is "
             "optimistic)",
    )
    run.add_argument("--initial-capital", type=float, default=100_000_000)
    run.add_argument("--fee-buy", type=float, default=0.0,
                     help="fraction of the buy value, e.g. 0.004")
    run.add_argument("--fee-sell", type=float, default=0.0,
                     help="fraction of the sell value, e.g. 0.009")
    run.add_argument("--with-portfolio", action="store_true",
                     help="give the risk agent the simulated position "
                          "(main.py does not pass one)")
    run.add_argument("--periods-per-year", type=int, default=240,
                     help="trading days per year used for the Sharpe ratio")
    run.add_argument("--graph", help="path to a graph.yaml (default: project's)")
    run.add_argument("--resume", type=uuid.UUID, metavar="BACKTEST_ID",
                     help="continue an interrupted/failed backtest; completed "
                          "days are reused, failed days are retried")
    run.add_argument("--force", action="store_true",
                     help="resume even if graph.yaml changed")

    ls = sub.add_parser("list", help="list stored backtests")
    ls.add_argument("--limit", type=int, default=20)

    show = sub.add_parser("show", help="print a stored backtest")
    show.add_argument("id", type=uuid.UUID)

    export = sub.add_parser("export", help="write graph/chart/CSV to a folder")
    export.add_argument("id", type=uuid.UUID)
    export.add_argument("--out", type=Path, default=Path("backtest_export"))

    args = parser.parse_args(argv)

    if args.command == "run":
        run_backtest(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            name=args.name,
            execution=args.execution,
            initial_capital=args.initial_capital,
            fee_buy=args.fee_buy,
            fee_sell=args.fee_sell,
            with_portfolio=args.with_portfolio,
            periods_per_year=args.periods_per_year,
            graph_path=args.graph,
            resume_id=args.resume,
            force=args.force,
        )
    elif args.command == "list":
        cmd_list(args.limit)
    elif args.command == "show":
        cmd_show(args.id)
    elif args.command == "export":
        cmd_export(args.id, args.out)


if __name__ == "__main__":
    sys.exit(main())
