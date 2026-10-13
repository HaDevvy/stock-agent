"""Database schema for backtests.

backtests                  one row per backtest run (id, settings, graph, metrics)
backtest_days              one row per simulated trading day
backtest_agent_responses   one row per agent/node execution per day

Tables are created with ``Base.metadata.create_all`` (see ``init_db`` in
backtest.py). Existing market-data tables are never touched.
"""

import datetime as dt
import uuid
from typing import Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


# JSONB on PostgreSQL, plain JSON elsewhere.
JSONType = JSON().with_variant(JSONB(), "postgresql")

# BIGSERIAL on PostgreSQL; SQLite only auto-increments INTEGER primary keys.
BigIntPK = BigInteger().with_variant(Integer(), "sqlite")


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class Backtest(Base):
    __tablename__ = "backtests"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )

    name: Mapped[Optional[str]] = mapped_column(String(255))
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    start_date: Mapped[dt.date] = mapped_column(Date)
    end_date: Mapped[dt.date] = mapped_column(Date)

    # running | completed | failed | interrupted
    status: Mapped[str] = mapped_column(
        String(20), default="running", index=True
    )

    # Everything needed to reproduce the run (never contains API keys).
    settings: Mapped[dict] = mapped_column(JSONType)

    # The agent graph used by this run.
    graph_yaml: Mapped[str] = mapped_column(Text)
    graph_mermaid: Mapped[str] = mapped_column(Text)
    graph_png: Mapped[Optional[bytes]] = mapped_column(
        LargeBinary, deferred=True
    )

    # Equity curve / trades chart, rendered when the run finishes.
    equity_chart_png: Mapped[Optional[bytes]] = mapped_column(
        LargeBinary, deferred=True
    )

    metrics: Mapped[Optional[dict]] = mapped_column(JSONType)
    error: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    finished_at: Mapped[Optional[dt.datetime]] = mapped_column(
        DateTime(timezone=True)
    )

    days: Mapped[list["BacktestDay"]] = relationship(
        back_populates="backtest",
        cascade="all, delete-orphan",
        order_by="BacktestDay.trade_date",
    )


class BacktestDay(Base):
    __tablename__ = "backtest_days" 
    __table_args__ = (
        UniqueConstraint("backtest_id", "trade_date", name="uq_backtest_day"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)

    backtest_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("backtests.id", ondelete="CASCADE"),
        index=True,
    )
    trade_date: Mapped[dt.date] = mapped_column(Date)

    # completed | error  (error = the agent graph failed; treated as hold)
    status: Mapped[str] = mapped_column(String(20), default="completed")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[Optional[str]] = mapped_column(Text)

    # Market data for the day.
    open_price: Mapped[Optional[float]] = mapped_column(Float)
    high_price: Mapped[Optional[float]] = mapped_column(Float)
    low_price: Mapped[Optional[float]] = mapped_column(Float)
    close_price: Mapped[Optional[float]] = mapped_column(Float)
    volume: Mapped[Optional[float]] = mapped_column(Float)

    # Decision pipeline summary (full detail is in backtest_agent_responses).
    proposed_action: Mapped[Optional[str]] = mapped_column(String(10))
    proposed_confidence: Mapped[Optional[float]] = mapped_column(Float)
    proposed_reason: Mapped[Optional[str]] = mapped_column(Text)

    risk_verdict: Mapped[Optional[str]] = mapped_column(String(10))
    risk_level: Mapped[Optional[str]] = mapped_column(String(10))
    risk_confidence: Mapped[Optional[float]] = mapped_column(Float)
    risk_reason: Mapped[Optional[str]] = mapped_column(Text)

    final_action: Mapped[Optional[str]] = mapped_column(String(10))
    final_confidence: Mapped[Optional[float]] = mapped_column(Float)
    final_reason: Mapped[Optional[str]] = mapped_column(Text)

    # Execution / portfolio.
    position_before: Mapped[Optional[str]] = mapped_column(String(10))
    position_after: Mapped[Optional[str]] = mapped_column(String(10))

    # Fill that happened *on this day* (may come from the previous day's signal).
    executed_action: Mapped[Optional[str]] = mapped_column(String(10))
    executed_price: Mapped[Optional[float]] = mapped_column(Float)
    executed_shares: Mapped[Optional[float]] = mapped_column(Float)
    executed_fee: Mapped[Optional[float]] = mapped_column(Float)
    executed_signal_date: Mapped[Optional[dt.date]] = mapped_column(Date)

    shares: Mapped[Optional[float]] = mapped_column(Float)
    cash: Mapped[Optional[float]] = mapped_column(Float)
    equity: Mapped[Optional[float]] = mapped_column(Float)
    daily_return: Mapped[Optional[float]] = mapped_column(Float)
    cumulative_return: Mapped[Optional[float]] = mapped_column(Float)
    drawdown: Mapped[Optional[float]] = mapped_column(Float)
    buy_hold_equity: Mapped[Optional[float]] = mapped_column(Float)
    buy_hold_daily_return: Mapped[Optional[float]] = mapped_column(Float)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    backtest: Mapped["Backtest"] = relationship(back_populates="days")
    agent_responses: Mapped[list["BacktestAgentResponse"]] = relationship(
        back_populates="day",
        cascade="all, delete-orphan",
        order_by="BacktestAgentResponse.id",
    )


class BacktestAgentResponse(Base):
    """Output of one graph node (technical, flow, codal, industry, decision,
    risk, ...) for one day. Failed nodes are stored too, with status='error'.
    """

    __tablename__ = "backtest_agent_responses"
    __table_args__ = (
        Index("ix_bar_backtest_agent", "backtest_id", "agent"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)

    backtest_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("backtests.id", ondelete="CASCADE"),
        index=True,
    )
    day_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("backtest_days.id", ondelete="CASCADE"),
        index=True,
    )
    trade_date: Mapped[dt.date] = mapped_column(Date)

    # Which run of this day (a resumed backtest retries failed days).
    attempt: Mapped[int] = mapped_column(Integer, default=1)

    agent: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default="ok")
    error: Mapped[Optional[str]] = mapped_column(Text)

    # Quick-query fields extracted from the output:
    #   specialists -> bullish/neutral/bearish, decision -> buy/hold/sell,
    #   risk        -> approve/veto
    label: Mapped[Optional[str]] = mapped_column(String(20))
    score: Mapped[Optional[float]] = mapped_column(Float)
    confidence: Mapped[Optional[float]] = mapped_column(Float)

    # Full structured response of the agent.
    output: Mapped[Optional[dict]] = mapped_column(JSONType)

    # Evidence the agent was given (specialist agents only).
    inputs: Mapped[Optional[dict]] = mapped_column(JSONType)

    started_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)

    day: Mapped["BacktestDay"] = relationship(back_populates="agent_responses")
