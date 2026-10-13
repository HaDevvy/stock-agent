"""PNG rendering for backtests (matplotlib, works offline).

* render_graph_png(config)  -> the agent graph defined in graph.yaml
* render_equity_chart(rows) -> strategy equity vs buy & hold, plus price with
                               executed buy/sell markers

Titles avoid the symbol name on purpose: matplotlib cannot shape Persian text
without extra fonts, so it would render as empty boxes.
"""

import datetime as dt
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Agent graph
# ---------------------------------------------------------------------------

def _graph_edges(config: dict[str, Any]) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []

    for edge in config["edges"]:
        sources = edge["from"]
        if not isinstance(sources, list):
            sources = [sources]
        for source in sources:
            edges.append((source, edge["to"]))

    return edges


def render_graph_png(config: dict[str, Any]) -> bytes:
    """Draw the graph.yaml DAG left-to-right and return PNG bytes."""
    edges = _graph_edges(config)

    names = ["START", *config["nodes"].keys(), "END"]

    # Longest-path layering from START.
    layer = {name: 0 for name in names}
    for _ in range(len(names)):
        changed = False
        for source, target in edges:
            if layer[target] < layer[source] + 1:
                layer[target] = layer[source] + 1
                changed = True
        if not changed:
            break
    else:
        raise ValueError("graph.yaml contains a cycle")

    layers: dict[int, list[str]] = {}
    for name in names:
        layers.setdefault(layer[name], []).append(name)

    position: dict[str, tuple[float, float]] = {}
    for x, members in layers.items():
        count = len(members)
        for index, name in enumerate(members):
            position[name] = (x, (count - 1) / 2 - index)

    max_rows = max(len(members) for members in layers.values())

    fig, ax = plt.subplots(
        figsize=(2.3 * len(layers) + 1, 0.95 * max_rows + 1.4),
    )
    ax.set_axis_off()
    ax.set_xlim(-0.6, len(layers) - 0.4)
    ax.set_ylim(-(max_rows - 1) / 2 - 0.7, (max_rows - 1) / 2 + 0.7)

    for source, target in edges:
        ax.annotate(
            "",
            xy=position[target],
            xytext=position[source],
            arrowprops={
                "arrowstyle": "->",
                "color": "#6c5ce7",
                "shrinkA": 30,
                "shrinkB": 30,
            },
        )

    for name, (x, y) in position.items():
        if name == "START":
            face = "white"
        elif name == "END":
            face = "#bfb6fc"
        else:
            face = "#f2f0ff"

        ax.text(
            x,
            y,
            name.lower() if name in ("START", "END") else name,
            ha="center",
            va="center",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.5",
                "fc": face,
                "ec": "#6c5ce7",
            },
        )

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Equity chart
# ---------------------------------------------------------------------------

def render_equity_chart(rows: list[dict], title: str) -> bytes:
    """rows: dicts with trade_date, equity, buy_hold_equity, close,
    executed_action, position_after (in date order)."""
    dates = [
        r["trade_date"]
        if isinstance(r["trade_date"], dt.date)
        else dt.date.fromisoformat(str(r["trade_date"])[:10])
        for r in rows
    ]

    fig, (ax_eq, ax_px) = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # --- equity -----------------------------------------------------------
    ax_eq.plot(
        dates,
        [r["equity"] for r in rows],
        label="Agent strategy",
        color="#6c5ce7",
        linewidth=1.8,
    )
    ax_eq.plot(
        dates,
        [r["buy_hold_equity"] for r in rows],
        label="Buy & hold",
        color="#888888",
        linewidth=1.4,
        linestyle="--",
    )
    ax_eq.set_title(title)
    ax_eq.set_ylabel("Equity")
    ax_eq.grid(alpha=0.3)
    ax_eq.legend(loc="best")

    # --- price + trades -----------------------------------------------------
    closes = [r["close"] for r in rows]
    ax_px.plot(dates, closes, color="#333333", linewidth=1.2)
    ax_px.set_ylabel("Close (adjusted)")
    ax_px.grid(alpha=0.3)

    # shade the days the strategy ends the day long
    start = None
    for index, row in enumerate(rows):
        long_now = row["position_after"] == "long"
        if long_now and start is None:
            start = index
        if (not long_now or index == len(rows) - 1) and start is not None:
            ax_px.axvspan(
                dates[start],
                dates[index],
                color="#6c5ce7",
                alpha=0.12,
                linewidth=0,
            )
            start = None

    buys = [
        (d, r["executed_price"] or r["close"])
        for d, r in zip(dates, rows)
        if r["executed_action"] == "buy"
    ]
    sells = [
        (d, r["executed_price"] or r["close"])
        for d, r in zip(dates, rows)
        if r["executed_action"] == "sell"
    ]

    if buys:
        ax_px.scatter(*zip(*buys), marker="^", color="#2ecc71", s=70,
                      zorder=3, label="Buy fill")
    if sells:
        ax_px.scatter(*zip(*sells), marker="v", color="#e74c3c", s=70,
                      zorder=3, label="Sell fill")
    if buys or sells:
        ax_px.legend(loc="best")

    fig.autofmt_xdate()
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=130)
    plt.close(fig)

    return buffer.getvalue()
