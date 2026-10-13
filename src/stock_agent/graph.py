from pathlib import Path
from typing import Callable

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from .graph_loader import (
    load_graph_config,
    load_handler,
)
from .state import AgentState


def resolve_node_name(
    node_name: str,
):
    if node_name == "START":
        return START

    if node_name == "END":
        return END

    return node_name


def build_graph(
    config_path: str | Path | None = None,
    handler_wrapper: Callable[[str, Callable], Callable] | None = None,
):
    """Build the graph from graph.yaml.

    handler_wrapper(node_name, handler) -> handler lets callers (e.g. the
    backtest) observe every node without changing the agents themselves.
    """
    config = load_graph_config(
        config_path
    )

    builder = StateGraph(
        AgentState
    )

    # -------------------------
    # Nodes
    # -------------------------

    for (
        node_name,
        node_config,
    ) in config["nodes"].items():

        handler = load_handler(
            node_config["handler"]
        )

        if handler_wrapper is not None:
            handler = handler_wrapper(
                node_name,
                handler,
            )

        builder.add_node(
            node_name,
            handler,
        )

    # -------------------------
    # Edges
    # -------------------------

    for edge in config["edges"]:

        source = edge["from"]
        target = edge["to"]

        if isinstance(source, list):

            source = [
                resolve_node_name(item)
                for item in source
            ]

        else:
            source = resolve_node_name(
                source
            )

        target = resolve_node_name(
            target
        )

        builder.add_edge(
            source,
            target,
        )

    return builder.compile()