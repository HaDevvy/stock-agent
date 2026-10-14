
from functools import partial
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


def resolve_node_name(node_name: str):
    if node_name == "START":
        return START

    if node_name == "END":
        return END

    return node_name


def build_graph(
    config_path: str | Path | None = None,
    handler_wrapper: Callable[[str, Callable], Callable] | None = None,
):
    """Build the graph dynamically from graph.yaml."""

    config = load_graph_config(config_path)
    review_config = config.get("review")

    builder = StateGraph(AgentState)

    def wrap_handler(node_name, handler):
        if handler_wrapper is not None:
            return handler_wrapper(node_name, handler)
        return handler

    # -------------------------
    # Original nodes
    # -------------------------

    for node_name, node_config in config["nodes"].items():
        handler = load_handler(node_config["handler"])

        # Inject YAML settings into the reviewer.
        if (
            review_config is not None
            and node_name == review_config["node"]
        ):
            handler = partial(
                handler,
                allowed_agents=tuple(review_config["agents"]),
                max_rechecks=review_config.get(
                    "max_rechecks", 2
                ),
            )

        handler = wrap_handler(node_name, handler)

        builder.add_node(node_name, handler)

    # -------------------------
    # Original edges
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
            source = resolve_node_name(source)

        target = resolve_node_name(target)

        builder.add_edge(source, target)

    # -------------------------
    # Optional decision review
    # -------------------------

    if review_config is not None:
        reviewer_name = review_config["node"]
        allowed_agents = tuple(review_config["agents"])
        return_to = review_config["return_to"]

        join_name = "recheck_join"

        if join_name in config["nodes"]:
            raise ValueError(
                f"Reserved node name already exists: {join_name}"
            )

        # Track completion after each agent finishes.
        def make_recheck_handler(agent_name, handler):
            def recheck_handler(state: AgentState):
                result = handler(state)

                if not isinstance(result, dict):
                    raise TypeError(
                        "Recheck handlers must return dict updates."
                    )

                return {
                    **result,
                    "recheck_completions": [
                        (
                            state["recheck_count"],
                            agent_name,
                        )
                    ],
                }

            return recheck_handler

        # Create one recheck node for each allowed agent
        for agent_name in allowed_agents:
            recheck_name = f"recheck_{agent_name}"

            handler_path = config["nodes"][
                agent_name
            ]["handler"]

            handler = load_handler(handler_path)

            handler = make_recheck_handler(
                agent_name,
                handler,
            )

            handler = wrap_handler(
                recheck_name,
                handler,
            )

            builder.add_node(recheck_name, handler)

        # Join node: receives results from rechecked agents
        builder.add_node(
            join_name,
            wrap_handler(
                join_name,
                lambda state: {},
            ),
        )

        for agent_name in allowed_agents:
            builder.add_edge(
                f"recheck_{agent_name}",
                join_name,
            )

        # Reviewer may return multiple destinations
        def review_router(state: AgentState) -> list[str]:
            review = state["decision_review"]

            if review.verdict != "recheck":
                return [END]

            invalid = (
                set(review.target_agents)
                - set(allowed_agents)
            )

            if invalid:
                raise ValueError(
                    f"Invalid recheck targets: {sorted(invalid)}"
                )

            return [
                f"recheck_{name}"
                for name in review.target_agents
            ]

        routes = {
            f"recheck_{name}": f"recheck_{name}"
            for name in allowed_agents
        }
        routes[END] = END

        builder.add_conditional_edges(
            reviewer_name,
            review_router,
            routes,
        )

        # Run decision only when ALL selected agents finish
        def join_router(state: AgentState) -> str:
            review = state["decision_review"]
            current_round = state["recheck_count"]

            completed = {
                agent
                for round_id, agent in state.get(
                    "recheck_completions", []
                )
                if round_id == current_round
            }

            required = set(review.target_agents)

            if required.issubset(completed):
                return return_to

            return END

        builder.add_conditional_edges(
            join_name,
            join_router,
            {
                return_to: return_to,
                END: END,
            },
        )

    return builder.compile()
