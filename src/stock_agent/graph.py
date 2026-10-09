from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from .state import AgentState
from .agents.technical import technical_agent
from .agents.decision import decision_agent


def build_graph():

    builder = StateGraph(AgentState)

    builder.add_node(
        "technical",
        technical_agent,
    )

    builder.add_node(
        "decision",
        decision_agent,
    )

    builder.add_edge(
        START,
        "technical",
    )

    builder.add_edge(
        "technical",
        "decision",
    )

    builder.add_edge(
        "decision",
        END,
    )

    return builder.compile()