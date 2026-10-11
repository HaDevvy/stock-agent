"""Build the evidence bundle for the post-decision LLM risk reviewer.

Specialist agents publish the same data they passed into their prompts in
state['agent_inputs']. The risk agent reads it without querying the database
a second time.
"""

import json
from typing import Any

from .state import AgentState


EXPECTED_AGENTS = ("technical", "flow", "codal", "industry")


def build_risk_context(state: AgentState) -> dict[str, Any]:
    """Return decision, analyst findings, and original evidence together."""
    if "decision" not in state:
        raise ValueError("Risk review requires a decision from decision_agent.")

    analyses = state.get("analyses", [])
    inputs = state.get("agent_inputs", [])

    present_inputs = {item["agent"] for item in inputs}
    present_analyses = {item.agent for item in analyses}

    context: dict[str, Any] = {
        "symbol": state["symbol"],
        "data_until": state.get("data_until"),
        "decision_date": state.get("decision_date"),
        "proposed_decision": state["decision"].model_dump(),
        "agent_analyses": [item.model_dump() for item in analyses],
        "agent_inputs": inputs,
        "missing_agent_inputs": [
            agent for agent in EXPECTED_AGENTS if agent not in present_inputs
        ],
        "missing_agent_analyses": [
            agent for agent in EXPECTED_AGENTS if agent not in present_analyses
        ],
    }

    # An absent portfolio is unknown, not an empty portfolio.
    context["portfolio"] = state.get("portfolio")

    return context


def serialize_risk_context(context: dict[str, Any]) -> str:
    """Serialize dates, Decimals and database scalar types safely."""
    return json.dumps(context, ensure_ascii=False, indent=2, default=str)