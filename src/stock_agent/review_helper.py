from .state import AgentState


def get_recheck_instructions(
    state: AgentState,
    agent_name: str,
) -> str | None:

    review = state.get("decision_review")

    if review is None:
        return None

    if review.verdict != "recheck":
        return None

    return review.recheck_instructions.get(agent_name)