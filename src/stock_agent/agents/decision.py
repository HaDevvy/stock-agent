from ..llm import llm
from ..schemas import FinalDecision
from ..state import AgentState


decision_llm = llm.with_structured_output(
    FinalDecision
)


def decision_agent(
    state: AgentState,
):
    technical = state["technical_analysis"]

    prompt = f"""
    You are the final decision maker.

    Technical analysis:

    Signal:
    {technical.signal}

    Score:
    {technical.score}

    Confidence:
    {technical.confidence}

    Summary:
    {technical.summary}

    Reasons:
    {technical.reasons}

    Based only on this analysis,
    make one decision:

    - buy
    - hold
    - sell
    """

    decision = decision_llm.invoke(prompt)

    return {
        "decision": decision
    }