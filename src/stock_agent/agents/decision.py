import json
import time

from ..llm import llm
from ..schemas import FinalDecision
from ..state import AgentState
from ..config import settings

decision_llm = llm.with_structured_output(
    FinalDecision
)


def decision_agent(
    state: AgentState,
):
    analyses = state["analyses"]

    analyses_data = [
        analysis.model_dump()
        for analysis in analyses
    ]

    prompt = f"""
    You are the final stock decision maker.
    
    You have received analyses from multiple
    specialized agents.
    
    Symbol:
    {state["symbol"]}
    
    Analyses:
    
    {json.dumps(
        analyses_data,
        ensure_ascii=False,
        indent=2
    )}
    
    Consider all available analyses.
    
    Pay attention to:
    - signal agreement
    - conflicting evidence
    - confidence of each agent
    - strength of reasons
    
    Make exactly one decision:
    
    - buy
    - hold
    - sell
    
    Return a structured final decision.
    """

    decision = decision_llm.invoke(prompt)
    time.sleep(settings.llm_rate_limit_delay)  # avoid rate limit

    return {
        "decision": decision
    }