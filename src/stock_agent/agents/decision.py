import json
import time

from ..llm import llm
from ..schemas import FinalDecision
from ..state import AgentState
from ..config import settings

if settings.llm_provider == "openai":
    decision_llm = llm.with_structured_output(FinalDecision)
elif settings.llm_provider == "google":
    decision_llm = llm.with_structured_output(FinalDecision, method="function_calling",)
else:
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def decision_agent(
    state: AgentState,
):
    analyses = state["analyses"]

    latest_analyses = {}
    for analysis in analyses:
        latest_analyses[analysis.agent] = analysis

    analyses_data = [
        analysis.model_dump()
        for analysis in latest_analyses.values()
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
    
    ***MANDATORY FINAL DECISION OUTPUT***
    Evaluate the available agent analyses and relevant evidence, then return a FinalDecision strictly matching the defined schema.


    Return structured output with:
    - action: Exactly one of "buy", "hold", or "sell".
    - confidence: A float between 0.0 and 1.0.
    - reason: A concise explanation for this final decision.

    Make the decision based on the strength, consistency, and reliability of the evidence. 
    Consider conflicting signals and uncertainty. Do not invent facts or treat missing information as certainty. 
    Ensure the output matches the FinalDecision schema.
    """

    decision = decision_llm.invoke(prompt)
    time.sleep(settings.llm_rate_limit_delay)  # avoid rate limit

    return {
        "decision": decision,
        "decision_history": [decision],
    }