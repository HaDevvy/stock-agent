import json
import time

from stock_agent.review_helper import get_recheck_instructions

from ..common import compress_json_for_llm
from ..config import settings

from ..database import get_flow_data
from ..flow_features import (
    calculate_flow_features,
)
from ..llm import llm
from ..schemas import AgentAnalysis
from ..state import AgentState


if settings.llm_provider == "openai":
    flow_llm = llm.with_structured_output(AgentAnalysis)
elif settings.llm_provider == "google":
    flow_llm = llm.with_structured_output(AgentAnalysis, method="function_calling",)
else:
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def flow_agent(
    state: AgentState,
):
    rows = get_flow_data(
        symbol=state["symbol"],
        data_until=state["data_until"],
        limit=30,
    )

    features = calculate_flow_features(
        rows
    )

    recent_rows = rows[:5]

    prompt = f"""
    You are a stock market order-flow analyst.
    
    Your responsibility is to analyze the behavior
    of buyers and sellers and determine whether the
    current trading flow is bullish, neutral, or bearish.
    
    Symbol:
    {state["symbol"]}
    
    Data available until:
    {state["data_until"]}
    
    Calculated flow features:
    
    {json.dumps(
        compress_json_for_llm(features),
        ensure_ascii=False,
        indent=2,
    )}
    
    Recent flow data:
    
    {json.dumps(
        compress_json_for_llm(recent_rows),
        ensure_ascii=False,
        indent=2,
        default=str,
    )}
    
    Analyze the following factors:
    
    1. real buyer power versus seller power.
    
    2. real investor money flow.
    
    3. Institutional/legal investor behavior.
    
    4. Whether trading volume is unusually high
       compared with recent history.
    
    5. Whether buying or selling pressure has been
       persistent during recent days.
    
    6. Whether the current flow appears sustainable
       or is only a one-day anomaly.
    
    7. Whether there are signs of accumulation
       or distribution.
    
    Important:
    
    - Do not calculate the metrics yourself.
    - Use the provided calculated features.
    - Do not make a final BUY or SELL decision.
    - Your job is only to evaluate trading flow.
    - A single strong day should not automatically
      result in a bullish signal.
    - Persistent multi-day behavior should receive
      more weight than a single day.
    
    ***MANDATORY AGENT ANALYSIS OUTPUT***
    Analyze the available evidence and return an AgentAnalysis strictly matching the defined schema.

    Return structured output with:
    - agent: Your identifier string (e.g., "technical", "flow", "codal", or "industry").
    - signal: Exactly one of "bullish", "neutral", or "bearish".
    - score: A float between -1.0 and 1.0.
    - confidence: A float between 0.0 and 1.0.
    - summary: A brief summary string.
    - reasons: A list of key supporting bullet points.

    Base your analysis on the available evidence. Ensure the signal, score, confidence, summary, and reasons are consistent. 
    Do not invent facts. 
    Return output matching the AgentAnalysis schema.
    """
    instructions = get_recheck_instructions(
        state,
        "codal",
    )

    if instructions:
        prompt += f"""

        *** RECHECK REQUEST ***

        The final reviewer requested a re-analysis.

        Specific instructions:
        {instructions}

        Investigate the identified issue using
        the available evidence.

        Correct your previous conclusion if necessary.

        Do not automatically repeat your previous result.
        """

    analysis = flow_llm.invoke(
        prompt
    )
    time.sleep(settings.llm_rate_limit_delay)  # avoid rate limit

    analysis.agent = "flow"

    return {
        "analyses": [analysis],
        "agent_inputs": [{
            "agent": "flow",
            "inputs": {
                "features": features,
                "recent_rows": recent_rows,
            },
        }],
    }