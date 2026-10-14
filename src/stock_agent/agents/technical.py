import time

from stock_agent.review_helper import get_recheck_instructions

from ..common import compress_json_for_llm

from ..config import settings
from ..database import get_price_data
from ..indicators import calculate_technical_indicators
from ..llm import llm
from ..schemas import AgentAnalysis
from ..state import AgentState


if settings.llm_provider == "openai":
    technical_llm = llm.with_structured_output(AgentAnalysis)
elif settings.llm_provider == "google":
    technical_llm = llm.with_structured_output(AgentAnalysis, method="function_calling",)
else:
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def technical_agent(
    state: AgentState,
):
    prices = get_price_data(
        symbol=state["symbol"],
        data_until=state["data_until"],
        limit=settings.technical_lookback_days,
    )

    indicators = calculate_technical_indicators(
        prices
    )

    recent_prices = prices[-10:]

    prompt = f"""
    You are a technical stock market analyst.

    Symbol:
    {state["symbol"]}

    Data available until:
    {state["data_until"]}

    Technical indicators:
    {compress_json_for_llm(indicators)}
    
    Recent price candles:
    {compress_json_for_llm(recent_prices)}
    
    Analyze the stock based on the provided
    technical indicators and recent price behavior.
    
    Consider:
    
    - trend
    - momentum
    - volatility
    - RSI condition
    - MACD condition
    - moving averages
    - volume behavior
    - distance from recent high/low
    - possible continuation or reversal
    
    Do NOT calculate indicators yourself.
    The provided indicator values are already calculated.
    

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

    analysis = technical_llm.invoke(prompt)
    time.sleep(settings.llm_rate_limit_delay)  # avoid rate limit

    analysis.agent = "technical"

    return {
        "analyses": [analysis],
        "agent_inputs": [{
            "agent": "technical",
            "inputs": {
                "indicators": indicators,
                "recent_prices": recent_prices,
            },
        }],
    }