import time

from ..common import compress_json_for_llm

from ..config import settings
from ..database import get_price_data
from ..indicators import calculate_technical_indicators
from ..llm import llm
from ..schemas import AgentAnalysis
from ..state import AgentState


technical_llm = llm.with_structured_output(
    AgentAnalysis
)


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
    
    Return a structured analysis.
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