from ..database import get_price_data
from ..llm import llm
from ..schemas import TechnicalAnalysis
from ..state import AgentState


technical_llm = llm.with_structured_output(
    TechnicalAnalysis
)


def technical_agent(
    state: AgentState,
):
    prices = get_price_data(
        symbol=state["symbol"],
        data_until=state["data_until"],
    )

    prompt = f"""
    You are a technical stock market analyst.

    Analyze the following price data.

    Symbol:
    {state["symbol"]}

    Historical data:
    {prices}

    Analyze:
    - trend
    - momentum
    - volatility
    - support/resistance
    - volume
    - possible continuation or reversal

    Return a structured technical analysis.
    """

    analysis = technical_llm.invoke(prompt)

    return {
        "technical_analysis": analysis
    }