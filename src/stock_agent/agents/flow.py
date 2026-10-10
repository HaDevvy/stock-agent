import json

from ..database import get_flow_data
from ..flow_features import (
    calculate_flow_features,
)
from ..llm import llm
from ..schemas import AgentAnalysis
from ..state import AgentState


flow_llm = llm.with_structured_output(
    AgentAnalysis
)


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
        features,
        ensure_ascii=False,
        indent=2,
    )}
    
    Recent flow data:
    
    {json.dumps(
        recent_rows,
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
    
    Return:
    
    signal:
        bullish / neutral / bearish
    
    score:
        from -1 to +1
    
    confidence:
        from 0 to 1
    
    summary:
        concise explanation
    
    reasons:
        most important evidence
    """

    analysis = flow_llm.invoke(
        prompt
    )

    analysis.agent = "flow"

    return {
        "analyses": [analysis]
    }