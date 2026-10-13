import json
import time

from ..common import compress_json_for_llm

from ..config import settings
from ..database import get_industry_data
from ..industry_features import (
    calculate_industry_features,
)
from ..llm import llm
from ..schemas import AgentAnalysis
from ..state import AgentState


industry_llm = llm.with_structured_output(
    AgentAnalysis
)


def industry_agent(
    state: AgentState,
):
    rows = get_industry_data(
        symbol=state["symbol"],
        data_until=state["data_until"],
        limit=settings.industry_lookback_days,
    )

    if not rows:
        raise ValueError(
            f"No industry data for {state['symbol']}"
        )

    features = calculate_industry_features(
        rows
    )

    # Database query may return descending dates.
    recent_rows = sorted(
        rows,
        key=lambda row: str(row["date"]),
        reverse=True,
    )[:5]

    # Keep only useful recent fields
    recent_fields = [
        "date",
        "top30_symbols",
        "top30_turnover_share",
        "equal_weight_return",
        "turnover_weighted_return",
        "top30_breadth",
        "industry_breadth",
        "real_net_money",
        "real_flow_strength",
        "real_buy_power",
    ]

    recent_rows = [
        {
            key: row.get(key)
            for key in recent_fields
        }
        for row in recent_rows
    ]

    prompt = f"""
    You are an Iranian stock market industry analyst.

    Your responsibility is to evaluate the overall
    condition of the industry associated with the
    target stock.

    Determine whether the current industry environment
    is bullish, neutral, or bearish.

    Symbol:
    {state["symbol"]}

    Data available until:
    {state["data_until"]}

    Industry analysis methodology:

    - Each trading day, the 30 stocks with the
      highest trading turnover are selected from
      the target stock's industry.

    - The selected 30 stocks may change from
      one trading day to another.

    - Industry features are calculated from
      these stocks.

    - Some metrics, such as industry_breadth,
      represent the broader industry.

    Calculated industry features:

    {json.dumps(
        compress_json_for_llm(features),
        ensure_ascii=False,
        indent=2,
        default=str,
    )}

    Recent industry data:

    {json.dumps(
        compress_json_for_llm(recent_rows),
        ensure_ascii=False,
        indent=2,
        default=str,
    )}

    Analyze the following factors:

    1. Industry trend

       - Is the industry generally strengthening
         or weakening?

       - Are recent returns positive or negative?

       - Is momentum improving or deteriorating?

    2. Market breadth

       - Are most active stocks participating
         in the industry's movement?

       - Is the movement broad-based or
         concentrated in a few stocks?

       - Compare top30_breadth with
         industry_breadth.

    3. Real investor money flow

       - Is real investor money entering
         or leaving the active part of the industry?

       - Is money flow persistent over
         multiple trading days?

       - Is the current flow supported
         by previous observations?

    4. Buyer versus seller power

       - Are real buyers stronger than sellers?

       - Is buyer power consistently above 1?

       - Is buyer power supported by
         positive real money flow?

    5. Institutional investor behavior

       - Are institutional/legal investors
         net buyers or net sellers?

       - Does their behavior reinforce or
         oppose real investor activity?

    6. Trading activity

       - Is trading turnover unusually high
         compared with recent history?

       - Does increased activity support
         the prevailing industry direction?

       - Could the movement be caused by
         temporary speculative activity?

    7. Industry participation and concentration

       - How much of total industry turnover
         is represented by the top 30 stocks?

       - Is the active-stock signal consistent
         with broader industry breadth?

       - Are there signs of divergence between
         the most actively traded stocks and
         the rest of the industry?

    8. Sustainability

       - Is the current industry strength
         supported by multiple factors?

       - Is the behavior persistent or
         potentially a one-day anomaly?

       - Are there signs of accumulation,
         distribution, continuation, or exhaustion?

    Important rules:

    - Do NOT calculate indicators yourself.
    - Use the provided calculated features.
    - Do NOT invent missing information.
    - Do NOT make a final BUY or SELL decision.
    - Evaluate the industry, not the technical
      condition of the individual target stock.
    - A single strong trading day is not
      sufficient for a bullish industry signal.
    - Persistent multi-day behavior must
      receive more weight than isolated events.
    - Positive industry returns without
      broad participation should be treated
      with caution.
    - Strong buyer power without sustained
      positive money flow is weaker evidence.
    - High turnover alone is not bullish.
    - The top 30 constituents change daily.
      Do not interpret rolling returns as
      a fixed portfolio's return.
    - Trading turnover weighting is not
      equivalent to market capitalization
      weighting.
    - Pay attention to data coverage.
      Reduce confidence when client-type
      coverage is incomplete.
    - If indicators conflict, explain
      the disagreement and avoid unjustified
      high confidence.
    - Missing features must not be interpreted
      as zero or neutral observations.

    Scoring guidelines:

    Bullish:
        Broad industry participation,
        persistent positive flow,
        strong real buyer power,
        and supportive trading activity.

    Neutral:
        Mixed signals, weak momentum,
        insufficient confirmation,
        or conflicting evidence.

    Bearish:
        Broad weakness, persistent
        real money outflow, seller dominance,
        and distribution-like activity.

    Return:

    signal:
        bullish / neutral / bearish

    score:
        from -1 to +1

    confidence:
        from 0 to 1

    summary:
        concise explanation of the industry's
        current condition and its implications
        for the target stock's environment.

    reasons:
        most important evidence supporting
        the analysis.

    Output the analysis in Persian.
    """

    analysis = industry_llm.invoke(
        prompt
    )
    time.sleep(settings.llm_rate_limit_delay)  # avoid rate limit

    analysis.agent = "industry"

    return {
        "analyses": [analysis],
        "agent_inputs": [{
            "agent": "industry",
            "inputs": {
                "features": features,
                "recent_rows": recent_rows,
            },
        }],
    }