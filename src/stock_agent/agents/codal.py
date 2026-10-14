import json
import time

from ..database import get_codal_data
from ..codal_file_manager import compact_for_llm, read_codal_data
from ..llm import llm
from ..schemas import AgentAnalysis
from ..state import AgentState
from ..config import settings

if settings.llm_provider == "openai":
    codal_llm = llm.with_structured_output(AgentAnalysis)
elif settings.llm_provider == "google":
    codal_llm = llm.with_structured_output(AgentAnalysis, method="function_calling",)
else:
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def codal_agent(state: AgentState):
    rows = get_codal_data(
        symbol=state["symbol"],
        data_until=state["data_until"],
        limit=1,
    )

    codal_reports = []

    for row in rows:
        data = read_codal_data(row["path"])

        compressed_data = compact_for_llm(data)

        codal_reports.append({
            "tracing_no": row["tracing_no"],
            "date": row["date"],
            "company_name": row["company_name"],
            "financial_data": compressed_data,
        })

    prompt = f"""
    You are a senior fundamental analyst specializing
    in companies listed on the Iranian stock market.

    Your task is to evaluate the fundamental outlook
    of the company based on financial and operational
    data extracted from official CODAL Excel reports.

    Symbol: {state["symbol"]}

    Data available until: {state["data_until"]}

    CODAL financial reports:

    {json.dumps(
        codal_reports,
        ensure_ascii=False,
        indent=2,
        default=str,
    )}

    IMPORTANT DATA RULES:

    - The primary source of fundamental information
    is the financial_data field.

    - Each report contains its publication date
    and the corresponding extracted Excel data.

    - Analyze financial and operational performance
    based on actual reported values.

    - Compare periods only when comparable data
    is available.

    - Consider sales growth, production performance,
    revenue, profitability, margins, costs,
    and financial stability when supported by data.

    - Distinguish absolute growth from percentage growth.

    - Pay attention to financial units, reporting
    periods, and differences between monthly,
    quarterly, and annual reports.

    - Do not assume that every Excel file contains
    the same financial metrics.

    - Do not confuse cumulative figures with
    single-period figures.

    - Do not double-count overlapping financial periods.

    - Do not invent missing financial information.

    - Give more weight to consistent multi-period
    fundamental improvements than isolated changes.

    - Never assume that improving revenue
    automatically means improving profitability.

    - A positive fundamental assessment does not
    necessarily imply an immediate stock price rise.


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

    analysis = codal_llm.invoke(prompt)
    time.sleep(settings.llm_rate_limit_delay)  # avoid rate limit

    analysis.agent = "codal"

    return {
        "analyses": [analysis],
        "agent_inputs": [{
            "agent": "codal",
            "inputs": {
                "codal_reports": codal_reports,
            },
        }],
    }