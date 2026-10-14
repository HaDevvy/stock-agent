import json
import datetime as dt
from decimal import Decimal
from typing import Any

from stock_agent.schemas import AgentAnalysis


def compress_json_for_llm(data: Any) -> str:
    """
    Compress Python data into LLM-friendly JSON.

    Supports:
    - list[dict]
    - Nested dictionaries and lists
    - datetime.date and datetime.datetime
    - Decimal values
    - Standard JSON types

    Uses column-based compression when it reduces size.
    """

    def normalize(value):
        # Convert dates to ISO strings
        if isinstance(value, (dt.date, dt.datetime, dt.time)):
            return value.isoformat()

        # Convert Decimal without losing precision
        if isinstance(value, Decimal):
            if value.is_finite() and value == value.to_integral_value():
                return int(value)
            return str(value)

        if isinstance(value, dict):
            return {
                key: normalize(val)
                for key, val in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]

        return value

    def dumps(value):
        return json.dumps(
            value,
            separators=(",", ":"),
            ensure_ascii=False
        )

    def compress(value):
        if isinstance(value, dict):
            return {
                key: compress(val)
                for key, val in value.items()
            }

        if isinstance(value, list):
            items = [compress(item) for item in value]

            if len(items) < 2:
                return items

            if not all(isinstance(item, dict) for item in items):
                return items

            columns = list(items[0].keys())

            if not all(set(item.keys()) == set(columns) for item in items):
                return items

            table = {
                "_cols": columns,
                "_rows": [
                    [item[col] for col in columns]
                    for item in items
                ]
            }

            if len(dumps(table)) < len(dumps(items)):
                return table

            return items

        return value

    # If input is already a JSON string
    if isinstance(data, str):
        data = json.loads(data)

    normalized = normalize(data)
    return dumps(compress(normalized))



def get_latest_analyses(
    analyses: list[AgentAnalysis],
) -> dict[str, AgentAnalysis]:
    """
    Return the latest analysis for each agent,
    while preserving the original history.
    """
    latest = {}

    for analysis in analyses:
        latest[analysis.agent] = analysis

    return latest


def get_agent_history(
    analyses: list[AgentAnalysis],
    agent_name: str,
) -> list[AgentAnalysis]:
    return [
        analysis
        for analysis in analyses
        if analysis.agent == agent_name
    ]
