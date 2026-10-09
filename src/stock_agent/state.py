import operator

from typing import Annotated, TypedDict

from .schemas import AgentAnalysis, FinalDecision


class AgentState(TypedDict, total=False):
    symbol: str

    decision_date: str

    data_until: str

    technical_indicators: dict[
        str,
        float | None,
    ]

    analyses: Annotated[
        list[AgentAnalysis],
        operator.add,
    ]

    decision: FinalDecision