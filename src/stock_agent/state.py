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

import operator

from typing import Annotated, Any, TypedDict

from .schemas import AgentAnalysis, FinalDecision, RiskReview


class AgentInput(TypedDict):
    """Original evidence provided to a specialist agent."""

    agent: str
    inputs: dict[str, Any]


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

    agent_inputs: Annotated[
        list[AgentInput],
        operator.add,
    ]

    # Optional portfolio information.
    # Missing means unknown, not zero holdings.
    portfolio: dict[str, Any]

    decision: FinalDecision
    risk_review: RiskReview
    final_decision: FinalDecision
