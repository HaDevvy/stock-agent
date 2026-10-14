
import operator

from typing import Annotated, Any, TypedDict

from .schemas import (
    AgentAnalysis,
    FinalDecision,
    RiskReview,
    DecisionReview,
)


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

    # Append-only: retains every analysis
    analyses: Annotated[
        list[AgentAnalysis],
        operator.add,
    ]

    # Append-only: retains every evidence record
    agent_inputs: Annotated[
        list[AgentInput],
        operator.add,
    ]

    portfolio: dict[str, Any]

    decision: FinalDecision

    decision_history: Annotated[
        list[FinalDecision],
        operator.add,
    ]

    risk_review: RiskReview
    final_decision: FinalDecision

    # New reviewer fields
    decision_review: DecisionReview
    recheck_count: int

    recheck_completions: Annotated[
        list[tuple[int, str]],
        operator.add,
    ]

    # Append-only reviewer history
    review_history: Annotated[
        list[DecisionReview],
        operator.add,
    ]
