from typing import TypedDict

from .schemas import TechnicalAnalysis, FinalDecision


class AgentState(TypedDict, total=False):
    symbol: str

    data_until: str

    technical_analysis: TechnicalAnalysis

    decision: FinalDecision