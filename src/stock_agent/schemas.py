from typing import Literal

from pydantic import BaseModel, Field


class AgentAnalysis(BaseModel):
    agent: str

    signal: Literal[
        "bullish",
        "neutral",
        "bearish",
    ]

    score: float = Field(
        ge=-1,
        le=1,
    )

    confidence: float = Field(
        ge=0,
        le=1,
    )

    summary: str

    reasons: list[str]


class FinalDecision(BaseModel):
    action: Literal[
        "buy",
        "hold",
        "sell",
    ]

    confidence: float = Field(
        ge=0,
        le=1,
    )

    reason: str


class RiskReview(BaseModel):
    """Independent LLM verification of a proposed directional decision."""

    verdict: Literal["approve", "veto"]
    risk_level: Literal["low", "medium", "high", "unknown"]
    confidence: float = Field(ge=0, le=1)
    reason: str
    risk_factors: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    suggested_precautions: list[str] = Field(default_factory=list)