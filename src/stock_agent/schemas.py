from typing import Literal

from pydantic import BaseModel, Field


class TechnicalAnalysis(BaseModel):
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