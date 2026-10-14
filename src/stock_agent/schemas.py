from typing import Literal
from pydantic import BaseModel, Field, model_validator


class AgentAnalysis(BaseModel):
    agent: str

    signal: Literal["bullish", "neutral", "bearish",]

    score: float = Field(ge=-1, le=1,)

    confidence: float = Field(ge=0, le=1,)

    summary: str

    reasons: list[str]


class FinalDecision(BaseModel):
    action: Literal["buy", "hold", "sell",]

    confidence: float = Field(ge=0, le=1,)

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


class DecisionReview(BaseModel):
    """Review the final decision after risk assessment."""

    verdict: Literal["approve", "recheck", "escalate",]

    target_agent: Literal["technical", "flow", "codal", "industry",] | None = None

    confidence: float = Field(ge=0, le=1)

    reason: str

    recheck_instructions: str | None = None

    @model_validator(mode="after")
    def validate_recheck(self):
        if self.verdict == "recheck":
            if not self.target_agent:
                raise ValueError("Recheck requires target_agent")
            if not self.recheck_instructions:
                raise ValueError("Recheck requires instructions")
        return self
