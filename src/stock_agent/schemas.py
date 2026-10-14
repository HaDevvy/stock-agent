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
    verdict: Literal[
        "approve",
        "recheck",
        "escalate",
    ]

    target_agents: list[str] = Field(
        default_factory=list,
    )

    confidence: float = Field(ge=0, le=1)

    reason: str

    recheck_instructions: dict[str, str] = Field(
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_recheck(self):
        if self.verdict == "recheck":
            if not self.target_agents:
                raise ValueError(
                    "At least one target agent is required."
                )

            if len(self.target_agents) != len(
                set(self.target_agents)
            ):
                raise ValueError(
                    "Duplicate target agents are not allowed."
                )

            if (
                set(self.recheck_instructions)
                != set(self.target_agents)
            ):
                raise ValueError(
                    "Every target agent must have instructions."
                )

            if any(
                not instruction.strip()
                for instruction in self.recheck_instructions.values()
            ):
                raise ValueError(
                    "Recheck instructions cannot be empty."
                )

        elif self.target_agents or self.recheck_instructions:
            raise ValueError(
                "Only recheck verdicts may specify targets."
            )

        return self


