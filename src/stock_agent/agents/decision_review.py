import json
import time


from ..common import compress_json_for_llm
from ..llm import llm
from ..schemas import DecisionReview
from ..state import AgentState
from ..config import settings


MAX_RECHECKS = 2

reviewer_llm = llm.with_structured_output(DecisionReview)


def final_reviewer(state: GraphState) -> dict:
    previous_review = state.get("decision_review")

    context = {
        "question": state["question"],
        "analyses": {
            name: state[f"{name}_analysis"].model_dump()
            for name in (
                "technical", "flow", "codal", "industry"
            )
        },
        "decision": state["final_decision"].model_dump(),
        "risk": state["risk_review"].model_dump(),
        "recheck_count": state.get("recheck_count", 0),
        "previous_review": (
            previous_review.model_dump()
            if previous_review else None
        ),
    }

    review = reviewer_llm.invoke([
        (
            "system",
            """
            You are an independent financial decision reviewer.

            Evaluate the proposed decision using all agent
            analyses and the independent risk review.

            Check for:
            - Contradictory signals or reasoning
            - Unsupported conclusions
            - Missing or outdated information
            - Unresolved risk factors

            Choose:
            approve: Evidence sufficiently supports the decision
                     and risk verdict is approve.
            recheck: A specific agent needs to verify something.
            escalate: Evidence remains insufficient, the risk
                      veto is unresolved, or review cannot
                      produce a reliable decision.

            For recheck:
            - Select exactly one agent.
            - Provide specific recheck instructions.

            Never approve a decision with a risk veto.
            """
        ),
        ("human", json.dumps(context, ensure_ascii=False))
    ])

    count = state.get("recheck_count", 0)

    # Enforce risk veto outside the LLM
    if (
        review.verdict == "approve"
        and state["risk_review"].verdict == "veto"
    ):
        review = DecisionReview(
            verdict="escalate",
            confidence=review.confidence,
            reason="Risk veto remains unresolved.",
        )

    # Enforce maximum number of rechecks
    if review.verdict == "recheck" and count >= MAX_RECHECKS:
        review = DecisionReview(
            verdict="escalate",
            confidence=review.confidence,
            reason="Maximum recheck attempts reached.",
        )

    return {
        "decision_review": review,
        "recheck_count": (
            count + 1 if review.verdict == "recheck"
            else count
        ),
    }