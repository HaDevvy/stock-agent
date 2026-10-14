
import json
import time

from ..llm import llm
from ..schemas import DecisionReview
from ..state import AgentState
from ..config import settings


if settings.llm_provider == "openai":
    reviewer_llm = llm.with_structured_output(
        DecisionReview
    )
elif settings.llm_provider == "google":
    reviewer_llm = llm.with_structured_output(
        DecisionReview,
        method="function_calling",
    )
else:
    raise ValueError(
        f"Unsupported LLM provider: {settings.llm_provider}"
    )


def decision_review_agent(
    state: AgentState,
    *,
    allowed_agents: tuple[str, ...],
    max_rechecks: int = 2,
):
    analyses = state.get("analyses", [])

    # Extract latest analysis without removing history
    latest_analyses = {}

    for analysis in analyses:
        latest_analyses[analysis.agent] = analysis

    # Only configured specialists that have results
    available_agents = [
        name
        for name in allowed_agents
        if name in latest_analyses
    ]

    # Preserve previous review context
    review_history = state.get("review_history", [])

    count = state.get("recheck_count", 0)

    context = {
        "symbol": state["symbol"],
        "decision_date": state.get("decision_date"),
        "data_until": state.get("data_until"),

        "available_recheck_agents": available_agents,

        "latest_analyses": {
            name: item.model_dump()
            for name, item in latest_analyses.items()
        },

        "analysis_history": [
            item.model_dump()
            for item in analyses
        ],

        "decision": state["decision"].model_dump(),

        "risk_review": state["risk_review"].model_dump(),

        "final_decision": state["final_decision"].model_dump(),

        "review_history": [
            item.model_dump()
            for item in review_history
        ],

        "recheck_count": count,
        "max_rechecks": max_rechecks,
    }

    prompt = f"""
    You are an independent senior financial decision reviewer.

    Available recheck agents:
    {json.dumps(available_agents)}

    Analysis and decision context:
    {json.dumps(context, ensure_ascii=False, indent=2)}

    Review the analyses, decision, risk assessment,
    and previous review history.

    Select one of these verdicts:

    1. approve
    The risk-adjusted decision is sufficiently supported.

    2. recheck
    One or more specialists need to verify important issues.

    You may select MULTIPLE agents simultaneously.

    Set target_agents to the selected agent names.

    For every selected agent, provide separate
    and specific instructions in recheck_instructions.

    Only select agents from available_recheck_agents.

    Do not request unrelated or unnecessary rechecks.

    3. escalate
    The decision cannot be approved and the available
    specialists cannot resolve the uncertainty.

    Important:
    - Previous analyses are historical evidence.
    - Use only the latest analysis per agent as the
    current recommendation.
    - Compare history to determine whether previous
    recheck requests were resolved.
    - Never approve a directional trade with an
    unresolved risk veto.

    Return a DecisionReview matching the schema.
    """

    review = reviewer_llm.invoke(prompt)

    time.sleep(settings.llm_rate_limit_delay)

    invalid_agents = (
        set(review.target_agents)
        - set(available_agents)
    )

    if invalid_agents:
        review = DecisionReview(
            verdict="escalate",
            confidence=review.confidence,
            reason=(
                "Invalid recheck targets: "
                f"{sorted(invalid_agents)}"
            ),
        )

    if (
        review.verdict == "recheck"
        and count >= max_rechecks
    ):
        review = DecisionReview(
            verdict="escalate",
            confidence=review.confidence,
            reason="Maximum recheck rounds reached.",
        )

    if (
        review.verdict == "approve"
        and state["risk_review"].verdict == "veto"
        and state["final_decision"].action != "hold"
    ):
        review = DecisionReview(
            verdict="escalate",
            confidence=review.confidence,
            reason="Unresolved risk veto.",
        )

    return {
        "decision_review": review,
        "review_history": [review],
        "recheck_count": (
            count + 1
            if review.verdict == "recheck"
            else count
        ),
    }
