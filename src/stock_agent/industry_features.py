import math

import pandas as pd


def calculate_industry_features(
    rows: list[dict],
) -> dict:

    if not rows:
        raise ValueError("No industry data available")

    df = pd.DataFrame(rows)

    required_columns = {
        "date",
        "top30_symbol_count",
        "industry_symbol_count",
        "top30_turnover",
        "top30_turnover_share",
        "equal_weight_return",
        "turnover_weighted_return",
        "top30_breadth",
        "industry_breadth",
        "real_net_money",
        "real_flow_strength",
        "real_buy_power",
        "legal_net_money",
        "client_turnover_coverage_ratio",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing industry columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError(
            "Industry data must contain one row per date"
        )

    df = df.sort_values("date").reset_index(drop=True)

    for col in required_columns - {"date"}:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    def number(value):
        if pd.isna(value):
            return None

        result = float(value)

        return result if math.isfinite(result) else None

    def window_5(column):
        if len(df) < 5:
            return None

        series = df[column].tail(5)

        if series.isna().any():
            return None

        return series

    latest = df.iloc[-1]

    # -----------------------------------
    # Trend
    # -----------------------------------

    returns_5d = window_5(
        "equal_weight_return"
    )

    if returns_5d is not None:

        daily_returns: list[float] = [
            float(value)
            for value in returns_5d.tolist()
        ]

        return_5d = number(
            math.prod(
                1.0 + r
                for r in daily_returns
            ) - 1.0
        )

    else:
        return_5d = None

    top30_breadth_5d = window_5(
        "top30_breadth"
    )

    industry_breadth_5d = window_5(
        "industry_breadth"
    )

    # -----------------------------------
    # Money flow
    # -----------------------------------

    real_flow_5d = window_5(
        "real_net_money"
    )

    real_strength_5d = window_5(
        "real_flow_strength"
    )

    buyer_power_5d = window_5(
        "real_buy_power"
    )

    # -----------------------------------
    # Volume / turnover behavior
    # -----------------------------------

    turnover_ratio_20d = None

    if len(df) >= 21:

        previous_turnover = (
            df["top30_turnover"]
            .iloc[-21:-1]
        )

        if previous_turnover.notna().all():

            avg_turnover = previous_turnover.mean()

            if avg_turnover > 0:
                turnover_ratio_20d = number(
                    latest["top30_turnover"]
                    / avg_turnover
                )

    # -----------------------------------
    # Data quality
    # -----------------------------------

    coverage_5d = window_5(
        "client_turnover_coverage_ratio"
    )

    # -----------------------------------
    # Final features
    # -----------------------------------

    return {
        "date": latest["date"].date().isoformat(),

        "market": {
            "top30_symbol_count": number(
                latest["top30_symbol_count"]
            ),
            "industry_symbol_count": number(
                latest["industry_symbol_count"]
            ),
            "top30_turnover_share": number(
                latest["top30_turnover_share"]
            ),
        },

        "trend": {
            "equal_weight_return": number(
                latest["equal_weight_return"]
            ),
            "turnover_weighted_return": number(
                latest["turnover_weighted_return"]
            ),
            "daily_top30_return_compounded_5d": (
                return_5d
            ),
            "top30_breadth": number(
                latest["top30_breadth"]
            ),
            "industry_breadth": number(
                latest["industry_breadth"]
            ),
            "top30_breadth_avg_5d": (
                number(top30_breadth_5d.mean())
                if top30_breadth_5d is not None
                else None
            ),
            "industry_breadth_avg_5d": (
                number(industry_breadth_5d.mean())
                if industry_breadth_5d is not None
                else None
            ),
        },

        "money_flow": {
            "real_net_money": number(
                latest["real_net_money"]
            ),
            "real_net_money_5d": (
                number(real_flow_5d.sum())
                if real_flow_5d is not None
                else None
            ),
            "real_flow_strength": number(
                latest["real_flow_strength"]
            ),
            "real_flow_strength_avg_5d": (
                number(real_strength_5d.mean())
                if real_strength_5d is not None
                else None
            ),
            "real_buy_power": number(
                latest["real_buy_power"]
            ),
            "buyer_power_above_one_days_5d": (
                int((buyer_power_5d > 1).sum())
                if buyer_power_5d is not None
                else None
            ),
            "positive_real_flow_days_5d": (
                int((real_flow_5d > 0).sum())
                if real_flow_5d is not None
                else None
            ),
            "legal_net_money": number(
                latest["legal_net_money"]
            ),
        },

        "activity": {
            "top30_turnover": number(
                latest["top30_turnover"]
            ),
            "turnover_ratio_20d": (
                turnover_ratio_20d
            ),
        },

        "data_quality": {
            "available_days": len(df),
            "client_turnover_coverage_ratio": number(
                latest["client_turnover_coverage_ratio"]
            ),
            "min_client_coverage_5d": (
                number(coverage_5d.min())
                if coverage_5d is not None
                else None
            ),
        },
    }