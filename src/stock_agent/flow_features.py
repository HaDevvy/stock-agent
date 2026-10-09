import math

import pandas as pd

def safe_float(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)

def calculate_flow_features(
    rows: list[dict],
) -> dict:
    if not rows:
        raise ValueError(
            "No flow data available."
        )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    numeric_columns = [
        "real_buy_count",
        "real_sell_count",
        "real_buy_volume",
        "real_sell_volume",
        "legal_buy_count",
        "legal_sell_count",
        "legal_buy_volume",
        "legal_sell_volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )
    
    df["volume"] = df["legal_buy_volume"] + df["real_buy_volume"]
    df["value"] = df["legal_buy_value"] + df["real_buy_value"]

    #
    # سرانه خرید و فروش حقیقی
    #

    df["real_buy_per_capita"] = (
        df["real_buy_volume"]
        / df["real_buy_count"].replace(0, pd.NA)
    )

    df["real_sell_per_capita"] = (
        df["real_sell_volume"]
        / df["real_sell_count"].replace(0, pd.NA)
    )

    #
    # قدرت خریدار حقیقی
    #

    df["real_power"] = (
        df["real_buy_per_capita"]
        / df["real_sell_per_capita"].replace(
            0,
            pd.NA,
        )
    )

    #
    # خالص جریان حقیقی
    #

    df["real_net_volume"] = (
        df["real_buy_volume"]
        - df["real_sell_volume"]
    )

    #
    # جریان حقوقی
    #

    df["legal_net_volume"] = (
        df["legal_buy_volume"]
        - df["legal_sell_volume"]
    )

    #
    # نسبت حجم روز به میانگین 20 روزه
    #

    df["volume_sma_20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio_20"] = (
        df["volume"]
        / df["volume_sma_20"]
    )

    #
    # پایداری جریان حقیقی
    #
    # درصد روزهای مثبت در 5 روز اخیر
    #

    df["real_positive_flow"] = (
        df["real_net_volume"] > 0
    ).astype(int)

    recent_5 = df.tail(5)

    if len(recent_5) > 0:
        positive_flow_ratio_5d = (
            recent_5["real_positive_flow"].mean()
        )
    else:
        positive_flow_ratio_5d = None

    #
    # متوسط قدرت خریدار در 5 روز
    #

    real_power_5d = (
        df["real_power"]
        .tail(5)
        .mean()
    )

    #
    # خالص جریان 5 روز
    #

    real_net_volume_5d = (
        df["real_net_volume"]
        .tail(5)
        .sum()
    )

    legal_net_volume_5d = (
        df["legal_net_volume"]
        .tail(5)
        .sum()
    )

    latest = df.iloc[-1]

    return {
        "real_buy_per_capita": safe_float(
            latest["real_buy_per_capita"]
        ),

        "real_sell_per_capita": safe_float(
            latest["real_sell_per_capita"]
        ),

        "real_power": safe_float(
            latest["real_power"]
        ),

        "real_power_5d": safe_float(
            real_power_5d
        ),

        "real_net_volume": safe_float(
            latest["real_net_volume"]
        ),

        "real_net_volume_5d": safe_float(
            real_net_volume_5d
        ),

        "legal_net_volume": safe_float(
            latest["legal_net_volume"]
        ),

        "legal_net_volume_5d": safe_float(
            legal_net_volume_5d
        ),

        "volume_ratio_20": safe_float(
            latest["volume_ratio_20"]
        ),

        "positive_flow_ratio_5d": safe_float(
            positive_flow_ratio_5d
        ),

        "volume": safe_float(
            latest["volume"]
        ),

        "value": safe_float(
            latest["value"]
        ),
    }