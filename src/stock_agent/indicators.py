import pandas as pd


def _safe_float(value):
    if pd.isna(value):
        return None

    return float(value)


def calculate_technical_indicators(
    prices: list[dict],
) -> dict[str, float | None]:

    df = pd.DataFrame(prices)

    if df.empty:
        raise ValueError("No price data available")

    # بسیار مهم برای rolling indicators
    df = df.sort_values("date").reset_index(drop=True)

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # -------------------------
    # Returns
    # -------------------------

    return_1d = close.pct_change(1) * 100
    return_5d = close.pct_change(5) * 100

    # -------------------------
    # SMA
    # -------------------------

    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()

    # -------------------------
    # EMA
    # -------------------------

    ema_20 = close.ewm(
        span=20,
        adjust=False,
    ).mean()

    # -------------------------
    # RSI 14
    # -------------------------

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    rs = avg_gain / avg_loss

    rsi_14 = 100 - (
        100 / (1 + rs)
    )

    # -------------------------
    # MACD
    # -------------------------

    ema_12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema_26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    macd = ema_12 - ema_26

    macd_signal = macd.ewm(
        span=9,
        adjust=False,
    ).mean()

    macd_histogram = macd - macd_signal

    # -------------------------
    # ATR 14
    # -------------------------

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_14 = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    atr_percent = (
        atr_14 / close
    ) * 100

    # -------------------------
    # Volume
    # -------------------------

    volume_sma_20 = volume.rolling(20).mean()

    volume_ratio_20 = (
        volume / volume_sma_20
    )

    # -------------------------
    # Support / Resistance
    # -------------------------

    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()

    latest = -1

    return {
        "close": _safe_float(close.iloc[latest]),

        "return_1d_pct": _safe_float(
            return_1d.iloc[latest]
        ),

        "return_5d_pct": _safe_float(
            return_5d.iloc[latest]
        ),

        "sma_20": _safe_float(
            sma_20.iloc[latest]
        ),

        "sma_50": _safe_float(
            sma_50.iloc[latest]
        ),

        "ema_20": _safe_float(
            ema_20.iloc[latest]
        ),

        "rsi_14": _safe_float(
            rsi_14.iloc[latest]
        ),

        "macd": _safe_float(
            macd.iloc[latest]
        ),

        "macd_signal": _safe_float(
            macd_signal.iloc[latest]
        ),

        "macd_histogram": _safe_float(
            macd_histogram.iloc[latest]
        ),

        "atr_14": _safe_float(
            atr_14.iloc[latest]
        ),

        "atr_percent": _safe_float(
            atr_percent.iloc[latest]
        ),

        "volume_ratio_20": _safe_float(
            volume_ratio_20.iloc[latest]
        ),

        "high_20": _safe_float(
            high_20.iloc[latest]
        ),

        "low_20": _safe_float(
            low_20.iloc[latest]
        ),
    }