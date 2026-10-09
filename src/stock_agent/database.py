from sqlalchemy import create_engine, text

from .config import settings


engine = create_engine(settings.database_url)


def get_price_data(
    symbol: str,
    data_until: str,
    limit: int = 200,
):
    query = text("""
        SELECT
            date,
            open,
            high,
            low,
            close,
            volume
        FROM (
            SELECT
                date,
                open,
                high,
                low,
                close,
                volume
            FROM closing_prices
            WHERE ins_code = (
                SELECT ins_code
                FROM instruments
                WHERE symbol = :symbol
            )
            AND adjusted_type = 2
            AND date <= :data_until
            ORDER BY date DESC
            LIMIT :limit
        ) AS recent_prices
        ORDER BY date ASC
    """)

    with engine.connect() as conn:
        result = conn.execute(
            query,
            {
                "symbol": symbol,
                "data_until": data_until,
                "limit": limit,
            },
        )

        return [
            dict(row)
            for row in result.mappings().all()
        ]