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
        FROM closing_prices
        WHERE ins_code = (select ins_code from instruments where symbol = :symbol) and adjusted_type = 2
        AND date <= :data_until
        ORDER BY date DESC
        LIMIT :limit
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

        return result.mappings().all()