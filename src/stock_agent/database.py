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
    

def get_flow_data(
    symbol: str,
    data_until: str,
    limit: int = 30,
):
    query = text("""
        SELECT
            date,

            real_buy_count,
            real_sell_count,

            real_buy_volume,
            real_sell_volume,
                 
            real_buy_value,
            real_sell_value,

            legal_buy_count,
            legal_sell_count,

            legal_buy_volume,
            legal_sell_volume,
            
            legal_buy_value,
            legal_sell_value
            

        FROM client_type

        WHERE ins_code = (
            SELECT ins_code
            FROM instruments
            WHERE symbol = :symbol
        )

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

        rows = result.mappings().all()

        return [
            dict(row)
            for row in rows
        ]