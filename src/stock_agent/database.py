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
    

def get_codal_data(
    symbol: str,
    data_until: str,
    limit: int = 30,
):
    query = text("""
        SELECT
            l.symbol as symbol,
            l.company_name as company_name,
            l.publish_date_time_gregorian as date,
            f.has_excel as has_excel,
            f.excel_status as excel_status,
            f.path as path,
            l.tracing_no as tracing_no
    
        FROM letters l
            LEFT JOIN files f ON l.tracing_no = f.tracing_no
    
        WHERE 
            l.symbol = :symbol AND
            l.publish_date_time_gregorian <= :data_until
    
        ORDER BY l.publish_date_time_gregorian DESC
    
        LIMIT :limit;
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
    

def get_industry_data(
    symbol: str,
    data_until: str,
    limit: int = 30,
):
    query = text("""
    WITH target_industry AS (
        SELECT industry_code
        FROM instruments
        WHERE symbol = :symbol
        LIMIT 1
    ),

    industry_symbols AS (
        SELECT i.ins_code, i.symbol
        FROM instruments i
        JOIN target_industry t
            ON i.industry_code = t.industry_code
    ),

    price_history AS (
        SELECT
            p.date,
            p.ins_code,
            i.symbol,
            p.close,
            p.volume,

            p.value,
            p.yesterday AS prev_close

        FROM closing_prices p
        JOIN industry_symbols i
            ON p.ins_code = i.ins_code
        WHERE p.date <= :data_until
    ),

    active_stocks AS (
        SELECT *
        FROM price_history
        WHERE volume > 0
        AND close > 0
    ),

    ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY date
                ORDER BY value DESC, ins_code
            ) AS turnover_rank
        FROM active_stocks
    ),

    -- All-industry reference metrics
    industry_all AS (
        SELECT
            date,
            COUNT(*) AS industry_symbol_count,
            SUM(value) AS industry_turnover,

            COUNT(*) FILTER (
                WHERE prev_close > 0
            ) AS return_count,

            COUNT(*) FILTER (
                WHERE prev_close > 0
                AND close > prev_close
            ) AS advancers,

            COUNT(*) FILTER (
                WHERE prev_close > 0
                AND close < prev_close
            ) AS decliners

        FROM active_stocks
        GROUP BY date
    ),

    top30 AS (
        SELECT
            r.*,

            c.ins_code AS client_ins_code,

            c.real_buy_count,
            c.real_sell_count,
            c.real_buy_value,
            c.real_sell_value,

            c.legal_buy_value,
            c.legal_sell_value

        FROM ranked r

        LEFT JOIN client_type c
            ON r.ins_code = c.ins_code
        AND r.date = c.date

        WHERE r.turnover_rank <= 30
    ),

    daily_top30 AS (
        SELECT
            date,

            COUNT(*) AS top30_symbol_count,

            ARRAY_AGG(
                symbol ORDER BY turnover_rank
            ) AS top30_symbols,

            SUM(value) AS top30_turnover,
            SUM(volume) AS top30_volume,

            AVG(
                close::numeric / NULLIF(prev_close, 0) - 1
            ) FILTER (
                WHERE prev_close > 0
            ) AS equal_weight_return,

            SUM(
                CASE WHEN prev_close > 0
                    THEN value *
                        (close::numeric / prev_close - 1)
                    ELSE 0
                END
            ) / NULLIF(
                SUM(
                    CASE WHEN prev_close > 0
                        THEN value
                        ELSE 0
                    END
                ), 0
            ) AS turnover_weighted_return,

            COUNT(*) FILTER (
                WHERE prev_close > 0
            ) AS return_count,

            COUNT(*) FILTER (
                WHERE prev_close > 0
                AND close > prev_close
            ) AS advancers,

            COUNT(*) FILTER (
                WHERE prev_close > 0
                AND close < prev_close
            ) AS decliners,

            SUM(real_buy_value)::numeric AS real_buy_value,
            SUM(real_sell_value)::numeric AS real_sell_value,

            SUM(real_buy_count)::numeric AS real_buy_count,
            SUM(real_sell_count)::numeric AS real_sell_count,

            SUM(legal_buy_value)::numeric AS legal_buy_value,
            SUM(legal_sell_value)::numeric AS legal_sell_value,

            COUNT(client_ins_code) AS client_covered_count,

            SUM(
                CASE
                    WHEN client_ins_code IS NOT NULL
                    THEN value
                    ELSE 0
                END
            ) AS client_covered_turnover

        FROM top30
        GROUP BY date
    )

    SELECT
        d.date,

        d.top30_symbol_count,
        d.top30_symbols,

        a.industry_symbol_count,

        -- Liquidity concentration
        d.top30_turnover,
        d.top30_volume,

        d.top30_turnover /
            NULLIF(a.industry_turnover, 0)
            AS top30_turnover_share,

        -- Performance
        d.equal_weight_return,
        d.turnover_weighted_return,

        -- Market breadth
        (
            d.advancers - d.decliners
        )::numeric /
            NULLIF(d.return_count, 0)
            AS top30_breadth,

        (
            a.advancers - a.decliners
        )::numeric /
            NULLIF(a.return_count, 0)
            AS industry_breadth,

        -- Real money flow
        d.real_buy_value - d.real_sell_value
            AS real_net_money,

        (
            d.real_buy_value - d.real_sell_value
        ) / NULLIF(
            d.real_buy_value + d.real_sell_value, 0
        ) AS real_flow_strength,

        -- Real buyer power
        (
            d.real_buy_value /
                NULLIF(d.real_buy_count, 0)
        ) / NULLIF(
            d.real_sell_value /
                NULLIF(d.real_sell_count, 0),
            0
        ) AS real_buy_power,

        -- Legal flow
        d.legal_buy_value - d.legal_sell_value
            AS legal_net_money,

        -- Data quality
        d.client_covered_count::numeric /
            NULLIF(d.top30_symbol_count, 0)
            AS client_coverage_ratio,

        d.client_covered_turnover /
            NULLIF(d.top30_turnover, 0)
            AS client_turnover_coverage_ratio

    FROM daily_top30 d
    JOIN industry_all a USING (date)

    ORDER BY d.date DESC
    LIMIT :limit;
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