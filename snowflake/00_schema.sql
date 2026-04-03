-- ============================================================================
-- NCLH Revenue Forecasting Engine — Snowflake Schema
-- ============================================================================
-- Run as SYSADMIN or a role with CREATE DATABASE privileges.
-- Sets up three layers: RAW (source), STAGING (transformed), ANALYTICS (output)
-- plus a SIGMA schema containing the views that Sigma Computing connects to.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Databases & Schemas
-- ----------------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS NCLH_FORECASTING
    COMMENT = 'NCLH Revenue Forecasting — bottom-up driver model';

USE DATABASE NCLH_FORECASTING;

CREATE SCHEMA IF NOT EXISTS RAW
    COMMENT = 'Raw ingest from source systems (ERP, booking engine, CRM)';

CREATE SCHEMA IF NOT EXISTS STAGING
    COMMENT = 'Cleaned, typed, and joined tables ready for modelling';

CREATE SCHEMA IF NOT EXISTS ANALYTICS
    COMMENT = 'Forecast outputs, scenario results, and walk-to-target runs';

CREATE SCHEMA IF NOT EXISTS SIGMA
    COMMENT = 'Views exposed directly to Sigma Computing workbooks';


-- ----------------------------------------------------------------------------
-- 2. RAW Layer — source tables (loaded by Fivetran / Airbyte / custom ELT)
-- ----------------------------------------------------------------------------
USE SCHEMA RAW;

CREATE TABLE IF NOT EXISTS SAILINGS_HIST (
    sailing_id              INTEGER       NOT NULL,
    ship_name               VARCHAR(100)  NOT NULL,
    ship_class              VARCHAR(50)   NOT NULL,
    brand                   VARCHAR(50)   NOT NULL,
    trade                   VARCHAR(100)  NOT NULL,
    itinerary_length        INTEGER       NOT NULL,
    departure_date          DATE          NOT NULL,
    lower_berth_capacity    INTEGER       NOT NULL,
    passengers_booked       INTEGER,
    gross_fare_per_diem     FLOAT,
    gross_ticket_revenue    FLOAT,
    discount_rate           FLOAT,
    promo_cost_per_pax      FLOAT,
    commission_rate         FLOAT,
    direct_booking_pct      FLOAT,
    air_inclusive_pct       FLOAT,
    air_cost_per_pax        FLOAT,
    taxes_fees_per_pax      FLOAT,
    net_ticket_revenue      FLOAT,
    _loaded_at              TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT pk_sailings PRIMARY KEY (sailing_id)
)
CLUSTER BY (departure_date, brand, trade)
COMMENT = 'Historical completed sailings with actuals';

CREATE TABLE IF NOT EXISTS SAILINGS_FUTURE (
    sailing_id              INTEGER       NOT NULL,
    ship_name               VARCHAR(100)  NOT NULL,
    ship_class              VARCHAR(50)   NOT NULL,
    brand                   VARCHAR(50)   NOT NULL,
    trade                   VARCHAR(100)  NOT NULL,
    itinerary_length        INTEGER       NOT NULL,
    departure_date          DATE          NOT NULL,
    lower_berth_capacity    INTEGER       NOT NULL,
    _loaded_at              TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT pk_future PRIMARY KEY (sailing_id)
)
CLUSTER BY (departure_date, brand, trade)
COMMENT = 'Forward sailing schedule — capacity known, revenue to be forecast';


-- ----------------------------------------------------------------------------
-- 3. STAGING Layer — cleansed, enriched monthly aggregates
-- ----------------------------------------------------------------------------
USE SCHEMA STAGING;

CREATE OR REPLACE VIEW MONTHLY_DRIVERS AS
/*
  Monthly time series of key revenue drivers aggregated at the level used
  by the bottom-up forecast engine: BRAND × TRADE × SHIP_CLASS.
  This is the input data set for the Snowpark Python stored procedure.
*/
SELECT
    DATE_TRUNC('MONTH', s.departure_date)           AS month,
    s.brand,
    s.trade,
    s.ship_class,
    -- Volume
    COUNT(*)                                         AS n_sailings,
    SUM(s.passengers_booked)                         AS total_passengers,
    SUM(s.lower_berth_capacity)                      AS total_capacity,
    -- Driver metrics (capacity-weighted averages)
    SUM(s.passengers_booked)
        / NULLIF(SUM(s.lower_berth_capacity), 0)     AS load_factor,
    SUM(s.gross_fare_per_diem * s.passengers_booked)
        / NULLIF(SUM(s.passengers_booked), 0)        AS gross_fare_per_diem,
    -- Revenue
    SUM(s.gross_ticket_revenue)                      AS gross_ticket_revenue,
    SUM(s.net_ticket_revenue)                        AS net_ticket_revenue,
    -- Waterfall rates (weighted averages)
    AVG(s.discount_rate)                             AS discount_rate,
    AVG(s.commission_rate)                           AS commission_rate,
    AVG(s.air_inclusive_pct)                         AS air_inclusive_pct,
    AVG(s.air_cost_per_pax)                          AS air_cost_per_pax,
    AVG(s.promo_cost_per_pax)                        AS promo_cost_per_pax,
    AVG(s.taxes_fees_per_pax)                        AS taxes_fees_per_pax,
    AVG(s.direct_booking_pct)                        AS direct_booking_pct
FROM RAW.SAILINGS_HIST s
GROUP BY 1, 2, 3, 4;


CREATE OR REPLACE VIEW DRIVER_DISTRIBUTIONS AS
/*
  Historical percentile distributions for each driver, used to:
    1. Populate the Sigma Input Table slider ranges
    2. Score scenario assumptions (Green / Amber / Red)
    3. Drive the Monte Carlo joint feasibility simulation
*/
SELECT
    brand,
    trade,
    ship_class,
    -- Load factor distribution
    PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY load_factor)       AS lf_p10,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY load_factor)       AS lf_p25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY load_factor)       AS lf_p50,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY load_factor)       AS lf_p75,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY load_factor)       AS lf_p90,
    STDDEV(load_factor)                                              AS lf_std,
    -- Per diem distribution
    PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY gross_fare_per_diem) AS pd_p10,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY gross_fare_per_diem) AS pd_p50,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY gross_fare_per_diem) AS pd_p90,
    STDDEV(gross_fare_per_diem)                                      AS pd_std,
    -- Discount rate distribution
    PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY discount_rate)     AS disc_p10,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY discount_rate)     AS disc_p50,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY discount_rate)     AS disc_p90,
    -- Commission distribution
    PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY commission_rate)   AS comm_p10,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY commission_rate)   AS comm_p50,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY commission_rate)   AS comm_p90
FROM STAGING.MONTHLY_DRIVERS
GROUP BY 1, 2, 3;
