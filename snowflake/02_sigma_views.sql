-- ============================================================================
-- NCLH Revenue Forecasting Engine — Sigma Computing Views
-- ============================================================================
-- These views are what Sigma workbooks connect to directly.
-- Sigma users see clean, business-friendly column names and pre-computed
-- waterfall metrics. Walk-to-target adjustments are applied server-side
-- via Sigma Input Tables that write back to ANALYTICS.WALK_TO_TARGET_RUNS.
-- ============================================================================

USE DATABASE NCLH_FORECASTING;
USE SCHEMA SIGMA;


-- ----------------------------------------------------------------------------
-- V_BASELINE_FORECAST  — primary view for the Sigma Forecast Dashboard page
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW SIGMA.V_BASELINE_FORECAST AS
/*
  One row per future sailing × forecast run. Joins the scheduled sailings
  (capacity known) against the latest forecast run to derive revenue metrics.
  Sigma connects to this view to power all Forecast Dashboard elements.
*/
WITH latest_run AS (
    -- Always surface only the most recent forecast run
    SELECT MAX(run_id) AS run_id
    FROM   ANALYTICS.FORECAST_RESULTS
),
drivers AS (
    SELECT f.*
    FROM   ANALYTICS.FORECAST_RESULTS f
    JOIN   latest_run                  lr ON f.run_id = lr.run_id
)
SELECT
    s.sailing_id,
    s.ship_name,
    s.ship_class,
    s.brand,
    s.trade,
    s.itinerary_length,
    s.departure_date,
    DATE_TRUNC('MONTH', s.departure_date)            AS forecast_month,
    QUARTER(s.departure_date)                        AS forecast_quarter,
    YEAR(s.departure_date)                           AS forecast_year,
    s.lower_berth_capacity,

    -- Forecasted drivers (with intervals)
    ROUND(d.load_factor_fcst, 4)                     AS load_factor,
    ROUND(d.load_factor_lo80, 4)                     AS load_factor_lo80,
    ROUND(d.load_factor_hi80, 4)                     AS load_factor_hi80,
    ROUND(d.gross_per_diem_fcst, 2)                  AS gross_fare_per_diem,
    ROUND(d.gross_per_diem_lo80, 2)                  AS gross_per_diem_lo80,
    ROUND(d.gross_per_diem_hi80, 2)                  AS gross_per_diem_hi80,

    -- Baseline waterfall rates
    d.discount_rate_base                             AS discount_rate,
    d.commission_rate_base                           AS commission_rate,
    d.air_inclusive_pct_base                         AS air_inclusive_pct,
    d.air_cost_per_pax_base                          AS air_cost_per_pax,
    d.promo_cost_per_pax_base                        AS promo_cost_per_pax,
    d.taxes_fees_per_pax_base                        AS taxes_fees_per_pax,
    d.direct_booking_pct_base                        AS direct_booking_pct,

    -- Derived revenue waterfall (computed in SQL, no Sigma formula needed)
    ROUND(s.lower_berth_capacity * d.load_factor_fcst)
                                                     AS passengers_forecast,
    ROUND(s.lower_berth_capacity * d.load_factor_fcst
          * d.gross_per_diem_fcst * s.itinerary_length)
                                                     AS gross_ticket_revenue,
    -- Deductions
    ROUND(s.lower_berth_capacity * d.load_factor_fcst
          * d.gross_per_diem_fcst * s.itinerary_length
          * d.discount_rate_base)                    AS discount_amount,
    ROUND(s.lower_berth_capacity * d.load_factor_fcst
          * d.promo_cost_per_pax_base)               AS promo_cost_total,
    ROUND(s.lower_berth_capacity * d.load_factor_fcst
          * d.gross_per_diem_fcst * s.itinerary_length
          * (1 - d.discount_rate_base)
          * d.commission_rate_base
          * (1 - d.direct_booking_pct_base))        AS ta_commission,
    ROUND(s.lower_berth_capacity * d.load_factor_fcst
          * d.air_inclusive_pct_base * d.air_cost_per_pax_base)
                                                     AS air_cost_total,
    ROUND(s.lower_berth_capacity * d.load_factor_fcst
          * d.taxes_fees_per_pax_base)               AS taxes_total,
    -- Net Ticket Revenue
    ROUND(
        s.lower_berth_capacity * d.load_factor_fcst
        * d.gross_per_diem_fcst * s.itinerary_length
        - (s.lower_berth_capacity * d.load_factor_fcst
           * d.gross_per_diem_fcst * s.itinerary_length * d.discount_rate_base)
        - (s.lower_berth_capacity * d.load_factor_fcst * d.promo_cost_per_pax_base)
        - (s.lower_berth_capacity * d.load_factor_fcst
           * d.gross_per_diem_fcst * s.itinerary_length
           * (1 - d.discount_rate_base) * d.commission_rate_base
           * (1 - d.direct_booking_pct_base))
        - (s.lower_berth_capacity * d.load_factor_fcst
           * d.air_inclusive_pct_base * d.air_cost_per_pax_base)
        - (s.lower_berth_capacity * d.load_factor_fcst * d.taxes_fees_per_pax_base)
    )                                                AS net_ticket_revenue,

    d.model_lf,
    d.model_pd,
    d.run_id                                         AS forecast_run_id

FROM RAW.SAILINGS_FUTURE   s
JOIN drivers               d
  ON  s.brand      = d.brand
  AND s.trade      = d.trade
  AND s.ship_class = d.ship_class
  AND DATE_TRUNC('MONTH', s.departure_date) = d.forecast_date;


-- ----------------------------------------------------------------------------
-- V_WATERFALL_SUMMARY  — monthly waterfall for the Sigma Waterfall chart
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW SIGMA.V_WATERFALL_SUMMARY AS
SELECT
    forecast_month,
    brand,
    trade,
    SUM(gross_ticket_revenue)   AS gross_ticket_revenue,
    SUM(discount_amount)        AS discount_amount,
    SUM(promo_cost_total)       AS promo_cost_total,
    SUM(ta_commission)          AS ta_commission,
    SUM(air_cost_total)         AS air_cost_total,
    SUM(taxes_total)            AS taxes_total,
    SUM(net_ticket_revenue)     AS net_ticket_revenue
FROM SIGMA.V_BASELINE_FORECAST
GROUP BY 1, 2, 3;


-- ----------------------------------------------------------------------------
-- V_DRIVER_DISTRIBUTIONS — powers the P10-P90 range bands in Sigma
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW SIGMA.V_DRIVER_DISTRIBUTIONS AS
SELECT * FROM STAGING.DRIVER_DISTRIBUTIONS;


-- ----------------------------------------------------------------------------
-- WALK_TO_TARGET_RUNS  — writeback table for Sigma Input Tables
-- ----------------------------------------------------------------------------
-- When a user adjusts drivers in the Sigma "Walk to Target" page and clicks
-- "Save Scenario", Sigma writes one row here via its Input Table feature.
-- A post-action SQL block then reruns the waterfall for the saved scenario.
CREATE TABLE IF NOT EXISTS ANALYTICS.WALK_TO_TARGET_RUNS (
    scenario_id         VARCHAR(50)   NOT NULL DEFAULT UUID_STRING(),
    scenario_name       VARCHAR(200),
    created_by          VARCHAR(200),
    created_at          TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    target_ntr_m        FLOAT,        -- target Net Ticket Revenue in $M
    -- Driver overrides (NULL = use baseline)
    load_factor_override        FLOAT,
    gross_per_diem_override     FLOAT,
    discount_rate_override      FLOAT,
    commission_rate_override    FLOAT,
    air_inclusive_pct_override  FLOAT,
    promo_cost_override         FLOAT,
    -- Computed results (populated by post-action stored procedure)
    scenario_ntr_m      FLOAT,
    feasibility_pct     FLOAT,       -- Monte Carlo joint probability (0-100)
    notes               VARCHAR(2000),
    CONSTRAINT pk_scenarios PRIMARY KEY (scenario_id)
);


-- ----------------------------------------------------------------------------
-- V_SCENARIO_COMPARISON  — powers the Sigma Scenario Comparison page
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW SIGMA.V_SCENARIO_COMPARISON AS
WITH base AS (
    SELECT SUM(net_ticket_revenue) / 1e6 AS baseline_ntr_m
    FROM   SIGMA.V_BASELINE_FORECAST
)
SELECT
    w.scenario_id,
    w.scenario_name,
    w.created_by,
    w.created_at,
    w.target_ntr_m,
    w.scenario_ntr_m,
    w.feasibility_pct,
    b.baseline_ntr_m,
    w.scenario_ntr_m - b.baseline_ntr_m        AS delta_vs_baseline_m,
    ROUND((w.scenario_ntr_m - b.baseline_ntr_m)
          / NULLIF(b.baseline_ntr_m, 0) * 100, 1) AS pct_vs_baseline,
    -- Driver assumption columns
    w.load_factor_override,
    w.gross_per_diem_override,
    w.discount_rate_override,
    w.commission_rate_override,
    w.air_inclusive_pct_override,
    w.promo_cost_override,
    w.notes
FROM ANALYTICS.WALK_TO_TARGET_RUNS w
CROSS JOIN base b
ORDER BY w.created_at DESC;
