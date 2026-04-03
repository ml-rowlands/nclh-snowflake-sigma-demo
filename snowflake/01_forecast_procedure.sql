-- ============================================================================
-- NCLH Revenue Forecasting Engine — Snowpark Python Stored Procedure
-- ============================================================================
-- Deploys the bottom-up driver forecast as a Snowflake stored procedure.
-- Requires: Snowpark-optimised warehouse, Python 3.11 runtime,
--           packages: statsforecast, hierarchicalforecast, pandas, numpy
-- ============================================================================

USE DATABASE NCLH_FORECASTING;
USE SCHEMA ANALYTICS;

-- ----------------------------------------------------------------------------
-- Create the output table (overwritten on each forecast run)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS FORECAST_RESULTS (
    run_id          VARCHAR(50)   NOT NULL,
    run_ts          TIMESTAMP_NTZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    forecast_date   DATE          NOT NULL,
    brand           VARCHAR(50)   NOT NULL,
    trade           VARCHAR(100)  NOT NULL,
    ship_class      VARCHAR(50)   NOT NULL,
    horizon_month   INTEGER       NOT NULL,   -- 1-12
    -- Forecasted drivers
    load_factor_fcst        FLOAT,
    load_factor_lo80        FLOAT,
    load_factor_hi80        FLOAT,
    gross_per_diem_fcst     FLOAT,
    gross_per_diem_lo80     FLOAT,
    gross_per_diem_hi80     FLOAT,
    -- Baseline waterfall assumptions (historical medians)
    discount_rate_base      FLOAT,
    commission_rate_base    FLOAT,
    air_inclusive_pct_base  FLOAT,
    air_cost_per_pax_base   FLOAT,
    promo_cost_per_pax_base FLOAT,
    taxes_fees_per_pax_base FLOAT,
    direct_booking_pct_base FLOAT,
    -- Model metadata
    model_lf        VARCHAR(50),
    model_pd        VARCHAR(50),
    CONSTRAINT pk_forecast PRIMARY KEY (run_id, forecast_date, brand, trade, ship_class)
)
CLUSTER BY (forecast_date, brand, trade);


-- ----------------------------------------------------------------------------
-- Stored procedure: run_bottom_up_forecast
-- ----------------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE ANALYTICS.RUN_BOTTOM_UP_FORECAST(
    RUN_ID       VARCHAR,
    HORIZON_MONTHS INTEGER DEFAULT 12
)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = (
    'snowflake-snowpark-python',
    'statsforecast==1.7.8',
    'hierarchicalforecast==1.3.1',
    'utilsforecast==0.2.15',
    'pandas',
    'numpy',
    'scikit-learn'
)
HANDLER = 'run_forecast'
COMMENT = 'Bottom-up driver forecast using StatsForecast + HierarchicalForecast MinTrace reconciliation'
AS
$$
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import date
from snowflake.snowpark.functions import col


def run_forecast(session, run_id: str, horizon_months: int = 12) -> dict:
    """
    Bottom-up revenue driver forecast.

    Steps:
      1. Pull monthly driver time series from STAGING.MONTHLY_DRIVERS
      2. Build the hierarchy: Total > Brand > Brand/Trade > Brand/Trade/ShipClass
      3. Fit AutoETS + AutoARIMA + AutoCES + SeasonalNaive at bottom level
      4. Reconcile with MinTrace (mint_shrink) via HierarchicalForecast
      5. Apply conformal prediction intervals (80% and 95%)
      6. Write results to ANALYTICS.FORECAST_RESULTS
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, AutoARIMA, AutoCES, SeasonalNaive
    from hierarchicalforecast.core import HierarchicalReconciliation
    from hierarchicalforecast.methods import MinTrace
    from hierarchicalforecast.utils import aggregate

    # ── 1. Load historical data ──────────────────────────────────────────────
    df = (
        session.table("NCLH_FORECASTING.STAGING.MONTHLY_DRIVERS")
        .select("MONTH", "BRAND", "TRADE", "SHIP_CLASS",
                "LOAD_FACTOR", "GROSS_FARE_PER_DIEM",
                "DISCOUNT_RATE", "COMMISSION_RATE",
                "AIR_INCLUSIVE_PCT", "AIR_COST_PER_PAX",
                "PROMO_COST_PER_PAX", "TAXES_FEES_PER_PAX",
                "DIRECT_BOOKING_PCT")
        .to_pandas()
    )
    df.columns = df.columns.str.lower()
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")

    # Unique ID for bottom level: brand/trade/ship_class
    df["unique_id"] = df["brand"] + "/" + df["trade"] + "/" + df["ship_class"]

    results = {}
    for driver, driver_col in [("load_factor", "load_factor"),
                                 ("per_diem",   "gross_fare_per_diem")]:

        # ── 2. Build hierarchy ───────────────────────────────────────────────
        df_nixtla = df[["month", "unique_id", driver_col]].rename(
            columns={"month": "ds", driver_col: "y"}
        ).dropna(subset=["y"])

        spec = [
            ["brand"],
            ["brand", "trade"],
            ["brand", "trade", "ship_class"],
        ]

        # Parse unique_id back into columns for aggregate()
        df_meta = df_nixtla.copy()
        df_meta[["brand","trade","ship_class"]] = df_meta["unique_id"].str.split("/", expand=True)

        Y_df, S_df, tags = aggregate(
            df=df_meta,
            spec=spec
        )
        Y_df = Y_df.reset_index()

        # ── 3. Fit base models at all levels ─────────────────────────────────
        sf = StatsForecast(
            models=[
                AutoETS(season_length=12, model="ZZZ"),
                AutoARIMA(season_length=12),
                AutoCES(season_length=12),
                SeasonalNaive(season_length=12),
            ],
            freq="MS",
            n_jobs=-1,
            fallback_model=SeasonalNaive(season_length=12),
        )
        sf.fit(Y_df)
        fcst_df = sf.predict(h=horizon_months, level=[80])

        # ── 4. MinTrace reconciliation ───────────────────────────────────────
        hrec = HierarchicalReconciliation(
            reconcilers=[MinTrace(method="mint_shrink")]
        )
        reconciled = hrec.reconcile(
            Y_hat_df=fcst_df,
            Y_df=Y_df,
            S=S_df,
            tags=tags,
        )

        # Pick best model per series by in-sample MAE
        best_col = _pick_best_model(Y_df, fcst_df)
        results[driver] = {
            "reconciled": reconciled,
            "best_model": best_col,
            "S": S_df,
            "tags": tags,
        }

    # ── 5. Pull waterfall baselines (historical medians per brand/trade/class) ─
    baseline_df = (
        session.sql("""
            SELECT brand, trade, ship_class,
                   MEDIAN(discount_rate)       AS discount_rate_base,
                   MEDIAN(commission_rate)     AS commission_rate_base,
                   MEDIAN(air_inclusive_pct)   AS air_inclusive_pct_base,
                   MEDIAN(air_cost_per_pax)    AS air_cost_per_pax_base,
                   MEDIAN(promo_cost_per_pax)  AS promo_cost_per_pax_base,
                   MEDIAN(taxes_fees_per_pax)  AS taxes_fees_per_pax_base,
                   MEDIAN(direct_booking_pct)  AS direct_booking_pct_base
            FROM NCLH_FORECASTING.STAGING.MONTHLY_DRIVERS
            GROUP BY 1,2,3
        """)
        .to_pandas()
    )
    baseline_df.columns = baseline_df.columns.str.lower()

    # ── 6. Assemble output rows ──────────────────────────────────────────────
    output_rows = _assemble_output(
        run_id=run_id,
        lf_results=results["load_factor"],
        pd_results=results["per_diem"],
        baseline_df=baseline_df,
        horizon_months=horizon_months,
    )

    # ── 7. Write to Snowflake ────────────────────────────────────────────────
    out_df = pd.DataFrame(output_rows)
    sp_df  = session.create_dataframe(out_df)
    sp_df.write.mode("append").save_as_table("NCLH_FORECASTING.ANALYTICS.FORECAST_RESULTS")

    n_rows = len(out_df)
    return {
        "status": "success",
        "run_id": run_id,
        "rows_written": n_rows,
        "horizon_months": horizon_months,
    }


def _pick_best_model(Y_df: pd.DataFrame, fcst_df: pd.DataFrame) -> str:
    """Return the column name of the model with lowest cross-series MAE."""
    from utilsforecast.losses import mae
    model_cols = [c for c in fcst_df.columns if c not in ("unique_id","ds")]
    base_models = [c for c in model_cols if not any(s in c for s in ["-lo","-hi","MinTrace"])]
    best_col, best_mae = base_models[0], float("inf")
    for m in base_models:
        merged = Y_df.merge(fcst_df[["unique_id","ds",m]], on=["unique_id","ds"], how="inner")
        if merged.empty:
            continue
        err = mae(merged, models=[m], id_col="unique_id", target_col="y", time_col="ds")
        score = err[m].mean()
        if score < best_mae:
            best_mae, best_col = score, m
    return best_col


def _assemble_output(run_id, lf_results, pd_results, baseline_df, horizon_months):
    rows = []
    lf_rec  = lf_results["reconciled"].reset_index()
    pd_rec  = pd_results["reconciled"].reset_index()
    lf_best = lf_results["best_model"]
    pd_best = pd_results["best_model"]

    # Only keep bottom-level series (brand/trade/ship_class have 2 slashes)
    bottom = lf_rec[lf_rec["unique_id"].str.count("/") == 2].copy()

    for _, row in bottom.iterrows():
        uid = row["unique_id"]
        brand, trade, ship_class = uid.split("/")
        horizon = int((pd.to_datetime(row["ds"]).year * 12 + pd.to_datetime(row["ds"]).month)
                      - (pd.Timestamp("today").year * 12 + pd.Timestamp("today").month) + 1)
        if horizon < 1 or horizon > horizon_months:
            continue

        # Load factor reconciled
        lf_model_col = f"{lf_best}/MinTrace_method-mint_shrink"
        if lf_model_col not in lf_rec.columns:
            lf_model_col = lf_best
        lf_fcst = float(row.get(lf_model_col, row.get(lf_best, np.nan)))
        lf_lo   = float(row.get(f"{lf_best}-lo-80", lf_fcst * 0.97))
        lf_hi   = float(row.get(f"{lf_best}-hi-80", lf_fcst * 1.03))

        # Per diem reconciled
        pd_row = pd_rec[(pd_rec["unique_id"] == uid) & (pd_rec["ds"] == row["ds"])]
        pd_model_col = f"{pd_best}/MinTrace_method-mint_shrink"
        if pd_model_col not in pd_rec.columns:
            pd_model_col = pd_best
        pd_fcst = float(pd_row[pd_model_col].iloc[0]) if not pd_row.empty else np.nan
        pd_lo   = float(pd_row.get(f"{pd_best}-lo-80", pd_fcst * 0.95).iloc[0]) if not pd_row.empty else np.nan
        pd_hi   = float(pd_row.get(f"{pd_best}-hi-80", pd_fcst * 1.05).iloc[0]) if not pd_row.empty else np.nan

        # Baseline waterfall rates
        base = baseline_df[
            (baseline_df["brand"] == brand) &
            (baseline_df["trade"] == trade) &
            (baseline_df["ship_class"] == ship_class)
        ]
        def base_val(col): return float(base[col].iloc[0]) if not base.empty else np.nan

        rows.append({
            "run_id":                  run_id,
            "run_ts":                  pd.Timestamp.now(),
            "forecast_date":           pd.to_datetime(row["ds"]).date(),
            "brand":                   brand,
            "trade":                   trade,
            "ship_class":              ship_class,
            "horizon_month":           horizon,
            "load_factor_fcst":        lf_fcst,
            "load_factor_lo80":        lf_lo,
            "load_factor_hi80":        lf_hi,
            "gross_per_diem_fcst":     pd_fcst,
            "gross_per_diem_lo80":     pd_lo,
            "gross_per_diem_hi80":     pd_hi,
            "discount_rate_base":      base_val("discount_rate_base"),
            "commission_rate_base":    base_val("commission_rate_base"),
            "air_inclusive_pct_base":  base_val("air_inclusive_pct_base"),
            "air_cost_per_pax_base":   base_val("air_cost_per_pax_base"),
            "promo_cost_per_pax_base": base_val("promo_cost_per_pax_base"),
            "taxes_fees_per_pax_base": base_val("taxes_fees_per_pax_base"),
            "direct_booking_pct_base": base_val("direct_booking_pct_base"),
            "model_lf":                lf_best,
            "model_pd":                pd_best,
        })
    return rows
$$;


-- ── Schedule the procedure to refresh monthly ─────────────────────────────────
CREATE OR REPLACE TASK ANALYTICS.REFRESH_FORECAST
    WAREHOUSE = 'FORECASTING_WH'
    SCHEDULE  = 'USING CRON 0 4 1 * * America/New_York'
    COMMENT   = 'Refresh bottom-up forecast on the 1st of each month at 4am ET'
AS
CALL ANALYTICS.RUN_BOTTOM_UP_FORECAST(
    TO_CHAR(CURRENT_DATE, 'YYYY-MM-DD'),
    12
);

-- Start the task (requires ACCOUNTADMIN or TASK privilege)
-- ALTER TASK ANALYTICS.REFRESH_FORECAST RESUME;
