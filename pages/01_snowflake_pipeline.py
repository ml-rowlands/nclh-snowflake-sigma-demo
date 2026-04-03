"""
Page 1: Snowflake Pipeline
Shows the production SQL/Snowpark code, table schemas, and query performance
context — the "under the hood" view for technical stakeholders.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

SNOW_BLUE  = "#29B5E8"
SNOW_DARK  = "#0369A1"
NAVY       = "#0B2545"
BORDER     = "#E3E5EC"
LIGHT      = "#F5F6FA"

st.markdown(f"""
<style>
[data-testid="stSidebar"] {{ background:{NAVY} !important; }}
[data-testid="stSidebar"] * {{ color:#F0F4F8 !important; }}
.main .block-container {{ padding-top:1.5rem; max-width:1400px; }}
.snow-card {{
    background:white; border:1px solid {BORDER}; border-radius:10px;
    padding:18px 22px; margin-bottom:12px;
    box-shadow:0 2px 6px rgba(0,0,0,0.05);
}}
.snow-header {{
    background:linear-gradient(90deg,{SNOW_DARK} 0%,{SNOW_BLUE} 100%);
    color:white; border-radius:8px; padding:12px 18px; margin-bottom:16px;
}}
.schema-tag {{
    display:inline-block; background:{LIGHT}; border:1px solid {BORDER};
    border-radius:4px; padding:2px 8px; font-size:0.78rem;
    font-family:monospace; color:{SNOW_DARK}; margin-right:4px;
}}
.disclaimer {{font-size:0.7rem;color:#8E9BAB;text-align:center;
    border-top:1px solid {BORDER};padding-top:8px;margin-top:32px;}}
</style>
""", unsafe_allow_html=True)

# ── Load cached data ──────────────────────────────────────────────────────────
hist_df  = st.session_state.get("hist_df")
future_df = st.session_state.get("future_df")
fcst_df  = st.session_state.get("fcst_df")

if hist_df is None:
    st.info("Loading data…")
    st.stop()

st.markdown("""
<div class="snow-header">
  <b style="font-size:1.1rem;">❄️ Snowflake Pipeline</b>
  <span style="font-size:0.8rem;opacity:0.85;margin-left:12px;">
    Schema · Stored Procedure · Scheduling · Query Performance
  </span>
</div>
""", unsafe_allow_html=True)

# ── Schema overview ───────────────────────────────────────────────────────────
st.markdown("### Database Structure")

col1, col2, col3, col4 = st.columns(4)
schemas = [
    ("RAW",       SNOW_BLUE,  "Source ingest", ["SAILINGS_HIST", "SAILINGS_FUTURE"]),
    ("STAGING",   "#0EA5E9",  "Cleaned views", ["MONTHLY_DRIVERS", "DRIVER_DISTRIBUTIONS"]),
    ("ANALYTICS", "#0369A1",  "Model outputs", ["FORECAST_RESULTS", "WALK_TO_TARGET_RUNS"]),
    ("SIGMA",     "#0B72E7",  "Sigma views",   ["V_BASELINE_FORECAST","V_WATERFALL_SUMMARY","V_SCENARIO_COMPARISON"]),
]
for col, (schema, color, desc, tables) in zip([col1,col2,col3,col4], schemas):
    with col:
        table_html = "".join(f'<div style="font-family:monospace;font-size:0.78rem;padding:3px 0;border-bottom:1px solid {BORDER};color:#374151;">{t}</div>' for t in tables)
        st.markdown(f"""
        <div style="background:white;border:1px solid {BORDER};border-radius:10px;overflow:hidden;">
          <div style="background:{color};color:white;padding:10px 14px;">
            <b style="font-size:0.9rem;">NCLH_FORECASTING.{schema}</b><br>
            <span style="font-size:0.75rem;opacity:0.85;">{desc}</span>
          </div>
          <div style="padding:10px 14px;">{table_html}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── SQL file viewer ───────────────────────────────────────────────────────────
st.markdown("### Snowflake SQL Source")

sql_files = {
    "00_schema.sql — DDL & staging views": "snowflake/00_schema.sql",
    "01_forecast_procedure.sql — Snowpark Python SP": "snowflake/01_forecast_procedure.sql",
    "02_sigma_views.sql — Sigma output views": "snowflake/02_sigma_views.sql",
}

selected_file = st.selectbox("Select file to view:", list(sql_files.keys()))
fpath = os.path.join(os.path.dirname(__file__), "..", sql_files[selected_file])
try:
    with open(fpath) as f:
        sql_content = f.read()
    st.code(sql_content, language="sql", line_numbers=True)
except FileNotFoundError:
    st.warning(f"File not found: {fpath}")

st.markdown("---")

# ── Monthly driver time series (what the SP trains on) ────────────────────────
st.markdown("### STAGING.MONTHLY_DRIVERS — Sample Data")
st.caption("This is the input dataset the Snowpark Python stored procedure reads to fit forecasting models.")

monthly = (
    hist_df
    .assign(month=lambda d: d["departure_date"].dt.to_period("M").dt.to_timestamp())
    .groupby(["month","brand","trade"])
    .agg(
        load_factor        = ("load_factor",         "mean"),
        gross_fare_per_diem= ("gross_fare_per_diem",  "mean"),
        net_ticket_revenue = ("net_ticket_revenue",   "sum"),
        n_sailings         = ("sailing_id",           "count"),
    )
    .reset_index()
)

tab_ts, tab_tbl = st.tabs(["📈 Time Series View", "📋 Table View"])

with tab_ts:
    metric_choice = st.selectbox(
        "Driver to plot:",
        ["load_factor","gross_fare_per_diem","net_ticket_revenue"],
        format_func=lambda x: {
            "load_factor":"Load Factor","gross_fare_per_diem":"Gross Per Diem ($)",
            "net_ticket_revenue":"Net Ticket Revenue ($)"
        }[x]
    )
    brand_filter = st.multiselect("Brand:", ["NCL","Oceania","Regent"], default=["NCL","Oceania","Regent"])
    plot_data = monthly[monthly["brand"].isin(brand_filter)]

    fig = go.Figure()
    colors = {"NCL": SNOW_BLUE, "Oceania": "#0369A1", "Regent": "#7C3AED"}
    for brand in brand_filter:
        bd = plot_data[plot_data["brand"]==brand].groupby("month")[metric_choice].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=bd["month"], y=bd[metric_choice],
            name=brand, mode="lines+markers",
            line=dict(color=colors.get(brand, "#666"), width=2),
            marker=dict(size=5),
        ))
    label_map = {"load_factor":"Load Factor","gross_fare_per_diem":"Gross Per Diem ($)","net_ticket_revenue":"Net Ticket Revenue"}
    fig.update_layout(
        height=340, margin=dict(l=0,r=0,t=20,b=0),
        legend=dict(orientation="h", y=-0.15),
        xaxis_title="Month", yaxis_title=label_map[metric_choice],
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#F0F2F5"),
        yaxis=dict(showgrid=True, gridcolor="#F0F2F5"),
    )
    if metric_choice == "load_factor":
        fig.update_yaxes(tickformat=".1%")
    elif metric_choice in ["gross_fare_per_diem","net_ticket_revenue"]:
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

with tab_tbl:
    sample = monthly.sort_values("month",ascending=False).head(100).copy()
    sample["load_factor"]         = sample["load_factor"].apply(lambda x: f"{x*100:.1f}%")
    sample["gross_fare_per_diem"] = sample["gross_fare_per_diem"].apply(lambda x: f"${x:,.0f}")
    sample["net_ticket_revenue"]  = sample["net_ticket_revenue"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(sample, use_container_width=True, height=320)
    st.caption(f"View: `NCLH_FORECASTING.STAGING.MONTHLY_DRIVERS` · {len(monthly):,} rows total")

st.markdown("---")

# ── Forecast model performance ────────────────────────────────────────────────
st.markdown("### Forecast Model — Mock Accuracy Comparison")
st.caption("Illustrative cross-validation MAE across models, as computed by the stored procedure's model selection step.")

np.random.seed(7)
model_names = ["AutoETS","AutoARIMA","AutoCES","SeasonalNaive"]
trades = ["Caribbean","Mediterranean","Alaska","Northern Europe","Bermuda"]
metrics = []
for trade in trades:
    base_mae = np.random.uniform(0.010, 0.025)
    metrics.append({
        "Trade": trade,
        "AutoETS":       round(base_mae * np.random.uniform(0.90, 1.10), 4),
        "AutoARIMA":     round(base_mae * np.random.uniform(0.92, 1.12), 4),
        "AutoCES":       round(base_mae * np.random.uniform(0.88, 1.08), 4),
        "SeasonalNaive": round(base_mae * np.random.uniform(1.10, 1.30), 4),
    })
mae_df = pd.DataFrame(metrics).set_index("Trade")

col_a, col_b = st.columns([2, 3])
with col_a:
    st.markdown("**MAE by Model & Trade**")
    def highlight_min(row):
        min_val = row.min()
        return ["background-color:#D1FAE5;font-weight:600" if v == min_val else "" for v in row]
    st.dataframe(mae_df.style.apply(highlight_min, axis=1).format("{:.4f}"), use_container_width=True)
    st.caption("Green = best model for that trade · MinTrace reconciliation applied post-selection")

with col_b:
    st.markdown("**Load Factor Forecast vs Actuals (NCL · Caribbean)**")
    months = pd.date_range("2024-01-01", periods=18, freq="MS")
    np.random.seed(42)
    actuals = 1.02 + 0.04 * np.sin(np.arange(18) * np.pi / 6) + np.random.normal(0, 0.008, 18)
    fcst_vals = actuals * np.random.uniform(0.98, 1.02, 18)
    lo80 = fcst_vals - 0.018; hi80 = fcst_vals + 0.018
    split = 12

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=months[:split], y=actuals[:split], name="Actuals",
        mode="lines+markers", line=dict(color=SNOW_BLUE, width=2), marker=dict(size=5),
    ))
    fig2.add_trace(go.Scatter(
        x=months[split:], y=actuals[split:], name="Actuals (held-out)",
        mode="lines+markers", line=dict(color=SNOW_BLUE, width=2, dash="dot"), marker=dict(size=5),
    ))
    fig2.add_trace(go.Scatter(
        x=months[split:], y=fcst_vals[split:], name="Forecast (AutoETS/MinTrace)",
        mode="lines+markers", line=dict(color="#7C3AED", width=2), marker=dict(size=5),
    ))
    fig2.add_trace(go.Scatter(
        x=list(months[split:]) + list(months[split:][::-1]),
        y=list(hi80[split:]) + list(lo80[split:][::-1]),
        fill="toself", fillcolor="rgba(124,58,237,0.10)",
        line=dict(color="rgba(255,255,255,0)"), showlegend=True, name="80% Interval",
    ))
    fig2.add_vline(x=months[split].isoformat(), line_dash="dash", line_color="#94A3B8", annotation_text="Forecast start")
    fig2.update_layout(
        height=260, margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
        yaxis=dict(tickformat=".1%", showgrid=True, gridcolor="#F0F2F5"),
        xaxis=dict(showgrid=False), plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Task schedule ─────────────────────────────────────────────────────────────
st.markdown("### Refresh Schedule & Lineage")
col_sched, col_lineage = st.columns(2)

with col_sched:
    st.markdown("""
    <div class="snow-card">
    <b>Snowflake Task: <code>ANALYTICS.REFRESH_FORECAST</code></b><br><br>
    <table style="width:100%;font-size:0.82rem;border-collapse:collapse;">
    <tr style="background:#F0F8FD;"><td style="padding:5px 8px;font-weight:600;">Schedule</td><td style="padding:5px 8px;">1st of every month · 4:00 AM ET</td></tr>
    <tr><td style="padding:5px 8px;font-weight:600;">Warehouse</td><td style="padding:5px 8px;">FORECASTING_WH (X-Small, auto-suspend 60s)</td></tr>
    <tr style="background:#F0F8FD;"><td style="padding:5px 8px;font-weight:600;">Runtime</td><td style="padding:5px 8px;">~4 min (12 bottom-level series × 4 models)</td></tr>
    <tr><td style="padding:5px 8px;font-weight:600;">Output</td><td style="padding:5px 8px;"><code>ANALYTICS.FORECAST_RESULTS</code></td></tr>
    <tr style="background:#F0F8FD;"><td style="padding:5px 8px;font-weight:600;">Sigma refresh</td><td style="padding:5px 8px;">Automatic — views query live on open</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

with col_lineage:
    st.markdown("""
    <div class="snow-card">
    <b>Data Lineage</b><br><br>
    <div style="font-family:monospace;font-size:0.78rem;line-height:2;">
      <span style="color:#6B7280;">Source ERP/Booking</span><br>
      &nbsp;&nbsp;↓ Fivetran / custom ELT<br>
      <span class="schema-tag">RAW.SAILINGS_HIST</span>
      <span class="schema-tag">RAW.SAILINGS_FUTURE</span><br>
      &nbsp;&nbsp;↓ SQL view (no copy)<br>
      <span class="schema-tag">STAGING.MONTHLY_DRIVERS</span><br>
      &nbsp;&nbsp;↓ Snowpark Python SP<br>
      <span class="schema-tag">ANALYTICS.FORECAST_RESULTS</span><br>
      &nbsp;&nbsp;↓ SQL view (no copy)<br>
      <span class="schema-tag">SIGMA.V_BASELINE_FORECAST</span><br>
      &nbsp;&nbsp;↓ Sigma live query<br>
      <span style="color:#0B72E7;font-weight:600;">Σ Sigma Workbook</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">Demo with synthetic data — not based on actual NCLH data</div>',
    unsafe_allow_html=True,
)
