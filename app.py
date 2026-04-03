"""
NCLH Revenue Forecasting — Snowflake + Sigma Architecture Demo
Landing page: architecture overview and data initialisation.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

st.set_page_config(
    page_title="NCLH — Snowflake + Sigma Demo",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data.generate import (
    generate_historical_sailings,
    generate_future_sailings,
    build_baseline_forecast,
    get_driver_stats,
)

# ── Color palette ─────────────────────────────────────────────────────────────
NAVY       = "#0B2545"
SNOW_BLUE  = "#29B5E8"   # Snowflake blue
SIGMA_BLUE = "#0B72E7"   # Sigma blue
TEAL       = "#13A89E"
CORAL      = "#E8593C"
LIGHT_GRAY = "#F5F6FA"
BORDER     = "#E3E5EC"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
[data-testid="stSidebar"] {{ background:{NAVY} !important; }}
[data-testid="stSidebar"] * {{ color:#F0F4F8 !important; }}
.main .block-container {{ padding-top:1.5rem; max-width:1400px; }}

div[data-testid="metric-container"] {{
    background:linear-gradient(135deg,{NAVY} 0%,#1a3a6b 100%);
    border-radius:10px; padding:16px;
    border:1px solid rgba(41,181,232,0.25);
}}
div[data-testid="metric-container"] label {{ color:{SNOW_BLUE} !important; font-size:0.75rem; text-transform:uppercase; }}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{ color:white !important; }}

.arch-card {{
    background:white; border:1px solid {BORDER}; border-radius:12px;
    padding:20px 24px; margin-bottom:12px;
    box-shadow:0 2px 8px rgba(0,0,0,0.06);
}}
.arch-card h4 {{ margin:0 0 6px 0; font-size:1rem; }}
.arch-card p  {{ margin:0; font-size:0.82rem; color:#6B7280; line-height:1.5; }}

.badge {{
    display:inline-block; border-radius:4px; padding:2px 8px;
    font-size:0.72rem; font-weight:600; letter-spacing:0.3px;
}}
.badge-snow  {{ background:#E8F6FD; color:{SNOW_BLUE}; }}
.badge-sigma {{ background:#EBF3FE; color:{SIGMA_BLUE}; }}
.badge-python {{ background:#FFF3CD; color:#856404; }}

.disclaimer {{ font-size:0.7rem; color:#8E9BAB; text-align:center;
    border-top:1px solid {BORDER}; padding-top:8px; margin-top:32px; }}
</style>
""", unsafe_allow_html=True)


# ── Cached data loading ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating synthetic sailing data…")
def load_data():
    hist   = generate_historical_sailings()
    future = generate_future_sailings(hist)
    fcst   = build_baseline_forecast(hist, future)
    stats  = get_driver_stats(hist)
    return hist, future, fcst, stats

hist_df, future_df, fcst_df, driver_stats = load_data()

# Store for sub-pages
st.session_state.update({
    "hist_df":      hist_df,
    "future_df":    future_df,
    "fcst_df":      fcst_df,
    "driver_stats": driver_stats,
})

# Pre-load example scenarios
if "scenarios" not in st.session_state:
    baseline_ntr = fcst_df["net_ticket_revenue"].sum()
    st.session_state["scenarios"] = [
        {"name": "Base Case",    "ntr": baseline_ntr,        "tag": "baseline"},
        {"name": "CFO Target",   "ntr": baseline_ntr * 1.08, "tag": "stretch"},
        {"name": "Conservative", "ntr": baseline_ntr * 0.93, "tag": "downside"},
    ]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 20px 0;">
      <div style="font-size:1.5rem;">❄️</div>
      <div style="font-size:0.95rem;font-weight:700;color:#29B5E8;letter-spacing:0.5px;">
        NCLH Revenue Forecasting
      </div>
      <div style="font-size:0.65rem;color:#8E9BAB;margin-top:4px;">
        Snowflake + Sigma Architecture Demo
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Pages")
    st.page_link("app.py",                           label="🏠 Architecture Overview")
    st.page_link("pages/01_snowflake_pipeline.py",   label="❄️ Snowflake Pipeline")
    st.page_link("pages/02_sigma_workbook.py",       label="Σ  Sigma Workbook Mockup")
    st.markdown("---")
    st.markdown("### Data Summary")
    st.metric("Historical Sailings", f"{len(hist_df):,}")
    st.metric("Future Sailings",     f"{len(future_df):,}")
    st.metric("Forecast Horizon",    "12 months")
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.65rem;color:#8E9BAB;text-align:center;">'
        'Demo · Synthetic data only<br>Not based on actual NCLH data'
        '</div>', unsafe_allow_html=True)


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("## NCLH Revenue Forecasting — Snowflake + Sigma")
st.markdown(
    "This demo shows how a **Snowflake-native** bottom-up forecast engine feeds "
    "directly into a **Sigma Computing** workbook, replacing the existing Excel model "
    "while preserving the analyst workflows the finance team already knows."
)

# ── KPIs ─────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
last12 = hist_df[hist_df["departure_date"] >= hist_df["departure_date"].max() - pd.DateOffset(months=12)]
c1.metric("NTR — Last 12M",      f"${last12['net_ticket_revenue'].sum()/1e9:.2f}B")
c2.metric("Forecast NTR (12M)",  f"${fcst_df['net_ticket_revenue'].sum()/1e9:.2f}B")
c3.metric("Avg Load Factor",     f"{hist_df['load_factor'].mean()*100:.1f}%")
c4.metric("Avg Gross Per Diem",  f"${hist_df['gross_fare_per_diem'].mean():,.0f}")

st.markdown("---")

# ── Architecture diagram ──────────────────────────────────────────────────────
st.markdown("### Solution Architecture")

col_diag, col_desc = st.columns([3, 2])

with col_diag:
    # Build a Plotly flow diagram
    fig = go.Figure()

    nodes = [
        # (x, y, label, sublabel, color, shape)
        (0.5, 0.85, "Source Systems",        "ERP · Booking Engine · CRM",   "#6B7280",   "square"),
        (0.5, 0.65, "Snowflake RAW",          "SAILINGS_HIST · SAILINGS_FUTURE", "#29B5E8", "square"),
        (0.5, 0.45, "Snowflake STAGING",      "MONTHLY_DRIVERS · DRIVER_DISTRIBUTIONS", "#0EA5E9", "square"),
        (0.5, 0.25, "Snowpark Python",        "Bottom-up Forecast\nAutoETS · AutoARIMA · MinTrace", "#0369A1", "square"),
        (0.5, 0.05, "Snowflake ANALYTICS",    "FORECAST_RESULTS · WALK_TO_TARGET_RUNS", "#0284C7", "square"),
        (1.1, 0.25, "Sigma Workbook",         "Forecast Dashboard\nWalk to Target · Scenarios", "#0B72E7", "square"),
    ]

    # Arrows
    arrows = [
        (0.5, 0.80, 0.5, 0.71),   # Source → RAW
        (0.5, 0.60, 0.5, 0.51),   # RAW → STAGING
        (0.5, 0.40, 0.5, 0.31),   # STAGING → Snowpark
        (0.5, 0.20, 0.5, 0.11),   # Snowpark → ANALYTICS
        (0.72, 0.05, 1.05, 0.20), # ANALYTICS → Sigma
        (0.72, 0.45, 1.05, 0.30), # STAGING → Sigma (driver distributions)
    ]
    for x0, y0, x1, y1 in arrows:
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="paper", yref="paper", axref="paper", ayref="paper",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
            arrowcolor="#94A3B8",
        )

    # Node boxes
    for x, y, label, sublabel, color, _ in nodes:
        w, h = 0.38, 0.10
        fig.add_shape(type="rect",
            x0=x-w/2, y0=y-h/2, x1=x+w/2, y1=y+h/2,
            xref="paper", yref="paper",
            fillcolor=color, line=dict(color="white", width=1.5),
            layer="above",
        )
        fig.add_annotation(
            x=x, y=y+0.025, xref="paper", yref="paper",
            text=f"<b>{label}</b>",
            showarrow=False, font=dict(color="white", size=11),
        )
        fig.add_annotation(
            x=x, y=y-0.025, xref="paper", yref="paper",
            text=f"<span style='font-size:9px'>{sublabel}</span>",
            showarrow=False, font=dict(color="rgba(255,255,255,0.8)", size=9),
        )

    fig.update_layout(
        height=480, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(visible=False, range=[0, 1.5]),
        yaxis=dict(visible=False, range=[-0.1, 1.0]),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_desc:
    st.markdown("""
    <div class="arch-card">
      <h4>❄️ Snowflake — Forecast Engine</h4>
      <p>
        A <b>Snowpark Python stored procedure</b> runs the full bottom-up
        forecast inside Snowflake on a monthly schedule. No data leaves
        the warehouse — models fit on <code>STAGING.MONTHLY_DRIVERS</code>
        and write results to <code>ANALYTICS.FORECAST_RESULTS</code>.
      </p>
      <br>
      <span class="badge badge-snow">AutoETS</span>&nbsp;
      <span class="badge badge-snow">AutoARIMA</span>&nbsp;
      <span class="badge badge-snow">AutoCES</span>&nbsp;
      <span class="badge badge-python">MinTrace</span>
    </div>

    <div class="arch-card">
      <h4>Σ Sigma — Analyst Workbook</h4>
      <p>
        Sigma connects directly to Snowflake views. Analysts see a
        spreadsheet-style interface they already understand, with
        <b>Input Tables</b> for walk-to-target adjustments that
        write back to Snowflake — no Python, no Streamlit for end users.
      </p>
      <br>
      <span class="badge badge-sigma">Forecast Dashboard</span>&nbsp;
      <span class="badge badge-sigma">Walk to Target</span>&nbsp;
      <span class="badge badge-sigma">Scenarios</span>
    </div>

    <div class="arch-card">
      <h4>📊 Hierarchy & Coherence</h4>
      <p>
        Forecasts are reconciled with <b>MinTrace (mint_shrink)</b>
        so that sailing-level numbers always sum to trade → brand → company
        totals. Finance gets one consistent number across all roll-up levels.
      </p>
    </div>
    """, unsafe_allow_html=True)

# ── Data layer preview ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Snowflake Data Layer Preview")
tab1, tab2, tab3 = st.tabs(["📂 Historical Sailings (RAW)", "📅 Future Sailings (RAW)", "📈 Baseline Forecast (ANALYTICS)"])

with tab1:
    display = hist_df.head(50).copy()
    for c in ["gross_ticket_revenue","net_ticket_revenue","discount_amount","ta_commission","air_cost_total","taxes_total","promo_cost_total"]:
        if c in display.columns:
            display[c] = display[c].apply(lambda x: f"${x:,.0f}")
    for c in ["load_factor","discount_rate","commission_rate","air_inclusive_pct","direct_booking_pct"]:
        if c in display.columns:
            display[c] = display[c].apply(lambda x: f"{x*100:.1f}%")
    display["gross_fare_per_diem"] = display["gross_fare_per_diem"].apply(lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else x)
    cols = ["sailing_id","ship_name","brand","trade","departure_date","lower_berth_capacity",
            "passengers_booked","load_factor","gross_fare_per_diem","gross_ticket_revenue","net_ticket_revenue"]
    st.dataframe(display[[c for c in cols if c in display.columns]], use_container_width=True, height=300)
    st.caption(f"Showing 50 of {len(hist_df):,} rows · Table: `NCLH_FORECASTING.RAW.SAILINGS_HIST`")

with tab2:
    display2 = future_df.head(50).copy()
    display2["lower_berth_capacity"] = display2["lower_berth_capacity"].apply(lambda x: f"{x:,}")
    cols2 = ["sailing_id","ship_name","brand","trade","departure_date","itinerary_length","lower_berth_capacity","season"]
    st.dataframe(display2[[c for c in cols2 if c in display2.columns]], use_container_width=True, height=300)
    st.caption(f"Showing 50 of {len(future_df):,} rows · Table: `NCLH_FORECASTING.RAW.SAILINGS_FUTURE`")

with tab3:
    display3 = fcst_df.head(50).copy()
    for c in ["gross_ticket_revenue","net_ticket_revenue"]:
        if c in display3.columns:
            display3[c] = display3[c].apply(lambda x: f"${x:,.0f}")
    for c in ["load_factor","discount_rate","commission_rate"]:
        if c in display3.columns:
            display3[c] = display3[c].apply(lambda x: f"{x*100:.1f}%")
    display3["gross_fare_per_diem"] = display3["gross_fare_per_diem"].apply(lambda x: f"${x:,.0f}" if isinstance(x, (int,float)) else x)
    cols3 = ["sailing_id","ship_name","brand","trade","departure_date","load_factor",
             "gross_fare_per_diem","gross_ticket_revenue","net_ticket_revenue"]
    st.dataframe(display3[[c for c in cols3 if c in display3.columns]], use_container_width=True, height=300)
    st.caption(f"Showing 50 of {len(fcst_df):,} rows · View: `NCLH_FORECASTING.SIGMA.V_BASELINE_FORECAST`")

st.markdown(
    '<div class="disclaimer">Demo with synthetic data — not based on actual NCLH data</div>',
    unsafe_allow_html=True,
)
