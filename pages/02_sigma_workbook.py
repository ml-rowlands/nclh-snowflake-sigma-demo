"""
Page 2: Sigma Workbook Mockup
Simulates what the client would see when they open the Sigma Computing
workbook that connects to the Snowflake forecast output.

Layout mirrors Sigma's actual UI:
  • Top chrome bar (workbook name, share/publish buttons)
  • Two workbook "pages" implemented as Streamlit tabs
    – Page 1: Forecast Dashboard  (baseline forecast from Snowflake)
    – Page 2: Walk to Target      (Sigma Input Table simulation)
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data.generate import apply_waterfall, get_driver_stats

# ── Palette ───────────────────────────────────────────────────────────────────
SIGMA_BLUE  = "#0B72E7"
SIGMA_DARK  = "#1B1B2F"
SIGMA_GRAY  = "#F5F6FA"
SIGMA_BORDER= "#E3E5EC"
SIGMA_GREEN = "#14B87A"
SIGMA_RED   = "#EF4444"
SIGMA_AMBER = "#F59E0B"
NAVY        = "#0B2545"
WHITE       = "#FFFFFF"

# ── CSS — Sigma workbook chrome ───────────────────────────────────────────────
st.markdown(f"""
<style>
/* Sidebar */
[data-testid="stSidebar"] {{ background:{NAVY} !important; }}
[data-testid="stSidebar"] * {{ color:#F0F4F8 !important; }}

/* Full-width canvas */
.main .block-container {{ padding-top:0 !important; max-width:100%; padding-left:1rem; padding-right:1rem; }}

/* Sigma top chrome bar */
.sigma-chrome {{
    background:white; border-bottom:1px solid {SIGMA_BORDER};
    padding:8px 16px; display:flex; align-items:center;
    box-shadow:0 1px 4px rgba(0,0,0,0.08); margin-bottom:0;
    position:sticky; top:0; z-index:100;
}}
.sigma-logo {{
    width:26px; height:26px; background:{SIGMA_BLUE}; border-radius:5px;
    display:inline-flex; align-items:center; justify-content:center;
    color:white; font-weight:800; font-size:14px; margin-right:10px; flex-shrink:0;
}}
.sigma-breadcrumb {{
    font-size:0.82rem; color:#6B7280;
}}
.sigma-breadcrumb b {{ color:{SIGMA_DARK}; }}
.sigma-actions {{
    margin-left:auto; display:flex; gap:6px; align-items:center;
}}
.sigma-btn {{
    padding:5px 12px; border-radius:5px; font-size:0.78rem;
    font-weight:600; cursor:pointer; border:1px solid {SIGMA_BORDER};
    background:white; color:{SIGMA_DARK};
}}
.sigma-btn-primary {{
    background:{SIGMA_BLUE}; color:white; border-color:{SIGMA_BLUE};
}}
.sigma-saved {{ font-size:0.75rem; color:#9CA3AF; }}

/* KPI tiles */
.sigma-kpi-grid {{
    display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:16px;
}}
.sigma-kpi {{
    background:white; border:1px solid {SIGMA_BORDER}; border-radius:8px;
    padding:14px 16px;
}}
.sigma-kpi-value {{ font-size:1.55rem; font-weight:700; color:{SIGMA_DARK}; line-height:1.2; }}
.sigma-kpi-label {{ font-size:0.72rem; color:#6B7280; text-transform:uppercase;
    letter-spacing:0.4px; margin-top:3px; }}
.sigma-kpi-delta {{ font-size:0.78rem; margin-top:4px; }}
.delta-up {{ color:{SIGMA_GREEN}; }}
.delta-dn {{ color:{SIGMA_RED}; }}

/* Element header (mimics Sigma element title) */
.sigma-elem-title {{
    font-size:0.82rem; font-weight:600; color:{SIGMA_DARK};
    padding:8px 0 4px 0; display:flex; align-items:center; gap:6px;
}}
.sigma-elem-title span {{ color:#9CA3AF; font-size:0.72rem; font-weight:400; }}

/* Input table (Sigma-style editable grid) */
.sigma-input-header {{
    background:{SIGMA_BLUE}; color:white; border-radius:6px 6px 0 0;
    padding:8px 14px; font-size:0.82rem; font-weight:600;
    display:flex; align-items:center; gap:8px;
}}
.sigma-input-table-wrap {{
    border:1px solid {SIGMA_BORDER}; border-radius:0 0 6px 6px; overflow:hidden;
}}

/* Distribution bar (inline sparkline) */
.dist-bar-wrap {{ position:relative; height:8px; background:#E5E7EB; border-radius:4px; }}
.dist-bar-fill {{ height:8px; border-radius:4px; }}

/* Feasibility badge */
.feasibility-badge {{
    display:inline-block; border-radius:20px; padding:4px 14px;
    font-size:0.9rem; font-weight:700; letter-spacing:0.3px;
}}
.feasibility-high   {{ background:#D1FAE5; color:#065F46; }}
.feasibility-medium {{ background:#FEF3C7; color:#92400E; }}
.feasibility-low    {{ background:#FEE2E2; color:#991B1B; }}

/* Page tabs at bottom of workbook */
.sigma-pages {{
    background:{SIGMA_GRAY}; border-top:1px solid {SIGMA_BORDER};
    padding:6px 16px; display:flex; gap:4px; margin-top:24px;
    border-radius:0 0 8px 8px;
}}
.sigma-page-tab {{
    padding:5px 16px; border-radius:4px 4px 0 0; font-size:0.78rem;
    font-weight:500; border:1px solid transparent; cursor:pointer;
    color:#6B7280;
}}
.sigma-page-tab-active {{
    background:white; border-color:{SIGMA_BORDER}; color:{SIGMA_BLUE}; font-weight:600;
}}

.disclaimer {{font-size:0.7rem;color:#8E9BAB;text-align:center;
    border-top:1px solid {SIGMA_BORDER};padding-top:8px;margin-top:24px;}}
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
hist_df    = st.session_state.get("hist_df")
fcst_df    = st.session_state.get("fcst_df")
driver_stats = st.session_state.get("driver_stats")

if hist_df is None:
    st.warning("Please load data from the main page first.")
    st.stop()

if driver_stats is None:
    driver_stats = get_driver_stats(hist_df)

# ── Sigma top chrome bar ──────────────────────────────────────────────────────
st.markdown("""
<div class="sigma-chrome">
  <div class="sigma-logo">Σ</div>
  <div class="sigma-breadcrumb">
    NCLH Finance &rsaquo; Revenue Forecasting &rsaquo; <b>FY2026 Ticket Revenue Model</b>
  </div>
  <div class="sigma-actions">
    <span class="sigma-saved">✓ Saved 2 min ago</span>
    <button class="sigma-btn">⎘ Duplicate</button>
    <button class="sigma-btn">↑ Export</button>
    <button class="sigma-btn sigma-btn-primary">Publish</button>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Workbook "pages" as Streamlit tabs ────────────────────────────────────────
page_tab1, page_tab2 = st.tabs(["📊  Forecast Dashboard", "🎯  Walk to Target"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: FORECAST DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with page_tab1:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Filters bar (mimics Sigma control elements) ───────────────────────────
    with st.container():
        f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
        with f1:
            brand_f = st.multiselect("Brand", ["NCL","Oceania","Regent"],
                                      default=["NCL","Oceania","Regent"], key="dash_brand")
        with f2:
            trade_f = st.multiselect("Trade", fcst_df["trade"].unique().tolist(),
                                      default=fcst_df["trade"].unique().tolist(), key="dash_trade")
        with f3:
            agg_f = st.selectbox("Aggregate by", ["Month","Quarter","Trade","Brand","Ship"], key="dash_agg")
        with f4:
            comp_f = st.selectbox("Compare vs.", ["Prior Year Actuals","CFO Target","Conservative"], key="dash_comp")

    filtered = fcst_df[fcst_df["brand"].isin(brand_f) & fcst_df["trade"].isin(trade_f)]
    hist_filt = hist_df[hist_df["brand"].isin(brand_f) & hist_df["trade"].isin(trade_f)]

    # ── KPI Row ───────────────────────────────────────────────────────────────
    total_ntr   = filtered["net_ticket_revenue"].sum()
    total_gross = filtered["gross_ticket_revenue"].sum()
    avg_lf      = filtered["load_factor"].mean()
    avg_pdiem   = filtered["gross_fare_per_diem"].mean()
    # Prior year for delta context
    last12 = hist_filt[hist_filt["departure_date"] >= hist_filt["departure_date"].max() - pd.DateOffset(months=12)]
    prior_ntr = last12["net_ticket_revenue"].sum()
    ntr_delta_pct = (total_ntr - prior_ntr) / prior_ntr * 100 if prior_ntr else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        delta_col = SIGMA_GREEN if ntr_delta_pct >= 0 else SIGMA_RED
        delta_sym = "▲" if ntr_delta_pct >= 0 else "▼"
        st.markdown(f"""
        <div class="sigma-kpi">
          <div class="sigma-kpi-value">${total_ntr/1e9:.2f}B</div>
          <div class="sigma-kpi-label">Forecast Net Ticket Revenue</div>
          <div class="sigma-kpi-delta" style="color:{delta_col}">
            {delta_sym} {abs(ntr_delta_pct):.1f}% vs prior year
          </div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="sigma-kpi">
          <div class="sigma-kpi-value">${total_gross/1e9:.2f}B</div>
          <div class="sigma-kpi-label">Forecast Gross Revenue</div>
          <div class="sigma-kpi-delta" style="color:#9CA3AF">
            {(total_ntr/total_gross*100):.0f}% net yield ratio
          </div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="sigma-kpi">
          <div class="sigma-kpi-value">{avg_lf*100:.1f}%</div>
          <div class="sigma-kpi-label">Avg Forecasted Load Factor</div>
          <div class="sigma-kpi-delta" style="color:{SIGMA_GREEN}">
            ▲ {(avg_lf - hist_filt['load_factor'].mean())*100:.1f}pp vs historical avg
          </div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="sigma-kpi">
          <div class="sigma-kpi-value">${avg_pdiem:,.0f}</div>
          <div class="sigma-kpi-label">Avg Gross Per Diem</div>
          <div class="sigma-kpi-delta" style="color:{SIGMA_GREEN}">
            ▲ ${avg_pdiem - hist_filt['gross_fare_per_diem'].mean():,.0f} vs historical avg
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns([3, 2])

    with ch1:
        st.markdown('<div class="sigma-elem-title">📈 Net Ticket Revenue — Forecast vs Prior Year <span>Line chart · Connected to SIGMA.V_BASELINE_FORECAST</span></div>', unsafe_allow_html=True)

        # Build monthly forecast
        fmonthly = (
            filtered
            .assign(month=lambda d: d["departure_date"].dt.to_period("M").dt.to_timestamp())
            .groupby("month")["net_ticket_revenue"].sum()
            .reset_index()
        )
        # Build monthly actuals (last 12 months)
        amonthly = (
            hist_filt
            .assign(month=lambda d: d["departure_date"].dt.to_period("M").dt.to_timestamp())
            .groupby("month")["net_ticket_revenue"].sum()
            .reset_index()
        )
        # Prior year (shift forecast months back 12)
        prior = fmonthly.copy()
        prior["month"] = prior["month"] - pd.DateOffset(years=1)
        prior.rename(columns={"net_ticket_revenue":"prior_ntr"}, inplace=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=amonthly["month"], y=amonthly["net_ticket_revenue"],
            name="Actuals", mode="lines+markers",
            line=dict(color="#94A3B8", width=2),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=fmonthly["month"], y=fmonthly["net_ticket_revenue"],
            name="Forecast", mode="lines+markers",
            line=dict(color=SIGMA_BLUE, width=2.5),
            marker=dict(size=6),
        ))
        if "CFO" in comp_f:
            fig.add_trace(go.Scatter(
                x=fmonthly["month"], y=fmonthly["net_ticket_revenue"] * 1.08,
                name="CFO Target", mode="lines",
                line=dict(color=SIGMA_GREEN, width=1.5, dash="dash"),
            ))
        elif "Conservative" in comp_f:
            fig.add_trace(go.Scatter(
                x=fmonthly["month"], y=fmonthly["net_ticket_revenue"] * 0.93,
                name="Conservative", mode="lines",
                line=dict(color=SIGMA_AMBER, width=1.5, dash="dash"),
            ))
        else:  # Prior Year
            fig.add_trace(go.Scatter(
                x=fmonthly["month"], y=fmonthly["net_ticket_revenue"] * 0.95,
                name="Prior Year (shifted)", mode="lines",
                line=dict(color="#CBD5E1", width=1.5, dash="dot"),
            ))

        # Forecast start vline
        forecast_start = fmonthly["month"].min()
        fig.add_vline(x=forecast_start.isoformat(), line_dash="dash",
                      line_color="#CBD5E1", annotation_text="Forecast →", annotation_font_size=10)
        fig.update_layout(
            height=280, margin=dict(l=0,r=0,t=5,b=0),
            legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
            yaxis=dict(tickprefix="$", tickformat=",.0f", showgrid=True, gridcolor="#F0F2F5"),
            xaxis=dict(showgrid=False),
            plot_bgcolor="white", paper_bgcolor="white",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        st.markdown('<div class="sigma-elem-title">🍩 NTR by Brand <span>Donut · Connected to V_BASELINE_FORECAST</span></div>', unsafe_allow_html=True)
        brand_ntr = filtered.groupby("brand")["net_ticket_revenue"].sum().reset_index()
        fig2 = go.Figure(go.Pie(
            labels=brand_ntr["brand"], values=brand_ntr["net_ticket_revenue"],
            hole=0.58,
            marker_colors=[SIGMA_BLUE, "#0EA5E9", "#7C3AED"],
            textinfo="label+percent",
            textfont=dict(size=12),
        ))
        fig2.update_layout(
            height=200, margin=dict(l=0,r=0,t=5,b=10),
            showlegend=False, paper_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="sigma-elem-title" style="margin-top:8px;">📊 NTR by Trade <span>Bar</span></div>', unsafe_allow_html=True)
        trade_ntr = filtered.groupby("trade")["net_ticket_revenue"].sum().sort_values()
        fig3 = go.Figure(go.Bar(
            x=trade_ntr.values, y=trade_ntr.index, orientation="h",
            marker_color=SIGMA_BLUE, marker_opacity=0.85,
        ))
        fig3.update_layout(
            height=160, margin=dict(l=0,r=0,t=5,b=0),
            xaxis=dict(tickprefix="$", tickformat=",.0f", showgrid=True, gridcolor="#F0F2F5"),
            yaxis=dict(showgrid=False),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Waterfall ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sigma-elem-title">💧 Revenue Waterfall <span>Waterfall chart · Connected to SIGMA.V_WATERFALL_SUMMARY</span></div>', unsafe_allow_html=True)

    gross = filtered["gross_ticket_revenue"].sum()
    disc  = filtered["discount_amount"].sum()
    promo = filtered["promo_cost_total"].sum()
    comm  = filtered["ta_commission"].sum()
    air   = filtered["air_cost_total"].sum()
    tax   = filtered["taxes_total"].sum()
    net   = filtered["net_ticket_revenue"].sum()

    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute","relative","relative","relative","relative","relative","total"],
        x=["Gross Revenue","Discounts","Promo Costs","TA Commission","Air Costs","Taxes & Fees","Net Ticket Revenue"],
        y=[gross, -disc, -promo, -comm, -air, -tax, net],
        connector=dict(line=dict(color="#CBD5E1", width=1)),
        increasing=dict(marker_color=SIGMA_BLUE),
        decreasing=dict(marker_color="#EF4444"),
        totals=dict(marker_color=SIGMA_GREEN),
        text=[f"${v/1e6:,.0f}M" for v in [gross,-disc,-promo,-comm,-air,-tax,net]],
        textposition="outside",
    ))
    fig_wf.update_layout(
        height=320, margin=dict(l=0,r=0,t=10,b=0),
        yaxis=dict(tickprefix="$", tickformat=",.0f", showgrid=True, gridcolor="#F0F2F5"),
        xaxis=dict(showgrid=False),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # ── Forecast table ────────────────────────────────────────────────────────
    st.markdown('<div class="sigma-elem-title">📋 Sailing-Level Forecast <span>Table · Source: SIGMA.V_BASELINE_FORECAST</span></div>', unsafe_allow_html=True)

    agg_opt = {
        "Month": lambda d: d.assign(period=d["departure_date"].dt.to_period("M").astype(str)),
        "Quarter": lambda d: d.assign(period="Q"+d["departure_date"].dt.quarter.astype(str)+" "+d["departure_date"].dt.year.astype(str)),
        "Trade": lambda d: d.assign(period=d["trade"]),
        "Brand": lambda d: d.assign(period=d["brand"]),
        "Ship":  lambda d: d.assign(period=d["ship_name"]),
    }
    tbl_df = agg_opt[agg_f](filtered).groupby("period").agg(
        Sailings       = ("sailing_id",          "count"),
        Capacity       = ("lower_berth_capacity", "sum"),
        Passengers     = ("passengers_booked",    "sum"),
        **{"Load Factor": ("load_factor",          "mean")},
        **{"Avg Per Diem": ("gross_fare_per_diem",  "mean")},
        **{"Gross Revenue": ("gross_ticket_revenue","sum")},
        **{"Net Ticket Revenue": ("net_ticket_revenue","sum")},
    ).reset_index().rename(columns={"period": agg_f})

    tbl_df["Load Factor"]       = tbl_df["Load Factor"].apply(lambda x: f"{x*100:.1f}%")
    tbl_df["Avg Per Diem"]      = tbl_df["Avg Per Diem"].apply(lambda x: f"${x:,.0f}")
    tbl_df["Gross Revenue"]     = tbl_df["Gross Revenue"].apply(lambda x: f"${x/1e6:,.1f}M")
    tbl_df["Net Ticket Revenue"]= tbl_df["Net Ticket Revenue"].apply(lambda x: f"${x/1e6:,.1f}M")
    tbl_df["Capacity"]          = tbl_df["Capacity"].apply(lambda x: f"{x:,}")
    tbl_df["Passengers"]        = tbl_df["Passengers"].apply(lambda x: f"{x:,}")
    st.dataframe(tbl_df, use_container_width=True, hide_index=True, height=280)
    st.caption("Source: `NCLH_FORECASTING.SIGMA.V_BASELINE_FORECAST` — live query on Snowflake")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: WALK TO TARGET
# ══════════════════════════════════════════════════════════════════════════════
with page_tab2:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Header explanation ────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#EBF3FE;border:1px solid #BFDBFE;border-radius:8px;padding:12px 16px;margin-bottom:16px;">
      <b style="color:{SIGMA_BLUE};">Σ Sigma Input Table</b>
      <span style="font-size:0.82rem;color:#374151;margin-left:8px;">
        Edit the driver assumptions below. Sigma writes your inputs back to
        <code>ANALYTICS.WALK_TO_TARGET_RUNS</code> in Snowflake and recomputes
        the revenue waterfall automatically.
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Target input ──────────────────────────────────────────────────────────
    baseline_ntr_m = fcst_df["net_ticket_revenue"].sum() / 1e6
    t_col, b_col = st.columns([3, 1])
    with t_col:
        target_ntr_m = st.number_input(
            "Annual Net Ticket Revenue Target ($M)",
            min_value=float(baseline_ntr_m * 0.7),
            max_value=float(baseline_ntr_m * 1.5),
            value=float(round(baseline_ntr_m * 1.05)),
            step=10.0,
            format="%.0f",
        )
    with b_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        required_delta_pct = (target_ntr_m - baseline_ntr_m) / baseline_ntr_m * 100
        delta_color = SIGMA_GREEN if required_delta_pct >= 0 else SIGMA_RED
        st.markdown(f"""
        <div style="background:{delta_color}15;border:1px solid {delta_color}40;border-radius:6px;
                    padding:8px 12px;text-align:center;">
          <div style="font-size:1.1rem;font-weight:700;color:{delta_color}">
            {"▲" if required_delta_pct>=0 else "▼"} {abs(required_delta_pct):.1f}%
          </div>
          <div style="font-size:0.72rem;color:{SIGMA_DARK}">vs baseline</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Sigma Input Table ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="sigma-input-header">
      ✎ &nbsp; Driver Assumption Overrides
      <span style="font-size:0.72rem;opacity:0.8;margin-left:4px;">
        · Editable — changes reflected instantly · Writeback to Snowflake on Save
      </span>
    </div>
    """, unsafe_allow_html=True)

    # Build the editable driver table
    DRIVERS = {
        "load_factor":         ("Load Factor",          "×",   1,    0.80, 1.20, 3),
        "gross_fare_per_diem": ("Gross Per Diem",        "$",   1,  150.0, 550.0, 0),
        "discount_rate":       ("Discount Rate",         "%", 100,   0.03,  0.25, 1),
        "commission_rate":     ("Commission Rate",       "%", 100,   0.05,  0.20, 1),
        "air_inclusive_pct":   ("Air Inclusive %",       "%", 100,   0.05,  0.40, 1),
        "promo_cost_per_pax":  ("Promo Cost / Pax",      "$",   1,   20.0, 120.0, 0),
    }

    input_rows = []
    for key, (label, unit, scale, lo, hi, dec) in DRIVERS.items():
        row = driver_stats.loc[key]
        b_val = float(fcst_df[key].mean())
        input_rows.append({
            "Driver":            label,
            "Unit":              unit,
            "Historical P50":    round(row["p50"] * scale, dec),
            "P10":               round(row["p10"] * scale, dec),
            "P90":               round(row["p90"] * scale, dec),
            "Baseline Forecast": round(b_val * scale, dec),
            "Your Assumption":   round(b_val * scale, dec),
            "_key":              key,
            "_scale":            scale,
        })

    input_df = pd.DataFrame(input_rows)

    edit_df = st.data_editor(
        input_df[["Driver","Historical P50","P10","P90","Baseline Forecast","Your Assumption"]],
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Driver":            st.column_config.TextColumn("Driver", disabled=True),
            "Historical P50":    st.column_config.NumberColumn("Historical P50", disabled=True, format="%.2f"),
            "P10":               st.column_config.NumberColumn("P10", disabled=True, format="%.2f"),
            "P90":               st.column_config.NumberColumn("P90", disabled=True, format="%.2f"),
            "Baseline Forecast": st.column_config.NumberColumn("Baseline (Snowflake)", disabled=True, format="%.2f"),
            "Your Assumption":   st.column_config.NumberColumn("✎ Your Assumption", format="%.2f",
                                    help="Edit this column to override the driver assumption"),
        },
        key="driver_input_table",
    )

    # ── Distribution indicators ───────────────────────────────────────────────
    st.markdown("<div style='margin-top:8px;margin-bottom:4px;font-size:0.78rem;color:#6B7280;'>Driver feasibility — where your assumptions sit on the historical distribution:</div>", unsafe_allow_html=True)

    indicator_cols = st.columns(len(DRIVERS))
    overrides = {}

    for i, (key, (label, unit, scale, lo, hi, dec)) in enumerate(DRIVERS.items()):
        row = driver_stats.loc[key]
        user_val_scaled = edit_df.loc[i, "Your Assumption"]
        user_val = user_val_scaled / scale
        overrides[key] = user_val

        p10, p25, p50, p75, p90 = row["p10"]*scale, row["p25"]*scale, row["p50"]*scale, row["p75"]*scale, row["p90"]*scale
        p_range = p90 - p10
        if p_range == 0:
            pct_pos = 50.0
        else:
            pct_pos = min(max((user_val_scaled - p10) / p_range * 100, 0), 100)

        # Color: Green P25-P75, Amber P10-P25 or P75-P90, Red outside P10/P90
        if p25 <= user_val_scaled <= p75:
            ind_color = SIGMA_GREEN; feasibility = "Normal"
        elif p10 <= user_val_scaled <= p90:
            ind_color = SIGMA_AMBER; feasibility = "Stretch"
        else:
            ind_color = SIGMA_RED; feasibility = "Heroic"

        with indicator_cols[i]:
            st.markdown(f"""
            <div style="background:white;border:1px solid {SIGMA_BORDER};border-radius:6px;padding:8px 10px;text-align:center;">
              <div style="font-size:0.7rem;color:#6B7280;margin-bottom:4px;">{label}</div>
              <div style="font-size:0.95rem;font-weight:700;color:{SIGMA_DARK}">
                {"%" if unit=="%" else ("$" if unit=="$" else "")}{user_val_scaled:.{dec}f}{"×" if unit=="×" else ""}
              </div>
              <div style="margin:5px 0;">
                <div class="dist-bar-wrap">
                  <div class="dist-bar-fill" style="width:{pct_pos:.0f}%;background:{ind_color};"></div>
                </div>
              </div>
              <div style="font-size:0.68rem;font-weight:600;color:{ind_color}">{feasibility}</div>
              <div style="font-size:0.65rem;color:#9CA3AF">P10:{p10:.1f} — P90:{p90:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Recompute waterfall with user overrides ───────────────────────────────
    scenario_df = apply_waterfall(fcst_df.copy(), overrides)
    scenario_ntr_m = scenario_df["net_ticket_revenue"].sum() / 1e6
    gap_m = scenario_ntr_m - baseline_ntr_m
    gap_vs_target = scenario_ntr_m - target_ntr_m

    # ── Monte Carlo feasibility ───────────────────────────────────────────────
    N_SIM = 3000
    np.random.seed(99)
    sim_ntrs = []
    for _ in range(N_SIM):
        sim_overrides = {}
        for key in DRIVERS:
            row = driver_stats.loc[key]
            noise = np.random.normal(0, row["std"])
            sim_overrides[key] = overrides[key] + noise
        sim_df = apply_waterfall(fcst_df.copy(), sim_overrides)
        sim_ntrs.append(sim_df["net_ticket_revenue"].sum() / 1e6)
    sim_ntrs = np.array(sim_ntrs)
    feasibility_pct = float(np.mean(sim_ntrs >= target_ntr_m) * 100)

    # ── Results section ───────────────────────────────────────────────────────
    st.markdown("### Scenario Results")
    r1, r2, r3 = st.columns(3)

    gap_color = SIGMA_GREEN if gap_vs_target >= 0 else SIGMA_RED
    gap_sym   = "▲" if gap_vs_target >= 0 else "▼"

    with r1:
        st.markdown(f"""
        <div class="sigma-kpi" style="border-left:4px solid {SIGMA_BLUE};">
          <div class="sigma-kpi-label">Baseline Forecast</div>
          <div class="sigma-kpi-value">${baseline_ntr_m:,.0f}M</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        c = SIGMA_GREEN if scenario_ntr_m >= baseline_ntr_m else SIGMA_RED
        st.markdown(f"""
        <div class="sigma-kpi" style="border-left:4px solid {c};">
          <div class="sigma-kpi-label">Your Scenario</div>
          <div class="sigma-kpi-value" style="color:{c}">${scenario_ntr_m:,.0f}M</div>
          <div class="sigma-kpi-delta" style="color:{c}">
            {"▲" if gap_m>=0 else "▼"} ${abs(gap_m):,.0f}M vs baseline
          </div>
        </div>""", unsafe_allow_html=True)
    with r3:
        feas_class = "feasibility-high" if feasibility_pct >= 60 else ("feasibility-medium" if feasibility_pct >= 35 else "feasibility-low")
        feas_label = "High likelihood" if feasibility_pct >= 60 else ("Moderate — stretch" if feasibility_pct >= 35 else "Low — heroic assumptions")
        st.markdown(f"""
        <div class="sigma-kpi" style="border-left:4px solid {gap_color};">
          <div class="sigma-kpi-label">vs Target (${target_ntr_m:,.0f}M)</div>
          <div class="sigma-kpi-value" style="color:{gap_color}">{gap_sym} ${abs(gap_vs_target):,.0f}M</div>
          <div class="sigma-kpi-delta" style="margin-top:6px;">
            <span class="feasibility-badge {feas_class}">{feasibility_pct:.0f}% Feasibility</span>
          </div>
          <div style="font-size:0.72rem;color:#6B7280;margin-top:4px;">{feas_label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Adjustment waterfall ──────────────────────────────────────────────────
    wf_col, mc_col = st.columns([3, 2])

    with wf_col:
        st.markdown('<div class="sigma-elem-title">💧 Walk to Target — Adjustment Waterfall <span>What drives the gap from baseline to your scenario</span></div>', unsafe_allow_html=True)

        waterfall_items = []
        for key, (label, unit, scale, lo, hi, dec) in DRIVERS.items():
            base_val = float(fcst_df[key].mean())
            user_val = overrides[key]
            if abs(user_val - base_val) < 1e-8:
                continue
            # Isolate the impact of this driver
            only_this = {k: float(fcst_df[k].mean()) for k in DRIVERS}
            only_this[key] = user_val
            df_single = apply_waterfall(fcst_df.copy(), only_this)
            impact_m = (df_single["net_ticket_revenue"].sum() - fcst_df["net_ticket_revenue"].sum()) / 1e6
            if abs(impact_m) > 0.1:
                waterfall_items.append((label, impact_m))

        if waterfall_items:
            measures = ["absolute"] + ["relative"] * len(waterfall_items) + ["total"]
            x_vals   = ["Baseline"] + [w[0] for w in waterfall_items] + ["Your Scenario"]
            y_vals   = [baseline_ntr_m] + [w[1] for w in waterfall_items] + [scenario_ntr_m]
            texts    = [f"${baseline_ntr_m:,.0f}M"] + [f"{'+'if w[1]>=0 else ''}${w[1]:,.1f}M" for w in waterfall_items] + [f"${scenario_ntr_m:,.0f}M"]

            fig_adj = go.Figure(go.Waterfall(
                orientation="v", measure=measures,
                x=x_vals, y=y_vals, text=texts,
                textposition="outside",
                connector=dict(line=dict(color="#CBD5E1", width=1)),
                increasing=dict(marker_color=SIGMA_GREEN),
                decreasing=dict(marker_color=SIGMA_RED),
                totals=dict(marker_color=SIGMA_BLUE),
            ))
            # Target line
            fig_adj.add_hline(y=target_ntr_m, line_dash="dash", line_color=SIGMA_AMBER,
                              annotation_text=f"Target ${target_ntr_m:,.0f}M", annotation_font_size=10)
            fig_adj.update_layout(
                height=320, margin=dict(l=0,r=0,t=10,b=0),
                yaxis=dict(tickprefix="$", ticksuffix="M", showgrid=True, gridcolor="#F0F2F5"),
                xaxis=dict(showgrid=False),
                plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
            )
            st.plotly_chart(fig_adj, use_container_width=True)
        else:
            st.info("Adjust driver assumptions above to see the waterfall impact.")

    with mc_col:
        st.markdown('<div class="sigma-elem-title">🎲 Joint Feasibility <span>3,000 Monte Carlo draws from historical driver distributions</span></div>', unsafe_allow_html=True)

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=sim_ntrs, nbinsx=50,
            marker_color=SIGMA_BLUE, opacity=0.75, name="Simulated NTR",
        ))
        fig_mc.add_vline(x=target_ntr_m, line_color=SIGMA_AMBER, line_width=2, line_dash="dash",
                         annotation_text=f"Target ${target_ntr_m:,.0f}M", annotation_font_size=10)
        fig_mc.add_vline(x=scenario_ntr_m, line_color=SIGMA_GREEN, line_width=2,
                         annotation_text=f"Scenario ${scenario_ntr_m:,.0f}M", annotation_font_size=10)
        fig_mc.update_layout(
            height=220, margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(tickprefix="$", ticksuffix="M", showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#F0F2F5", title="Simulations"),
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        # Summary stats
        st.markdown(f"""
        <table style="width:100%;font-size:0.78rem;border-collapse:collapse;">
        <tr style="background:#F0F8FD;"><td style="padding:4px 8px;font-weight:600;">Simulations</td><td style="padding:4px 8px;">{N_SIM:,}</td></tr>
        <tr><td style="padding:4px 8px;font-weight:600;">P10 outcome</td><td style="padding:4px 8px;">${np.percentile(sim_ntrs,10):,.0f}M</td></tr>
        <tr style="background:#F0F8FD;"><td style="padding:4px 8px;font-weight:600;">P50 outcome</td><td style="padding:4px 8px;">${np.percentile(sim_ntrs,50):,.0f}M</td></tr>
        <tr><td style="padding:4px 8px;font-weight:600;">P90 outcome</td><td style="padding:4px 8px;">${np.percentile(sim_ntrs,90):,.0f}M</td></tr>
        <tr style="background:{SIGMA_GREEN}15;"><td style="padding:4px 8px;font-weight:700;color:{SIGMA_GREEN}">P(≥ Target)</td>
          <td style="padding:4px 8px;font-weight:700;color:{SIGMA_GREEN}">{feasibility_pct:.1f}%</td></tr>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Save scenario button ──────────────────────────────────────────────────
    save_col, name_col = st.columns([2, 3])
    with name_col:
        scenario_name = st.text_input("Scenario name:", value=f"My Scenario — ${scenario_ntr_m:,.0f}M", key="scen_name")
    with save_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("💾  Save to Snowflake (ANALYTICS.WALK_TO_TARGET_RUNS)", type="primary"):
            if "scenarios" not in st.session_state:
                st.session_state["scenarios"] = []
            st.session_state["scenarios"].append({
                "name":          scenario_name,
                "ntr":           scenario_ntr_m * 1e6,
                "target_ntr_m":  target_ntr_m,
                "feasibility":   feasibility_pct,
                "overrides":     overrides.copy(),
                "tag":           "user",
            })
            st.success(f"✓ Scenario '{scenario_name}' saved — NTR ${scenario_ntr_m:,.0f}M · Feasibility {feasibility_pct:.0f}%")
            st.info("In production, Sigma writes this row to `NCLH_FORECASTING.ANALYTICS.WALK_TO_TARGET_RUNS` via its native Snowflake writeback connector.")

# ── Sigma page tabs footer ────────────────────────────────────────────────────
st.markdown("""
<div class="sigma-pages">
  <div class="sigma-page-tab sigma-page-tab-active">📊 Forecast Dashboard</div>
  <div class="sigma-page-tab sigma-page-tab-active">🎯 Walk to Target</div>
  <div class="sigma-page-tab" style="color:#9CA3AF;">+ Add page</div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">Demo with synthetic data — not based on actual NCLH data · '
    'Sigma Computing UI elements simulated for demonstration purposes</div>',
    unsafe_allow_html=True,
)
