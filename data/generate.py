"""
Synthetic NCLH sailing data generation.
Produces historical sailings, future scheduled sailings, and a simple
baseline forecast using seasonal averages (no heavy ML dependencies).
The forecasting methodology is shown as production Snowpark Python in
the snowflake/ SQL files; this module just generates plausible demo data.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)

# ── Constants ─────────────────────────────────────────────────────────────────

SHIPS = {
    # ship_name: (brand, ship_class, lower_berth_capacity)
    "Norwegian Prima":    ("NCL", "Prima",    3215),
    "Norwegian Viva":     ("NCL", "Prima",    3219),
    "Norwegian Encore":   ("NCL", "Breakaway",3998),
    "Norwegian Bliss":    ("NCL", "Breakaway",4004),
    "Norwegian Escape":   ("NCL", "Breakaway",4266),
    "Norwegian Getaway":  ("NCL", "Breakaway",3963),
    "Norwegian Jade":     ("NCL", "Jewel",    2402),
    "Norwegian Gem":      ("NCL", "Jewel",    2394),
    "Riviera":            ("Oceania", "Riviera", 1238),
    "Marina":             ("Oceania", "Riviera", 1238),
    "Vista":              ("Oceania", "Allura",  1200),
    "Seven Seas Explorer":("Regent", "Explorer", 738),
    "Seven Seas Grandeur":("Regent", "Explorer", 746),
}

TRADE_CONFIG = {
    # trade: (peak_months, itinerary_days, base_per_diem_range, load_factor_range)
    "Caribbean":         ([11,12,1,2,3,4],  [7,7,7,10],    (240, 320), (1.00, 1.12)),
    "Mediterranean":     ([5,6,7,8,9,10],   [7,7,10,10],   (260, 360), (0.97, 1.10)),
    "Alaska":            ([5,6,7,8,9],       [7,7,7],       (280, 380), (0.98, 1.08)),
    "Northern Europe":   ([6,7,8],           [10,10,14],    (290, 410), (0.95, 1.05)),
    "Bermuda":           ([4,5,6,7,8,9,10],  [5,7],         (220, 290), (0.94, 1.06)),
}

BRAND_TRADE_WEIGHTS = {
    "NCL":     {"Caribbean":0.40,"Mediterranean":0.25,"Alaska":0.20,"Northern Europe":0.10,"Bermuda":0.05},
    "Oceania": {"Caribbean":0.30,"Mediterranean":0.40,"Alaska":0.15,"Northern Europe":0.15,"Bermuda":0.00},
    "Regent":  {"Caribbean":0.25,"Mediterranean":0.40,"Alaska":0.15,"Northern Europe":0.20,"Bermuda":0.00},
}

REGENT_MULTIPLIER = 1.45  # Regent commands premium per diems


def _season(month: int) -> str:
    return {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
            6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}[month]


def _is_peak(month: int, trade: str) -> bool:
    return month in TRADE_CONFIG[trade][0]


def _pick_trade(brand: str, rng) -> str:
    trades = list(BRAND_TRADE_WEIGHTS[brand].keys())
    weights = list(BRAND_TRADE_WEIGHTS[brand].values())
    return rng.choice(trades, p=weights)


def generate_historical_sailings(start_year: int = 2022, n_years: int = 3) -> pd.DataFrame:
    """Generate ~2,000 historical sailings over n_years."""
    start_date = date(start_year, 1, 1)
    end_date   = date(start_year + n_years, 1, 1)

    rows = []
    sailing_id = 1000

    current = start_date
    while current < end_date:
        # Year-over-year growth factor (fleet expansion)
        yr_offset = (current.year - start_year)
        growth = 1 + 0.065 * yr_offset

        for ship_name, (brand, ship_class, base_cap) in SHIPS.items():
            trade = _pick_trade(brand, RNG)
            peak_months, itin_opts, pdiem_range, lf_range = TRADE_CONFIG[trade]

            # Only schedule if this trade runs in this month
            if trade == "Alaska" and current.month not in [5,6,7,8,9]:
                continue
            if trade == "Northern Europe" and current.month not in [6,7,8]:
                continue
            if trade == "Bermuda" and current.month not in [4,5,6,7,8,9,10]:
                continue

            # 1-2 sailings per ship per month
            n_sailings = RNG.integers(1, 3)
            for _ in range(n_sailings):
                itin = int(RNG.choice(itin_opts))
                dep_day = RNG.integers(1, 29)
                try:
                    dep_date = current.replace(day=dep_day)
                except ValueError:
                    dep_date = current.replace(day=28)

                peak = _is_peak(dep_date.month, trade)
                lf_base = float(RNG.uniform(*lf_range))
                if peak:
                    lf_base = min(lf_base * 1.03, 1.15)

                cap = int(base_cap * growth)
                pax = min(int(cap * lf_base), int(cap * 1.15))
                lf  = pax / cap

                pd_lo, pd_hi = pdiem_range
                if brand == "Regent":
                    pd_lo *= REGENT_MULTIPLIER; pd_hi *= REGENT_MULTIPLIER
                elif brand == "Oceania":
                    pd_lo *= 1.15; pd_hi *= 1.15
                per_diem = float(RNG.uniform(pd_lo, pd_hi))
                if peak:
                    per_diem *= 1.08
                per_diem *= (1 + RNG.normal(0, 0.03))
                per_diem = max(per_diem, pd_lo * 0.85)

                gross_rev = pax * per_diem * itin

                # Waterfall components
                disc_rate  = float(RNG.uniform(0.05, 0.18)) * (0.85 if peak else 1.0)
                promo_pax  = float(RNG.uniform(30, 80))
                dir_pct    = float(RNG.uniform(0.15, 0.30))
                comm_rate  = float(RNG.uniform(0.10, 0.16))
                override_pct = float(RNG.choice([0, 0, 0.40])) * RNG.uniform(0.01, 0.03)
                kicker     = float(RNG.choice([0, 0, RNG.uniform(0, 75)]))
                air_pct    = float(RNG.uniform(0.10, 0.25))
                air_cost   = float(RNG.uniform(250, 800) if trade in ["Mediterranean","Northern Europe"] else RNG.uniform(250, 450))
                tax_pax    = float(RNG.uniform(80, 200))

                disc_amt   = gross_rev * disc_rate
                promo_tot  = pax * promo_pax
                net_rev_pre_comm = gross_rev - disc_amt
                ta_comm    = net_rev_pre_comm * comm_rate * (1 - dir_pct)
                override   = ta_comm * override_pct
                kicker_tot = (pax / 2) * kicker
                air_tot    = pax * air_pct * air_cost
                tax_tot    = pax * tax_pax
                net_rev    = gross_rev - disc_amt - promo_tot - ta_comm - override - kicker_tot - air_tot - tax_tot

                rows.append({
                    "sailing_id":          sailing_id,
                    "ship_name":           ship_name,
                    "ship_class":          ship_class,
                    "brand":               brand,
                    "trade":               trade,
                    "itinerary_length":    itin,
                    "departure_date":      dep_date,
                    "departure_month":     dep_date.month,
                    "season":              _season(dep_date.month),
                    "lower_berth_capacity":cap,
                    "passengers_booked":   pax,
                    "load_factor":         round(lf, 4),
                    "gross_fare_per_diem": round(per_diem, 2),
                    "gross_ticket_revenue":round(gross_rev, 2),
                    "discount_rate":       round(disc_rate, 4),
                    "promo_cost_per_pax":  round(promo_pax, 2),
                    "commission_rate":     round(comm_rate, 4),
                    "direct_booking_pct":  round(dir_pct, 4),
                    "air_inclusive_pct":   round(air_pct, 4),
                    "air_cost_per_pax":    round(air_cost, 2),
                    "taxes_fees_per_pax":  round(tax_pax, 2),
                    "discount_amount":     round(disc_amt, 2),
                    "promo_cost_total":    round(promo_tot, 2),
                    "ta_commission":       round(ta_comm, 2),
                    "air_cost_total":      round(air_tot, 2),
                    "taxes_total":         round(tax_tot, 2),
                    "net_ticket_revenue":  round(net_rev, 2),
                })
                sailing_id += 1

        current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)

    df = pd.DataFrame(rows)
    df["departure_date"] = pd.to_datetime(df["departure_date"])
    return df.sort_values("departure_date").reset_index(drop=True)


def generate_future_sailings(hist_df: pd.DataFrame) -> pd.DataFrame:
    """Generate next 12 months of scheduled sailings (no revenue — to be forecast)."""
    last_hist = hist_df["departure_date"].max()
    start = (last_hist + pd.offsets.MonthBegin(1)).date()
    end   = (last_hist + pd.offsets.MonthBegin(13)).date()

    rows = []
    sailing_id = 9000
    current = start

    while current < end:
        growth = 1 + 0.065 * 3  # Year 4 growth
        for ship_name, (brand, ship_class, base_cap) in SHIPS.items():
            trade = _pick_trade(brand, RNG)
            _, itin_opts, _, _ = TRADE_CONFIG[trade]

            if trade == "Alaska" and current.month not in [5,6,7,8,9]:
                continue
            if trade == "Northern Europe" and current.month not in [6,7,8]:
                continue
            if trade == "Bermuda" and current.month not in [4,5,6,7,8,9,10]:
                continue

            n_sailings = RNG.integers(1, 3)
            for _ in range(n_sailings):
                itin = int(RNG.choice(itin_opts))
                dep_day = RNG.integers(1, 29)
                try:
                    dep_date = current.replace(day=dep_day)
                except ValueError:
                    dep_date = current.replace(day=28)

                cap = int(base_cap * growth)
                rows.append({
                    "sailing_id":           sailing_id,
                    "ship_name":            ship_name,
                    "ship_class":           ship_class,
                    "brand":                brand,
                    "trade":                trade,
                    "itinerary_length":     itin,
                    "departure_date":       dep_date,
                    "departure_month":      dep_date.month,
                    "season":               _season(dep_date.month),
                    "lower_berth_capacity": cap,
                })
                sailing_id += 1

        current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)

    df = pd.DataFrame(rows)
    df["departure_date"] = pd.to_datetime(df["departure_date"])
    return df.sort_values("departure_date").reset_index(drop=True)


def build_baseline_forecast(hist_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple seasonal-average driver forecast applied to future sailings.
    In production this runs as a Snowpark Python stored procedure in Snowflake.
    """
    # Compute seasonal × trade × ship_class medians from history
    grp = hist_df.groupby(["trade", "ship_class", "season"]).agg(
        lf_med    = ("load_factor",         "median"),
        pd_med    = ("gross_fare_per_diem",  "median"),
        disc_med  = ("discount_rate",        "median"),
        comm_med  = ("commission_rate",      "median"),
        air_med   = ("air_inclusive_pct",    "median"),
        air_cost_med = ("air_cost_per_pax",  "median"),
        promo_med = ("promo_cost_per_pax",   "median"),
        tax_med   = ("taxes_fees_per_pax",   "median"),
        dir_med   = ("direct_booking_pct",   "median"),
    ).reset_index()

    # Fall back to trade × season if ship_class combo missing
    grp_trade = hist_df.groupby(["trade", "season"]).agg(
        lf_med    = ("load_factor",         "median"),
        pd_med    = ("gross_fare_per_diem",  "median"),
        disc_med  = ("discount_rate",        "median"),
        comm_med  = ("commission_rate",      "median"),
        air_med   = ("air_inclusive_pct",    "median"),
        air_cost_med = ("air_cost_per_pax",  "median"),
        promo_med = ("promo_cost_per_pax",   "median"),
        tax_med   = ("taxes_fees_per_pax",   "median"),
        dir_med   = ("direct_booking_pct",   "median"),
    ).reset_index()

    # Apply a 2% YoY growth trend to per diem and 0.5% load factor improvement
    trend_pd = 1.02
    trend_lf = 1.005

    fdf = future_df.copy()
    fdf = fdf.merge(grp, on=["trade","ship_class","season"], how="left")

    # Fill missing with trade-level fallback
    missing = fdf["lf_med"].isna()
    fdf_miss = fdf[missing].drop(columns=[c for c in grp.columns if c not in ["trade","season"]])
    fdf_miss = fdf_miss.merge(grp_trade, on=["trade","season"], how="left")
    fdf.loc[missing, grp.columns.difference(["trade","ship_class","season"])] = fdf_miss[grp.columns.difference(["trade","ship_class","season"])].values

    # Fill any remaining NaN with overall median
    med_col_map = {
        "lf_med":       "load_factor",
        "pd_med":       "gross_fare_per_diem",
        "disc_med":     "discount_rate",
        "comm_med":     "commission_rate",
        "air_med":      "air_inclusive_pct",
        "air_cost_med": "air_cost_per_pax",
        "promo_med":    "promo_cost_per_pax",
        "tax_med":      "taxes_fees_per_pax",
        "dir_med":      "direct_booking_pct",
    }
    for col, hist_col in med_col_map.items():
        fdf[col] = fdf[col].fillna(hist_df[hist_col].median())

    fdf["load_factor"]        = fdf["lf_med"] * trend_lf
    fdf["gross_fare_per_diem"]= fdf["pd_med"] * trend_pd
    fdf["discount_rate"]      = fdf["disc_med"]
    fdf["commission_rate"]    = fdf["comm_med"]
    fdf["air_inclusive_pct"]  = fdf["air_med"]
    fdf["air_cost_per_pax"]   = fdf["air_cost_med"]
    fdf["promo_cost_per_pax"] = fdf["promo_med"]
    fdf["taxes_fees_per_pax"] = fdf["tax_med"]
    fdf["direct_booking_pct"] = fdf["dir_med"]

    fdf["passengers_booked"]   = (fdf["lower_berth_capacity"] * fdf["load_factor"]).astype(int)
    fdf["gross_ticket_revenue"]= fdf["passengers_booked"] * fdf["gross_fare_per_diem"] * fdf["itinerary_length"]

    # Waterfall
    fdf["discount_amount"]  = fdf["gross_ticket_revenue"] * fdf["discount_rate"]
    fdf["promo_cost_total"] = fdf["passengers_booked"] * fdf["promo_cost_per_pax"]
    nr_pre = fdf["gross_ticket_revenue"] - fdf["discount_amount"]
    fdf["ta_commission"]    = nr_pre * fdf["commission_rate"] * (1 - fdf["direct_booking_pct"])
    fdf["air_cost_total"]   = fdf["passengers_booked"] * fdf["air_inclusive_pct"] * fdf["air_cost_per_pax"]
    fdf["taxes_total"]      = fdf["passengers_booked"] * fdf["taxes_fees_per_pax"]
    fdf["net_ticket_revenue"] = (fdf["gross_ticket_revenue"] - fdf["discount_amount"]
                                 - fdf["promo_cost_total"] - fdf["ta_commission"]
                                 - fdf["air_cost_total"] - fdf["taxes_total"])

    drop_cols = [c for c in fdf.columns if c.endswith("_med")]
    return fdf.drop(columns=drop_cols).reset_index(drop=True)


def get_driver_stats(hist_df: pd.DataFrame) -> pd.DataFrame:
    """Return P10/P25/P50/P75/P90 + mean/std for each key driver."""
    drivers = {
        "load_factor":         "Load Factor",
        "gross_fare_per_diem": "Gross Per Diem ($)",
        "discount_rate":       "Discount Rate",
        "commission_rate":     "Commission Rate",
        "air_inclusive_pct":   "Air Inclusive %",
        "promo_cost_per_pax":  "Promo Cost/Pax ($)",
    }
    rows = []
    for col, label in drivers.items():
        s = hist_df[col]
        rows.append({
            "driver": col, "label": label,
            "mean": s.mean(), "std": s.std(),
            "p10": s.quantile(0.10), "p25": s.quantile(0.25),
            "p50": s.quantile(0.50), "p75": s.quantile(0.75),
            "p90": s.quantile(0.90),
        })
    return pd.DataFrame(rows).set_index("driver")


def apply_waterfall(df: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    """Re-run the revenue waterfall with user-supplied driver overrides."""
    d = df.copy()
    for k, v in overrides.items():
        d[k] = v
    d["passengers_booked"]    = (d["lower_berth_capacity"] * d["load_factor"]).astype(int)
    d["gross_ticket_revenue"] = d["passengers_booked"] * d["gross_fare_per_diem"] * d["itinerary_length"]
    d["discount_amount"]      = d["gross_ticket_revenue"] * d["discount_rate"]
    d["promo_cost_total"]     = d["passengers_booked"] * d["promo_cost_per_pax"]
    nr_pre = d["gross_ticket_revenue"] - d["discount_amount"]
    d["ta_commission"]        = nr_pre * d["commission_rate"] * (1 - d["direct_booking_pct"])
    d["air_cost_total"]       = d["passengers_booked"] * d["air_inclusive_pct"] * d["air_cost_per_pax"]
    d["taxes_total"]          = d["passengers_booked"] * d["taxes_fees_per_pax"]
    d["net_ticket_revenue"]   = (d["gross_ticket_revenue"] - d["discount_amount"]
                                 - d["promo_cost_total"] - d["ta_commission"]
                                 - d["air_cost_total"] - d["taxes_total"])
    return d
