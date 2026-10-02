"""
Advanced data summary engine (Part 1, Step 3 of the brief).

Computes the four required analytical pillars:
  1. Sales performance by time period
  2. Product & regional analysis
  3. Customer segmentation by demographics
  4. Statistical measures (median, std dev, etc.)

Every function returns plain pandas objects (DataFrames/dicts) so this
module has **zero dependency on LangChain or any LLM** — it can be unit
tested in complete isolation and reused by the retriever, the evaluator,
and the Streamlit visualizations alike.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. Sales performance by time period
# --------------------------------------------------------------------------
_FREQ_ALIASES = {"M": "ME", "Q": "QE", "Y": "YE"}


def sales_by_period(df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Aggregate revenue/profit/units by calendar period.

    freq: 'D' (day), 'W' (week), 'M' (month), 'Q' (quarter), 'Y' (year)
    """
    resample_freq = _FREQ_ALIASES.get(freq, freq)
    g = (
        df.set_index("order_date")
        .resample(resample_freq)
        .agg(revenue=("revenue", "sum"),
             profit=("profit", "sum"),
             units_sold=("units_sold", "sum"),
             orders=("order_id", "nunique"))
        .reset_index()
    )
    g["period"] = g["order_date"].dt.to_period(freq).astype(str)
    g["revenue_growth_pct"] = g["revenue"].pct_change().mul(100).round(2)
    g["profit_margin_pct"] = (g["profit"] / g["revenue"].replace(0, np.nan) * 100).round(2)
    return g


def period_over_period_highlights(period_df: pd.DataFrame) -> dict[str, Any]:
    """Extract headline trend facts from a `sales_by_period` frame."""
    if period_df.empty:
        return {}
    latest = period_df.iloc[-1]
    prior = period_df.iloc[-2] if len(period_df) > 1 else None
    best = period_df.loc[period_df["revenue"].idxmax()]
    worst = period_df.loc[period_df["revenue"].idxmin()]
    return {
        "latest_period": latest["period"],
        "latest_revenue": round(float(latest["revenue"]), 2),
        "latest_profit_margin_pct": None if pd.isna(latest["profit_margin_pct"]) else round(float(latest["profit_margin_pct"]), 2),
        "period_over_period_growth_pct": (
            None if prior is None or pd.isna(latest["revenue_growth_pct"])
            else round(float(latest["revenue_growth_pct"]), 2)
        ),
        "best_period": best["period"],
        "best_period_revenue": round(float(best["revenue"]), 2),
        "worst_period": worst["period"],
        "worst_period_revenue": round(float(worst["revenue"]), 2),
    }


# --------------------------------------------------------------------------
# 2. Product & regional analysis
# --------------------------------------------------------------------------
def product_performance(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    g = (
        df.groupby(["product_category", "product_name"])
        .agg(revenue=("revenue", "sum"),
             profit=("profit", "sum"),
             units_sold=("units_sold", "sum"),
             avg_unit_price=("unit_price", "mean"),
             orders=("order_id", "nunique"))
        .reset_index()
    )
    g["profit_margin_pct"] = (g["profit"] / g["revenue"] * 100).round(2)
    g["avg_unit_price"] = g["avg_unit_price"].round(2)
    return g.sort_values("revenue", ascending=False).head(top_n).reset_index(drop=True)


def category_performance(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("product_category")
        .agg(revenue=("revenue", "sum"),
             profit=("profit", "sum"),
             units_sold=("units_sold", "sum"),
             orders=("order_id", "nunique"))
        .reset_index()
    )
    g["revenue_share_pct"] = (g["revenue"] / g["revenue"].sum() * 100).round(2)
    g["profit_margin_pct"] = (g["profit"] / g["revenue"] * 100).round(2)
    return g.sort_values("revenue", ascending=False).reset_index(drop=True)


def regional_analysis(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["region", "country"])
        .agg(revenue=("revenue", "sum"),
             profit=("profit", "sum"),
             units_sold=("units_sold", "sum"),
             orders=("order_id", "nunique"),
             customers=("customer_id", "nunique"))
        .reset_index()
    )
    g["revenue_share_pct"] = (g["revenue"] / g["revenue"].sum() * 100).round(2)
    g["avg_order_value"] = (g["revenue"] / g["orders"]).round(2)
    return g.sort_values("revenue", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Customer segmentation by demographics
# --------------------------------------------------------------------------
def customer_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["customer_segment", "age_group"])
        .agg(revenue=("revenue", "sum"),
             profit=("profit", "sum"),
             orders=("order_id", "nunique"),
             customers=("customer_id", "nunique"))
        .reset_index()
    )
    g["revenue_share_pct"] = (g["revenue"] / g["revenue"].sum() * 100).round(2)
    g["avg_revenue_per_customer"] = (g["revenue"] / g["customers"]).round(2)
    return g.sort_values("revenue", ascending=False).reset_index(drop=True)


def segment_totals(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("customer_segment")
        .agg(revenue=("revenue", "sum"),
             profit=("profit", "sum"),
             customers=("customer_id", "nunique"),
             orders=("order_id", "nunique"))
        .reset_index()
    )
    g["revenue_share_pct"] = (g["revenue"] / g["revenue"].sum() * 100).round(2)
    return g.sort_values("revenue", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. Statistical measures
# --------------------------------------------------------------------------
@dataclass
class DescriptiveStats:
    metric: str
    mean: float
    median: float
    std_dev: float
    variance: float
    min: float
    max: float
    p25: float
    p75: float
    skewness: float
    count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def descriptive_statistics(df: pd.DataFrame, column: str) -> DescriptiveStats:
    s = df[column].dropna()
    return DescriptiveStats(
        metric=column,
        mean=round(float(s.mean()), 2),
        median=round(float(s.median()), 2),
        std_dev=round(float(s.std()), 2),
        variance=round(float(s.var()), 2),
        min=round(float(s.min()), 2),
        max=round(float(s.max()), 2),
        p25=round(float(s.quantile(0.25)), 2),
        p75=round(float(s.quantile(0.75)), 2),
        skewness=round(float(s.skew()), 3),
        count=int(s.count()),
    )


def full_statistical_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Descriptive statistics across all key numeric business metrics."""
    per_order_revenue = df.groupby("order_id")["revenue"].sum()
    metrics = {
        "revenue_per_order": per_order_revenue,
        "unit_price": df["unit_price"],
        "units_sold": df["units_sold"],
        "profit": df["profit"],
    }
    out = {}
    for name, series in metrics.items():
        tmp = df.copy()
        tmp["_metric"] = series if len(series) == len(df) else np.nan
        s = series.dropna()
        out[name] = DescriptiveStats(
            metric=name,
            mean=round(float(s.mean()), 2),
            median=round(float(s.median()), 2),
            std_dev=round(float(s.std()), 2),
            variance=round(float(s.var()), 2),
            min=round(float(s.min()), 2),
            max=round(float(s.max()), 2),
            p25=round(float(s.quantile(0.25)), 2),
            p75=round(float(s.quantile(0.75)), 2),
            skewness=round(float(s.skew()), 3),
            count=int(s.count()),
        ).to_dict()
    return out


# --------------------------------------------------------------------------
# Orchestrator: builds every analytical artifact in one call
# --------------------------------------------------------------------------
def build_full_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Compute every analytical artifact the RAG retriever & UI need.

    This is the single entry point used by both the knowledge-base builder
    (for retrieval) and the Streamlit dashboard (for visualization) so the
    numbers are guaranteed to be consistent everywhere in the app.
    """
    monthly = sales_by_period(df, "M")
    quarterly = sales_by_period(df, "Q")
    return {
        "monthly_trend": monthly,
        "quarterly_trend": quarterly,
        "trend_highlights": period_over_period_highlights(monthly),
        "top_products": product_performance(df),
        "category_performance": category_performance(df),
        "regional_analysis": regional_analysis(df),
        "customer_segmentation": customer_segmentation(df),
        "segment_totals": segment_totals(df),
        "statistical_summary": full_statistical_summary(df),
        "totals": {
            "total_revenue": round(float(df["revenue"].sum()), 2),
            "total_profit": round(float(df["profit"].sum()), 2),
            "total_orders": int(df["order_id"].nunique()),
            "total_customers": int(df["customer_id"].nunique()),
            "overall_margin_pct": round(float(df["profit"].sum() / df["revenue"].sum() * 100), 2),
            "date_range": [df["order_date"].min().date().isoformat(), df["order_date"].max().date().isoformat()],
        },
    }
