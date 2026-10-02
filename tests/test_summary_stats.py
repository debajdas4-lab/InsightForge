from __future__ import annotations

import numpy as np
import pytest

from src.analysis import summary_stats as stats


def test_sales_by_period_sums_match_totals(sample_df):
    monthly = stats.sales_by_period(sample_df, "M")
    assert abs(monthly["revenue"].sum() - sample_df["revenue"].sum()) < 1e-6
    assert abs(monthly["profit"].sum() - sample_df["profit"].sum()) < 1e-6
    assert monthly["units_sold"].sum() == sample_df["units_sold"].sum()


def test_trend_highlights_identifies_best_and_worst(sample_df):
    monthly = stats.sales_by_period(sample_df, "M")
    hl = stats.period_over_period_highlights(monthly)
    best_row = monthly.loc[monthly["period"] == hl["best_period"]].iloc[0]
    worst_row = monthly.loc[monthly["period"] == hl["worst_period"]].iloc[0]
    assert best_row["revenue"] == monthly["revenue"].max()
    assert worst_row["revenue"] == monthly["revenue"].min()


def test_product_performance_revenue_share_sums_to_100(sample_df):
    cat_perf = stats.category_performance(sample_df)
    assert abs(cat_perf["revenue_share_pct"].sum() - 100.0) < 0.05


def test_regional_analysis_revenue_share_sums_to_100(sample_df):
    reg = stats.regional_analysis(sample_df)
    assert abs(reg["revenue_share_pct"].sum() - 100.0) < 0.05


def test_customer_segmentation_totals_match(sample_df):
    seg_totals = stats.segment_totals(sample_df)
    assert abs(seg_totals["revenue"].sum() - sample_df["revenue"].sum()) < 1e-6


def test_descriptive_statistics_matches_numpy(sample_df):
    result = stats.descriptive_statistics(sample_df, "unit_price")
    expected_mean = round(float(sample_df["unit_price"].mean()), 2)
    expected_median = round(float(sample_df["unit_price"].median()), 2)
    expected_std = round(float(sample_df["unit_price"].std()), 2)
    assert result.mean == expected_mean
    assert result.median == expected_median
    assert result.std_dev == expected_std
    assert result.count == len(sample_df)


def test_descriptive_statistics_median_le_or_ge_mean_when_skewed(sample_df):
    """Right-skewed distributions should show mean > median (sanity check)."""
    result = stats.descriptive_statistics(sample_df, "unit_price")
    if result.skewness > 0.5:
        assert result.mean > result.median


def test_build_full_analysis_totals_are_internally_consistent(sample_df):
    analysis = stats.build_full_analysis(sample_df)
    totals = analysis["totals"]
    assert totals["total_orders"] == sample_df["order_id"].nunique()
    assert totals["total_customers"] == sample_df["customer_id"].nunique()
    assert abs(totals["total_revenue"] - sample_df["revenue"].sum()) < 1e-6
    expected_margin = round(float(sample_df["profit"].sum() / sample_df["revenue"].sum() * 100), 2)
    assert totals["overall_margin_pct"] == expected_margin


def test_build_full_analysis_returns_all_expected_keys(sample_df):
    analysis = stats.build_full_analysis(sample_df)
    expected_keys = {
        "monthly_trend", "quarterly_trend", "trend_highlights", "top_products",
        "category_performance", "regional_analysis", "customer_segmentation",
        "segment_totals", "statistical_summary", "totals",
    }
    assert expected_keys.issubset(analysis.keys())
