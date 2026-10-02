"""
Turns the computed analytics (src.analysis.summary_stats) into small,
self-contained natural-language "fact documents".

Why fact documents instead of raw-row chunking?
-------------------------------------------------
Naively chunking thousands of transaction rows and embedding them is a poor
retrieval strategy for tabular BI data — a vector search over raw rows
rarely surfaces the *aggregate* fact ("Q4 2025 revenue grew 18% QoQ") that a
business question actually needs. Instead, InsightForge pre-computes every
statistic once (via `summary_stats`), renders each as a short factual
sentence with rich metadata, and indexes *those*. This gives the retriever
precise, numerically-correct grounding text no matter which embedding model
or vector store backs it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.analysis import summary_stats as stats


@dataclass
class FactDocument:
    doc_id: str
    text: str
    category: str  # e.g. "trend", "product", "region", "segment", "statistics"
    metadata: dict[str, Any] = field(default_factory=dict)


def _trend_facts(analysis: dict[str, Any]) -> list[FactDocument]:
    facts: list[FactDocument] = []
    monthly: pd.DataFrame = analysis["monthly_trend"]
    for _, row in monthly.iterrows():
        growth = row["revenue_growth_pct"]
        growth_txt = "N/A (first period)" if pd.isna(growth) else f"{growth:+.2f}%"
        text = (
            f"In {row['period']}, total revenue was ${row['revenue']:,.2f} "
            f"across {int(row['orders'])} orders ({int(row['units_sold'])} units sold), "
            f"generating ${row['profit']:,.2f} profit "
            f"({row['profit_margin_pct']:.2f}% margin). "
            f"Month-over-month revenue growth was {growth_txt}."
        )
        facts.append(FactDocument(
            doc_id=f"trend-{row['period']}", text=text, category="trend",
            metadata={"period": row["period"], "revenue": float(row["revenue"])},
        ))

    hl = analysis["trend_highlights"]
    if hl:
        text = (
            f"Overall sales trend summary: the most recent period ({hl['latest_period']}) "
            f"recorded ${hl['latest_revenue']:,.2f} in revenue "
            f"(margin {hl['latest_profit_margin_pct']}%). "
            f"The best-performing period on record was {hl['best_period']} "
            f"with ${hl['best_period_revenue']:,.2f} in revenue, while the "
            f"weakest was {hl['worst_period']} with ${hl['worst_period_revenue']:,.2f}."
        )
        facts.append(FactDocument(doc_id="trend-highlights", text=text, category="trend", metadata=hl))
    return facts


def _product_facts(analysis: dict[str, Any]) -> list[FactDocument]:
    facts: list[FactDocument] = []
    top_products: pd.DataFrame = analysis["top_products"]
    if not top_products.empty:
        best_rev = top_products.iloc[0]
        best_margin_row = top_products.loc[top_products["profit_margin_pct"].idxmax()]
        best_profit_row = top_products.assign(_p=top_products["revenue"] * top_products["profit_margin_pct"]).sort_values("profit", ascending=False).iloc[0] if "profit" in top_products else best_rev
        text = (
            f"The single highest-revenue product is '{best_rev['product_name']}' "
            f"({best_rev['product_category']}) with ${best_rev['revenue']:,.2f} in revenue. "
            f"The product with the highest profit margin is '{best_margin_row['product_name']}' "
            f"at {best_margin_row['profit_margin_pct']:.2f}% margin. "
            f"The product generating the most total profit is '{best_profit_row['product_name']}' "
            f"with ${best_profit_row['profit']:,.2f} in profit."
        )
        facts.append(FactDocument(doc_id="product-superlatives", text=text, category="product",
                                   metadata={"top_revenue_product": best_rev["product_name"]}))
    for _, row in top_products.iterrows():
        text = (
            f"Product '{row['product_name']}' (category: {row['product_category']}) "
            f"generated ${row['revenue']:,.2f} in revenue from {int(row['units_sold'])} units "
            f"sold across {int(row['orders'])} orders, at an average price of "
            f"${row['avg_unit_price']:,.2f} and a profit margin of {row['profit_margin_pct']:.2f}%."
        )
        facts.append(FactDocument(
            doc_id=f"product-{row['product_name']}", text=text, category="product",
            metadata={"product_name": row["product_name"], "category": row["product_category"]},
        ))
    cat_perf: pd.DataFrame = analysis["category_performance"]
    if not cat_perf.empty:
        top_cat = cat_perf.iloc[0]
        facts.append(FactDocument(
            doc_id="category-superlative",
            text=(
                f"The best-performing product category by revenue is "
                f"'{top_cat['product_category']}', contributing ${top_cat['revenue']:,.2f} "
                f"({top_cat['revenue_share_pct']:.2f}% of total revenue)."
            ),
            category="product", metadata={"top_category": top_cat["product_category"]},
        ))
    for _, row in cat_perf.iterrows():
        text = (
            f"The '{row['product_category']}' product category contributed "
            f"${row['revenue']:,.2f} in revenue ({row['revenue_share_pct']:.2f}% of total sales) "
            f"with a profit margin of {row['profit_margin_pct']:.2f}%."
        )
        facts.append(FactDocument(
            doc_id=f"category-{row['product_category']}", text=text, category="product",
            metadata={"category": row["product_category"]},
        ))
    return facts


def _regional_facts(analysis: dict[str, Any]) -> list[FactDocument]:
    facts = []
    reg_df: pd.DataFrame = analysis["regional_analysis"]
    if not reg_df.empty:
        top_region = reg_df.iloc[0]
        facts.append(FactDocument(
            doc_id="region-superlative",
            text=(
                f"The best-performing market is {top_region['country']} "
                f"({top_region['region']} region), generating ${top_region['revenue']:,.2f} "
                f"in revenue, {top_region['revenue_share_pct']:.2f}% of the global total."
            ),
            category="region", metadata={"top_country": top_region["country"]},
        ))
    for _, row in reg_df.iterrows():
        text = (
            f"In {row['country']} ({row['region']} region), sales reached "
            f"${row['revenue']:,.2f} ({row['revenue_share_pct']:.2f}% of global revenue) "
            f"from {int(row['customers'])} unique customers across {int(row['orders'])} orders, "
            f"with an average order value of ${row['avg_order_value']:,.2f}."
        )
        facts.append(FactDocument(
            doc_id=f"region-{row['country']}", text=text, category="region",
            metadata={"region": row["region"], "country": row["country"]},
        ))
    return facts


def _segmentation_facts(analysis: dict[str, Any]) -> list[FactDocument]:
    facts = []
    seg_df: pd.DataFrame = analysis["segment_totals"]
    if not seg_df.empty:
        top_seg = seg_df.iloc[0]
        facts.append(FactDocument(
            doc_id="segment-superlative",
            text=(
                f"The highest-revenue customer segment is '{top_seg['customer_segment']}', "
                f"contributing ${top_seg['revenue']:,.2f} ({top_seg['revenue_share_pct']:.2f}% of total) "
                f"from {int(top_seg['customers']):,} unique customers."
            ),
            category="segment", metadata={"top_segment": top_seg["customer_segment"]},
        ))
    for _, row in seg_df.iterrows():
        text = (
            f"The '{row['customer_segment']}' customer segment generated "
            f"${row['revenue']:,.2f} in revenue ({row['revenue_share_pct']:.2f}% of total) "
            f"from {int(row['customers'])} unique customers."
        )
        facts.append(FactDocument(
            doc_id=f"segment-{row['customer_segment']}", text=text, category="segment",
            metadata={"segment": row["customer_segment"]},
        ))
    for _, row in analysis["customer_segmentation"].iterrows():
        text = (
            f"Within the '{row['customer_segment']}' segment, customers aged "
            f"{row['age_group']} generated ${row['revenue']:,.2f} in revenue "
            f"({row['revenue_share_pct']:.2f}% of total), averaging "
            f"${row['avg_revenue_per_customer']:,.2f} revenue per customer."
        )
        facts.append(FactDocument(
            doc_id=f"segment-{row['customer_segment']}-{row['age_group']}", text=text,
            category="segment",
            metadata={"segment": row["customer_segment"], "age_group": row["age_group"]},
        ))
    return facts


def _statistical_facts(analysis: dict[str, Any]) -> list[FactDocument]:
    facts = []
    totals = analysis["totals"]
    text = (
        f"Across the full dataset ({totals['date_range'][0]} to {totals['date_range'][1]}), "
        f"InsightForge recorded ${totals['total_revenue']:,.2f} in total revenue, "
        f"${totals['total_profit']:,.2f} in total profit ({totals['overall_margin_pct']:.2f}% margin), "
        f"across {totals['total_orders']:,} orders from {totals['total_customers']:,} unique customers."
    )
    facts.append(FactDocument(doc_id="totals", text=text, category="statistics", metadata=totals))

    for name, s in analysis["statistical_summary"].items():
        text = (
            f"Statistical profile of '{name}': mean=${s['mean']:,.2f}, median=${s['median']:,.2f}, "
            f"standard deviation=${s['std_dev']:,.2f}, variance={s['variance']:,.2f}, "
            f"range=${s['min']:,.2f}-${s['max']:,.2f}, IQR=${s['p25']:,.2f}-${s['p75']:,.2f}, "
            f"skewness={s['skewness']} (n={s['count']:,}). "
            + (
                "A skewness above 1 indicates a long right tail, i.e. a small number of "
                "high-value orders/products pull the mean well above the median."
                if s["skewness"] > 1 else ""
            )
        )
        facts.append(FactDocument(doc_id=f"stats-{name}", text=text, category="statistics", metadata=s))
    return facts


def build_fact_documents(analysis: dict[str, Any]) -> list[FactDocument]:
    """Build the full set of retrievable fact documents from computed analytics."""
    docs: list[FactDocument] = []
    docs += _trend_facts(analysis)
    docs += _product_facts(analysis)
    docs += _regional_facts(analysis)
    docs += _segmentation_facts(analysis)
    docs += _statistical_facts(analysis)
    return docs
