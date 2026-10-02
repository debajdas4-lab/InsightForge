"""
Data visualization (Part 2, Step 7): builds the four required chart types
using Plotly (interactive, Streamlit-native, no server-side rendering deps).

Each function takes the already-computed analytics from
`src.analysis.summary_stats.build_full_analysis` so charts and narrative
answers are always numerically consistent with each other.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_TEMPLATE = "plotly_white"
_COLOR_SEQ = px.colors.qualitative.Bold


def sales_trend_chart(monthly_trend: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_trend["period"], y=monthly_trend["revenue"],
        name="Revenue", marker_color="#3B82F6", yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=monthly_trend["period"], y=monthly_trend["profit_margin_pct"],
        name="Profit margin %", mode="lines+markers",
        line=dict(color="#F59E0B", width=3), yaxis="y2",
    ))
    fig.update_layout(
        template=_TEMPLATE, title="Sales Trend Over Time (Revenue & Margin)",
        xaxis=dict(title="Period", tickangle=-45),
        yaxis=dict(title="Revenue ($)", side="left"),
        yaxis2=dict(title="Profit margin (%)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=430, margin=dict(t=70, b=90),
    )
    return fig


def product_performance_chart(top_products: pd.DataFrame) -> go.Figure:
    df = top_products.sort_values("revenue", ascending=True)
    fig = px.bar(
        df, x="revenue", y="product_name", color="product_category",
        orientation="h", template=_TEMPLATE, color_discrete_sequence=_COLOR_SEQ,
        labels={"revenue": "Revenue ($)", "product_name": "Product", "product_category": "Category"},
        title="Product Performance Comparison (by Revenue)",
        hover_data={"profit_margin_pct": True, "units_sold": True},
    )
    fig.update_layout(height=430, margin=dict(t=70, l=10))
    return fig


def regional_analysis_chart(regional: pd.DataFrame) -> go.Figure:
    fig = px.treemap(
        regional, path=["region", "country"], values="revenue",
        color="revenue_share_pct", color_continuous_scale="Blues",
        template=_TEMPLATE, title="Regional Revenue Breakdown",
        hover_data={"orders": True, "customers": True, "avg_order_value": True},
    )
    fig.update_layout(height=430, margin=dict(t=70, l=10, r=10, b=10))
    return fig


def customer_segmentation_chart(segmentation: pd.DataFrame) -> go.Figure:
    fig = px.sunburst(
        segmentation, path=["customer_segment", "age_group"], values="revenue",
        color="customer_segment", color_discrete_sequence=_COLOR_SEQ,
        template=_TEMPLATE, title="Customer Segmentation: Segment x Age Group",
        hover_data={"customers": True, "avg_revenue_per_customer": True},
    )
    fig.update_layout(height=430, margin=dict(t=70, l=10, r=10, b=10))
    return fig


def category_share_chart(category_performance: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        category_performance, names="product_category", values="revenue",
        hole=0.45, template=_TEMPLATE, color_discrete_sequence=_COLOR_SEQ,
        title="Revenue Share by Product Category",
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=400, margin=dict(t=70))
    return fig


def statistical_distribution_chart(df: pd.DataFrame, column: str = "revenue") -> go.Figure:
    fig = px.histogram(
        df, x=column, nbins=40, template=_TEMPLATE,
        color_discrete_sequence=["#6366F1"],
        title=f"Distribution of {column.replace('_', ' ').title()} (per order line)",
        marginal="box",
    )
    fig.update_layout(height=400, margin=dict(t=70))
    return fig
