"""
Generates a realistic, reproducible synthetic sales dataset for InsightForge.

This stands in for a "pre-prepared" SME dataset (per the project brief,
step 1 explicitly focuses on *analysis*, not data cleaning). Swap this out
for a real extract by pointing `INSIGHTFORGE_DATA_PATH` at your own CSV with
the same column contract (see `SCHEMA` below).

Run directly:
    python -m src.data.generate_sample_data
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_settings

SCHEMA = [
    "order_id", "order_date", "region", "country", "product_category",
    "product_name", "customer_id", "customer_segment", "age_group",
    "units_sold", "unit_price", "revenue", "unit_cost", "profit",
]

REGIONS = {
    "North America": ["USA", "Canada"],
    "Europe": ["Germany", "France", "UK"],
    "Asia Pacific": ["India", "Australia", "Japan"],
    "Latin America": ["Brazil", "Mexico"],
}

PRODUCTS = {
    "Laptops": [("ProBook 14", 850), ("UltraSlim X1", 1200), ("Workstation Z", 2100)],
    "Accessories": [("Wireless Mouse", 25), ("Mechanical Keyboard", 70), ("USB-C Hub", 40)],
    "Displays": [("24in Monitor", 180), ("27in 4K Monitor", 340)],
    "Networking": [("Home Router", 60), ("Business Switch 24-port", 310)],
    "Software & Services": [("Extended Warranty", 90), ("Managed IT Support", 220)],
}

SEGMENTS = ["Small Business", "Mid-Market", "Enterprise", "Consumer"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]

SEASONAL_MULTIPLIER = {  # crude yearly seasonality by month
    1: 0.85, 2: 0.85, 3: 0.95, 4: 1.0, 5: 1.0, 6: 1.05,
    7: 0.95, 8: 0.9, 9: 1.05, 10: 1.15, 11: 1.35, 12: 1.5,
}


def generate_sales_data(
    n_rows: int = 6000,
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic, but statistically coherent, sales transaction table."""
    rng = np.random.default_rng(seed)

    date_range = pd.date_range(start_date, end_date, freq="D")
    # Weight day sampling by seasonal multiplier of its month for realistic trends
    day_weights = np.array([SEASONAL_MULTIPLIER[d.month] for d in date_range])
    day_weights = day_weights / day_weights.sum()
    order_dates = rng.choice(date_range, size=n_rows, p=day_weights)

    regions = list(REGIONS.keys())
    region_choices = rng.choice(regions, size=n_rows, p=[0.35, 0.28, 0.27, 0.10])

    categories = list(PRODUCTS.keys())
    # Category popularity weights
    category_choices = rng.choice(
        categories, size=n_rows, p=[0.28, 0.30, 0.16, 0.12, 0.14]
    )

    rows = []
    for i in range(n_rows):
        region = region_choices[i]
        country = rng.choice(REGIONS[region])
        category = category_choices[i]
        product_name, base_price = PRODUCTS[category][
            rng.integers(0, len(PRODUCTS[category]))
        ]
        # Price jitter +/- 8%
        unit_price = round(base_price * rng.uniform(0.92, 1.08), 2)
        unit_cost = round(unit_price * rng.uniform(0.55, 0.75), 2)
        units_sold = int(rng.choice([1, 1, 1, 2, 2, 3, 4, 5, 8, 10], p=[
            0.28, 0.18, 0.12, 0.12, 0.08, 0.08, 0.06, 0.04, 0.02, 0.02
        ]))
        segment = rng.choice(SEGMENTS, p=[0.32, 0.28, 0.18, 0.22])
        age_group = rng.choice(AGE_GROUPS)

        revenue = round(unit_price * units_sold, 2)
        profit = round((unit_price - unit_cost) * units_sold, 2)

        rows.append({
            "order_id": f"ORD-{100000 + i}",
            "order_date": pd.Timestamp(order_dates[i]).date().isoformat(),
            "region": region,
            "country": country,
            "product_category": category,
            "product_name": product_name,
            "customer_id": f"CUST-{rng.integers(1000, 4000)}",
            "customer_segment": segment,
            "age_group": age_group,
            "units_sold": units_sold,
            "unit_price": unit_price,
            "revenue": revenue,
            "unit_cost": unit_cost,
            "profit": profit,
        })

    df = pd.DataFrame(rows, columns=SCHEMA)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df = df.sort_values("order_date").reset_index(drop=True)
    return df


def main() -> None:
    settings = get_settings()
    df = generate_sales_data()
    settings.data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.data_path, index=False)
    print(f"Wrote {len(df):,} rows -> {settings.data_path}")


if __name__ == "__main__":
    main()
