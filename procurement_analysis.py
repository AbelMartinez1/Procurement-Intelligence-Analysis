"""
Procurement Intelligence Analysis
---------------------------------
End-to-end analytics project on SYNTHETIC purchasing data:
relational modeling (SQLite), SQL window functions, data-quality checks,
transformation (pandas) and BI-style visualization (seaborn).
"""
import os
import sqlite3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# 1. CONFIGURATION & SYNTHETIC PROCUREMENT DATA
# =====================================================

np.random.seed(42)

N_POS = 300
N_SUPPLIERS = 10

categories = [
    "IT Equipment",
    "Office Supplies",
    "Logistics Services",
    "Maintenance",
    "Consulting",
]

df_suppliers = pd.DataFrame({
    "supplier_id": range(1, N_SUPPLIERS + 1),
    "supplier_name": [f"Supplier {i}" for i in range(1, N_SUPPLIERS + 1)],
})

df_categories = pd.DataFrame({
    "category_id": range(1, len(categories) + 1),
    "category_name": categories,
})

# Each supplier has its own delivery behavior (avg days vs. promise).
# Negative = tends to deliver early, positive = tends to deliver late.
supplier_bias = {1: -4, 2: -3, 3: -2, 4: -1, 5: 0,
                 6: 1, 7: 2, 8: 3, 9: 4, 10: 6}

supplier_ids = np.random.randint(1, N_SUPPLIERS + 1, N_POS)
promised = np.random.randint(7, 36, N_POS)
noise = np.random.normal(0, 2.5, N_POS)
bias = np.array([supplier_bias[s] for s in supplier_ids])
actual = np.maximum(1, np.round(promised + bias + noise)).astype(int)

df_pos = pd.DataFrame({
    "po_id": range(1, N_POS + 1),
    "po_number": [f"PO-{1000 + i}" for i in range(N_POS)],
    "supplier_id": supplier_ids,
    "order_date": pd.date_range(
        "2025-01-01", periods=N_POS, freq="D"
    ).strftime("%Y-%m-%d"),
    "actual_lead_time": actual,
    "promised_lead_time": promised,
})

# Purchase order lines: 1-3 lines per PO
lines = []
for po_id in df_pos["po_id"]:
    for _ in range(np.random.randint(1, 4)):
        lines.append({
            "po_id": po_id,
            "category_id": np.random.randint(1, len(categories) + 1),
            "quantity": np.random.randint(5, 200),
            "unit_price": round(np.random.uniform(20, 1500), 2),
        })

df_lines = pd.DataFrame(lines)
df_lines.insert(0, "line_id", range(1, len(df_lines) + 1))

# Introduce missing prices for data-quality analysis
missing_idx = df_lines.sample(frac=0.05).index
df_lines.loc[missing_idx, "unit_price"] = np.nan

# =====================================================
# 2. RELATIONAL DATABASE - SQLITE
# =====================================================

conn = sqlite3.connect(":memory:")
conn.execute("PRAGMA foreign_keys = ON")

conn.executescript("""
CREATE TABLE suppliers (
    supplier_id INTEGER PRIMARY KEY,
    supplier_name TEXT NOT NULL UNIQUE
);

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL UNIQUE
);

CREATE TABLE purchase_orders (
    po_id INTEGER PRIMARY KEY,
    po_number TEXT NOT NULL UNIQUE,
    supplier_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    actual_lead_time INTEGER NOT NULL,
    promised_lead_time INTEGER NOT NULL,
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
);

CREATE TABLE po_lines (
    line_id INTEGER PRIMARY KEY,
    po_id INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL,
    FOREIGN KEY (po_id) REFERENCES purchase_orders(po_id),
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
""")

df_suppliers.to_sql("suppliers", conn, if_exists="append", index=False)
df_categories.to_sql("categories", conn, if_exists="append", index=False)
df_pos.to_sql("purchase_orders", conn, if_exists="append", index=False)
df_lines.to_sql("po_lines", conn, if_exists="append", index=False)

# =====================================================
# 3. SQL ANALYTICS - SPEND & SUPPLIER CONCENTRATION
# =====================================================

spend_query = """
WITH LineSpend AS (
    SELECT
        c.category_name,
        s.supplier_name,
        po.po_id,
        pl.line_id,
        pl.quantity * pl.unit_price AS line_spend,
        pl.unit_price
    FROM po_lines pl
    JOIN purchase_orders po ON pl.po_id = po.po_id
    JOIN suppliers s ON po.supplier_id = s.supplier_id
    JOIN categories c ON pl.category_id = c.category_id
),

SupplierCategorySpend AS (
    SELECT
        category_name,
        supplier_name,
        COUNT(DISTINCT po_id) AS total_pos,
        COUNT(line_id) AS total_lines,
        SUM(line_spend) AS known_spend,
        SUM(CASE WHEN unit_price IS NULL THEN 1 ELSE 0 END)
            AS missing_price_lines
    FROM LineSpend
    GROUP BY category_name, supplier_name
),

CategoryTotals AS (
    SELECT category_name, SUM(known_spend) AS category_spend
    FROM SupplierCategorySpend
    GROUP BY category_name
),

RankedSuppliers AS (
    SELECT
        sc.*,
        ct.category_spend,

        DENSE_RANK() OVER (
            PARTITION BY sc.category_name
            ORDER BY sc.known_spend DESC
        ) AS spend_rank,

        ROUND(100.0 * sc.known_spend /
              NULLIF(ct.category_spend, 0), 2) AS spend_share_pct,

        ROUND(
            100.0 * SUM(sc.known_spend) OVER (
                PARTITION BY sc.category_name
                ORDER BY sc.known_spend DESC
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) / NULLIF(ct.category_spend, 0), 2
        ) AS cumulative_spend_pct

    FROM SupplierCategorySpend sc
    JOIN CategoryTotals ct ON sc.category_name = ct.category_name
)

SELECT * FROM RankedSuppliers
ORDER BY category_name, spend_rank;
"""

df_spend = pd.read_sql_query(spend_query, conn)

print("\n--- SUPPLIER SPEND ANALYTICS ---")
print(df_spend.head(15).to_string(index=False))

# =====================================================
# 4. MONTHLY SPEND TREND - LAG & MOVING AVERAGE
# =====================================================

monthly_query = """
WITH MonthlySpend AS (
    SELECT
        STRFTIME('%Y-%m', po.order_date) AS order_month,
        SUM(pl.quantity * pl.unit_price) AS monthly_spend,
        COUNT(DISTINCT po.po_id) AS total_pos
    FROM po_lines pl
    JOIN purchase_orders po ON pl.po_id = po.po_id
    GROUP BY STRFTIME('%Y-%m', po.order_date)
)
SELECT
    order_month,
    total_pos,
    ROUND(monthly_spend, 2) AS monthly_spend,
    ROUND(LAG(monthly_spend) OVER (ORDER BY order_month), 2)
        AS prev_month_spend,
    ROUND(100.0 * (monthly_spend - LAG(monthly_spend)
        OVER (ORDER BY order_month)) /
        NULLIF(LAG(monthly_spend) OVER (ORDER BY order_month), 0), 2)
        AS mom_change_pct,
    ROUND(AVG(monthly_spend) OVER (
        ORDER BY order_month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2)
        AS moving_avg_3m
FROM MonthlySpend
ORDER BY order_month;
"""

df_monthly = pd.read_sql_query(monthly_query, conn)

print("\n--- MONTHLY SPEND TREND ---")
print(df_monthly.to_string(index=False))

# =====================================================
# 5. SUPPLIER DELIVERY PERFORMANCE
# =====================================================

delivery_query = """
WITH SupplierDelivery AS (
    SELECT
        s.supplier_name,
        COUNT(po.po_id) AS total_pos,
        ROUND(AVG(po.actual_lead_time), 2) AS avg_actual_lead_time,
        ROUND(AVG(po.promised_lead_time), 2) AS avg_promised_lead_time,
        SUM(CASE WHEN po.actual_lead_time > po.promised_lead_time
                 THEN 1 ELSE 0 END) AS late_pos
    FROM purchase_orders po
    JOIN suppliers s ON po.supplier_id = s.supplier_id
    GROUP BY s.supplier_name
)
SELECT *,
    ROUND(100.0 * (total_pos - late_pos) /
          NULLIF(total_pos, 0), 2) AS on_time_delivery_pct,
    ROUND(avg_actual_lead_time - avg_promised_lead_time, 2)
        AS avg_delay_days
FROM SupplierDelivery
ORDER BY on_time_delivery_pct DESC;
"""

df_delivery = pd.read_sql_query(delivery_query, conn)

print("\n--- SUPPLIER DELIVERY PERFORMANCE ---")
print(df_delivery.to_string(index=False))

# =====================================================
# 6. DATA QUALITY & TRANSFORMATION
# =====================================================

quality_query = """
SELECT
    COUNT(*) AS total_lines,
    SUM(CASE WHEN unit_price IS NULL THEN 1 ELSE 0 END) AS missing_prices,
    ROUND(100.0 * SUM(CASE WHEN unit_price IS NOT NULL THEN 1 ELSE 0 END)
          / COUNT(*), 2) AS price_completeness_pct
FROM po_lines;
"""

df_quality = pd.read_sql_query(quality_query, conn)

print("\n--- DATA QUALITY ---")
print(df_quality.to_string(index=False))

# PO-level dataset (one row per PO) for delivery analysis.
# Using line-level data here would double-count POs with 2-3 lines.
df_po_level = pd.read_sql_query("""
SELECT
    po.po_number,
    po.order_date,
    s.supplier_name,
    po.actual_lead_time,
    po.promised_lead_time
FROM purchase_orders po
JOIN suppliers s ON po.supplier_id = s.supplier_id
""", conn)

df_po_level["delivery_status"] = np.where(
    df_po_level["actual_lead_time"] > df_po_level["promised_lead_time"],
    "Late", "On Time",
)

# =====================================================
# 7. BUSINESS INTELLIGENCE & VISUALIZATION
# =====================================================

sns.set_theme(style="whitegrid", context="notebook")
supplier_order = sorted(df_delivery["supplier_name"],
                        key=lambda x: int(x.split()[-1]))

# ---- Dashboard (2 x 3) ----
fig, axes = plt.subplots(2, 3, figsize=(21, 13))

# Chart 1: Known spend by category
cat_spend = (df_spend.groupby("category_name", as_index=False)
             ["known_spend"].sum()
             .sort_values("known_spend", ascending=False))
sns.barplot(data=cat_spend, x="category_name", y="known_spend",
            ax=axes[0, 0])
axes[0, 0].set_title("Procurement Spend by Category", fontweight="bold")
axes[0, 0].set_xlabel("Procurement Category")
axes[0, 0].set_ylabel("Known Spend (USD)")
axes[0, 0].tick_params(axis="x", rotation=40)

# Chart 2: Lead time variability (PO level)
sns.boxplot(data=df_po_level, x="supplier_name", y="actual_lead_time",
            hue="delivery_status", order=supplier_order, ax=axes[0, 1])
axes[0, 1].set_title("Supplier Lead Time & Delivery Risk (PO level)",
                     fontweight="bold")
axes[0, 1].set_xlabel("Supplier")
axes[0, 1].set_ylabel("Actual Lead Time (Days)")
axes[0, 1].tick_params(axis="x", rotation=45)

# Chart 3: On-time delivery vs. target
sns.barplot(data=df_delivery, x="supplier_name",
            y="on_time_delivery_pct", order=supplier_order,
            ax=axes[0, 2])
axes[0, 2].axhline(95, linestyle="--", color="red", label="95% Target")
axes[0, 2].set_title("Supplier On-Time Delivery Performance",
                     fontweight="bold")
axes[0, 2].set_xlabel("Supplier")
axes[0, 2].set_ylabel("On-Time Delivery (%)")
axes[0, 2].tick_params(axis="x", rotation=45)
axes[0, 2].legend()

# Chart 4: Monthly spend + 3-month moving average
axes[1, 0].bar(df_monthly["order_month"], df_monthly["monthly_spend"],
               alpha=0.6, label="Monthly spend")
axes[1, 0].plot(df_monthly["order_month"], df_monthly["moving_avg_3m"],
                color="black", marker="o", label="3-month moving avg")
axes[1, 0].set_title("Monthly Spend & 3-Month Moving Average",
                     fontweight="bold")
axes[1, 0].set_xlabel("Month")
axes[1, 0].set_ylabel("Known Spend (USD)")
axes[1, 0].tick_params(axis="x", rotation=45)
axes[1, 0].legend()

# Chart 5: Month-over-month change (LAG)
mom = df_monthly.dropna(subset=["mom_change_pct"])
colors = ["#2a9d8f" if v >= 0 else "#e76f51"
          for v in mom["mom_change_pct"]]
axes[1, 1].bar(mom["order_month"], mom["mom_change_pct"], color=colors)
axes[1, 1].axhline(0, color="black", linewidth=0.8)
axes[1, 1].set_title("Month-over-Month Spend Change (%)",
                     fontweight="bold")
axes[1, 1].set_xlabel("Month")
axes[1, 1].set_ylabel("Change vs. Previous Month (%)")
axes[1, 1].tick_params(axis="x", rotation=45)

# Chart 6: Average delay (actual - promised)
sns.barplot(data=df_delivery, x="supplier_name", y="avg_delay_days",
            order=supplier_order, ax=axes[1, 2])
axes[1, 2].axhline(0, color="black", linewidth=0.8)
axes[1, 2].set_title("Average Delay vs. Promise (Days)", fontweight="bold")
axes[1, 2].set_xlabel("Supplier")
axes[1, 2].set_ylabel("Actual - Promised Lead Time (Days)")
axes[1, 2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/procurement_intelligence_dashboard.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

# ---- Pareto per category (uses cumulative_spend_pct) ----
fig, axes = plt.subplots(2, 3, figsize=(21, 11))
axes = axes.flatten()

for ax, (cat, g) in zip(axes, df_spend.groupby("category_name")):
    g = g.sort_values("spend_rank")
    ax.bar(g["supplier_name"], g["spend_share_pct"], alpha=0.7)
    ax.set_ylabel("Spend Share (%)")
    ax.tick_params(axis="x", rotation=60)
    ax2 = ax.twinx()
    ax2.plot(g["supplier_name"], g["cumulative_spend_pct"],
             color="black", marker="o")
    ax2.axhline(80, linestyle="--", color="red")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative Spend (%)")
    ax2.grid(False)
    ax.set_title(f"Pareto: {cat}", fontweight="bold")

axes[-1].axis("off")
axes[-1].text(0.05, 0.5,
              "Bars: supplier share of category spend\n"
              "Line: cumulative share\n"
              "Red dashed line: 80% threshold",
              fontsize=13, va="center")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/supplier_pareto_by_category.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

# =====================================================
# 8. EXPORT ANALYTICAL DATASETS
# =====================================================

df_spend.to_csv(f"{OUTPUT_DIR}/supplier_spend_analytics.csv", index=False)
df_monthly.to_csv(f"{OUTPUT_DIR}/monthly_spend_trend.csv", index=False)
df_delivery.to_csv(f"{OUTPUT_DIR}/supplier_delivery_performance.csv",
                   index=False)
df_quality.to_csv(f"{OUTPUT_DIR}/data_quality_metrics.csv", index=False)

conn.close()
print("\nProcurement Intelligence Analysis Completed.")
