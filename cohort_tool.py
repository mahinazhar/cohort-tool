import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CSV_PATH = "Online_Retail.csv"

# Chart chrome (light mode), from the dataviz skill's reference palette
CHART_SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
STATUS_CRITICAL = "#d03b3b"

# Sequential single-hue (blue) ramp, steps 250->700, kept off the lightest
# steps so every line stays >=2:1 against the light surface (ordinal rule)
SEQUENTIAL_BLUE_STOPS = [
    "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b",
]


def cohort_colors(n):
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE_STOPS)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def style_axes(ax):
    ax.set_facecolor(CHART_SURFACE)
    ax.figure.set_facecolor(CHART_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_LINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

con = duckdb.connect()

con.execute(f"""
    CREATE OR REPLACE TABLE raw_retail AS
    SELECT
        *,
        strptime(InvoiceDate, '%-m/%-d/%y %-H:%M') AS InvoiceTimestamp,
        Quantity * UnitPrice AS Revenue
    FROM read_csv_auto('{CSV_PATH}', sample_size=-1, encoding='latin-1')
""")

raw_count = con.execute("SELECT COUNT(*) FROM raw_retail").fetchone()[0]

# Step 1: drop rows with missing CustomerID, track rows + revenue excluded
missing_customer_count, missing_customer_revenue = con.execute("""
    SELECT COUNT(*), COALESCE(SUM(Revenue), 0)
    FROM raw_retail
    WHERE CustomerID IS NULL
""").fetchone()

con.execute("""
    CREATE OR REPLACE TABLE step1_has_customer AS
    SELECT * FROM raw_retail WHERE CustomerID IS NOT NULL
""")

# Step 2: drop exact duplicate rows (every original column identical)
original_columns = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]
partition_cols = ", ".join(original_columns)

con.execute(f"""
    CREATE OR REPLACE TABLE step2_deduped AS
    SELECT * EXCLUDE (rn) FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY {partition_cols}) AS rn
        FROM step1_has_customer
    )
    WHERE rn = 1
""")

step1_count = con.execute("SELECT COUNT(*) FROM step1_has_customer").fetchone()[0]
step2_count = con.execute("SELECT COUNT(*) FROM step2_deduped").fetchone()[0]
duplicate_count = step1_count - step2_count

# Step 3: drop negative quantities
negative_qty_count = con.execute(
    "SELECT COUNT(*) FROM step2_deduped WHERE Quantity < 0"
).fetchone()[0]

con.execute("""
    CREATE OR REPLACE TABLE step3_positive_qty AS
    SELECT * FROM step2_deduped WHERE Quantity >= 0
""")

# Step 4: drop non-positive unit price
non_positive_price_count = con.execute(
    "SELECT COUNT(*) FROM step3_positive_qty WHERE UnitPrice <= 0"
).fetchone()[0]

con.execute("""
    CREATE OR REPLACE TABLE clean_retail AS
    SELECT
        *,
        date_trunc('month', InvoiceTimestamp) AS InvoiceMonth,
        strftime(date_trunc('month', InvoiceTimestamp), '%Y-%m') AS InvoiceMonthStr
    FROM step3_positive_qty
    WHERE UnitPrice > 0
""")

con.execute("DROP TABLE step1_has_customer")
con.execute("DROP TABLE step2_deduped")
con.execute("DROP TABLE step3_positive_qty")

final_row_count = con.execute("SELECT COUNT(*) FROM clean_retail").fetchone()[0]
final_customer_count = con.execute(
    "SELECT COUNT(DISTINCT CustomerID) FROM clean_retail"
).fetchone()[0]

print("Data cleaning summary")
print("=" * 40)
print(f"Starting row count:                 {raw_count:>8}")
print(f"Removed - missing CustomerID:        {missing_customer_count:>8}  (${missing_customer_revenue:,.2f} revenue excluded)")
print(f"Removed - duplicate rows:            {duplicate_count:>8}")
print(f"Removed - negative quantity:          {negative_qty_count:>8}")
print(f"Removed - unit price <= 0:            {non_positive_price_count:>8}")
print("-" * 40)
print(f"Remaining rows:                       {final_row_count:>8}")
print(f"Remaining unique CustomerIDs:          {final_customer_count:>8}")

# Cohort assignment: each customer's cohort = month of their first purchase
con.execute("""
    CREATE OR REPLACE TABLE customer_cohort AS
    SELECT
        CustomerID,
        MIN(InvoiceMonth) AS CohortMonth,
        strftime(MIN(InvoiceMonth), '%Y-%m') AS CohortMonthStr
    FROM clean_retail
    GROUP BY CustomerID
""")

cohort_customer_count = con.execute("SELECT COUNT(*) FROM customer_cohort").fetchone()[0]

print("\nCohort assignment")
print("=" * 40)
print(f"Customers assigned a cohort: {cohort_customer_count}")

print("\nSample CustomerID -> CohortMonth:")
sample = con.execute(
    "SELECT CustomerID, CohortMonthStr AS CohortMonth FROM customer_cohort "
    "ORDER BY CohortMonth, CustomerID LIMIT 10"
).fetchdf()
print(sample.to_string(index=False))

print("\nCustomers per cohort month:")
cohort_sizes = con.execute("""
    SELECT CohortMonthStr AS CohortMonth, COUNT(*) AS NumCustomers
    FROM customer_cohort
    GROUP BY CohortMonth, CohortMonthStr
    ORDER BY CohortMonth
""").fetchdf()
print(cohort_sizes.to_string(index=False))

# Period index: months elapsed between the transaction's month and the
# customer's cohort month (CohortMonth = period 0)
con.execute("""
    CREATE OR REPLACE TABLE cohort_transactions AS
    SELECT
        cc.CohortMonthStr AS CohortMonth,
        date_diff('month', cc.CohortMonth, r.InvoiceMonth) AS PeriodIndex,
        r.CustomerID,
        r.Revenue
    FROM clean_retail r
    JOIN customer_cohort cc ON r.CustomerID = cc.CustomerID
""")

con.execute("""
    CREATE OR REPLACE TABLE cohort_summary AS
    SELECT
        CohortMonth,
        PeriodIndex,
        COUNT(DISTINCT CustomerID) AS NumCustomers,
        SUM(Revenue) AS Revenue
    FROM cohort_transactions
    GROUP BY CohortMonth, PeriodIndex
    ORDER BY CohortMonth, PeriodIndex
""")

cohort_summary_df = con.execute("SELECT * FROM cohort_summary").fetchdf()

customer_matrix = cohort_summary_df.pivot(
    index="CohortMonth", columns="PeriodIndex", values="NumCustomers"
).fillna(0).astype(int)

revenue_matrix = cohort_summary_df.pivot(
    index="CohortMonth", columns="PeriodIndex", values="Revenue"
).fillna(0).round(2)

print("\nCohort matrix - distinct customers by period:")
print(customer_matrix.to_string())

print("\nCohort matrix - revenue by period:")
print(revenue_matrix.to_string())

# Retention % = each period's value relative to that cohort's period 0 value
customer_retention = customer_matrix.div(customer_matrix[0], axis=0) * 100

print("\nCohort retention % - distinct customers (period 0 = 100%):")
print(customer_retention.round(1).to_string())

# Cumulative revenue per customer = running total of revenue per period,
# divided by the cohort's original (period 0) customer count
cumulative_revenue_per_customer = (
    revenue_matrix.cumsum(axis=1).div(customer_matrix[0], axis=0)
).round(2)

print("\nCohort matrix - cumulative revenue per customer by period:")
print(cumulative_revenue_per_customer.to_string())


def payback_period(cac, revenue_per_customer=cumulative_revenue_per_customer):
    """For each cohort, find the first period where cumulative revenue per
    customer reaches the given CAC. Returns {CohortMonth: period or None}."""
    results = {}
    for cohort, row in revenue_per_customer.iterrows():
        reached = row[row >= cac]
        results[cohort] = int(reached.index[0]) if not reached.empty else None
    return results


avg_order_value = con.execute("""
    SELECT AVG(Revenue) FROM (
        SELECT InvoiceNo, SUM(Revenue) AS Revenue FROM clean_retail GROUP BY InvoiceNo
    )
""").fetchone()[0]
avg_revenue_per_customer = con.execute(
    "SELECT SUM(Revenue) / COUNT(DISTINCT CustomerID) FROM clean_retail"
).fetchone()[0]

print(f"\nSuggested CAC to test: ~$500 (avg order value is ${avg_order_value:,.2f}; "
      f"avg revenue per customer over the full window is ${avg_revenue_per_customer:,.2f}, "
      f"so $500 gives a ~4:1 LTV:CAC ratio while still being high enough that some "
      f"cohorts won't pay back within the observed window)")

cac_input = input("\nEnter an assumed CAC ($) to compute payback period per cohort: ").strip()
cac = float(cac_input) if cac_input else 500.0
if cac_input:
    payback = payback_period(cac)
    print(f"\nPayback period per cohort (CAC = ${cac:,.2f}):")
    for cohort, period in payback.items():
        label = f"Period {period}" if period is not None else "Not recovered within observed window"
        print(f"  {cohort}: {label}")

# --- Charts ---------------------------------------------------------------

cohorts = list(customer_retention.index)
colors = cohort_colors(len(cohorts))

fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
style_axes(ax)
for cohort, color in zip(cohorts, colors):
    row = customer_retention.loc[cohort]
    valid = row[row > 0].index.tolist() or [0]
    last_period = max(valid)
    ax.plot(
        customer_retention.columns[: last_period + 1],
        row.iloc[: last_period + 1],
        color=color, linewidth=1.8, solid_capstyle="round", label=cohort,
    )
ax.set_title("Customer retention by cohort", color=INK_PRIMARY, fontsize=13, pad=12)
ax.set_xlabel("Period (months since first purchase)", color=INK_SECONDARY, fontsize=10)
ax.set_ylabel("Retention (% of cohort's period-0 customers)", color=INK_SECONDARY, fontsize=10)
ax.set_xticks(customer_retention.columns)
ax.set_ylim(0, None)
legend = ax.legend(
    title="Cohort", loc="upper left", bbox_to_anchor=(1.02, 1.0),
    fontsize=8, title_fontsize=9, frameon=False, ncol=1,
)
plt.setp(legend.get_texts(), color=INK_SECONDARY)
legend.get_title().set_color(INK_PRIMARY)
fig.text(
    0.01, 0.01,
    "Each cohort's final point may reflect a partial month — data ends 2011-12-09.",
    color=INK_MUTED, fontsize=8, style="italic",
)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("retention_curves.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
style_axes(ax)
for cohort, color in zip(cohorts, colors):
    row = cumulative_revenue_per_customer.loc[cohort]
    active = customer_retention.loc[cohort]
    last_period = max(active[active > 0].index.tolist() or [0])
    ax.plot(
        cumulative_revenue_per_customer.columns[: last_period + 1],
        row.iloc[: last_period + 1],
        color=color, linewidth=1.8, solid_capstyle="round", label=cohort,
    )
ax.axhline(cac, color=STATUS_CRITICAL, linewidth=1.5, linestyle="--", zorder=3)
ax.text(
    ax.get_xlim()[1], cac, f"  CAC = ${cac:,.0f}", color=STATUS_CRITICAL,
    fontsize=9, va="center", ha="left", fontweight="bold",
)
ax.set_title("Cumulative revenue per customer vs. CAC", color=INK_PRIMARY, fontsize=13, pad=12)
ax.set_xlabel("Period (months since first purchase)", color=INK_SECONDARY, fontsize=10)
ax.set_ylabel("Cumulative revenue per customer ($)", color=INK_SECONDARY, fontsize=10)
ax.set_xticks(cumulative_revenue_per_customer.columns)
ax.set_ylim(0, None)
legend = ax.legend(
    title="Cohort", loc="upper left", bbox_to_anchor=(1.02, 1.0),
    fontsize=8, title_fontsize=9, frameon=False, ncol=1,
)
plt.setp(legend.get_texts(), color=INK_SECONDARY)
legend.get_title().set_color(INK_PRIMARY)
fig.text(
    0.01, 0.01,
    "Each cohort's final point may reflect a partial month — data ends 2011-12-09.",
    color=INK_MUTED, fontsize=8, style="italic",
)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("cumulative_revenue_cac.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

print("\nSaved charts: retention_curves.png, cumulative_revenue_cac.png")
