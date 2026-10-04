"""Core cleaning / cohort / charting logic shared by the Streamlit app.

Pipeline: load CSV -> clean (drop missing CustomerID, exact duplicates,
unparseable dates, non-positive unit price; net cancellations against the
purchases they reverse) -> assign each customer a cohort (first purchase
month) -> compute each transaction's period index (months since the
customer's cohort month) -> aggregate into cohort matrices (customer
count, revenue, retention %, cumulative revenue per customer) -> render
heatmaps/line charts. A trailing partial calendar month (if the data
doesn't reach that month's last day) is excluded from the cohort analysis
so it doesn't understate retention/revenue for every cohort still active.
"""

import io

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REQUIRED_COLUMNS = ["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]

CURRENCY_SYMBOL = "£"  # GBP

CANDIDATE_DATE_FORMATS = [
    "%m/%d/%y %H:%M",
    "%m/%d/%Y %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M",
    "%d/%m/%y %H:%M",
    "%m/%d/%Y",
    "%d/%m/%Y",
]

# Chart chrome, from the dataviz skill's reference palette (light mode)
CHART_SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
STATUS_CRITICAL = "#d03b3b"
SEQUENTIAL_BLUE_STOPS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]


def sequential_cmap():
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE_STOPS)
    cmap.set_bad(color=GRIDLINE)
    return cmap


def cohort_line_colors(n):
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


# --------------------------------------------------------------------------
# Loading & cleaning
# --------------------------------------------------------------------------

def load_csv(file_bytes: bytes) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not decode CSV as utf-8 or latin-1: {last_error}")


def validate_columns(df: pd.DataFrame):
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def parse_dates(series: pd.Series) -> pd.Series:
    best, best_valid = None, -1
    for fmt in CANDIDATE_DATE_FORMATS:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        valid = int(parsed.notna().sum())
        if valid > best_valid:
            best, best_valid = parsed, valid
        if valid == len(series):
            return best
    if best_valid < len(series) * 0.95:
        fallback = pd.to_datetime(series, errors="coerce")
        if int(fallback.notna().sum()) > best_valid:
            return fallback
    return best


def net_out_cancellations(df: pd.DataFrame):
    """Nets cancellations (negative-Quantity rows) against the customer's
    prior purchases of the same product, so a cancelled sale contributes
    zero net revenue instead of the return being dropped while the
    original sale it reversed still counts in full.

    Requires a `StockCode` column to match a cancellation to the product
    it reversed; without one, cancellations are just dropped (old
    behaviour) and this is reported via `cancellation_netting_applied`.

    Processes each (CustomerID, StockCode) group in chronological order
    and matches each cancellation to its most recent *prior* purchase
    first (LIFO), falling back to older purchases, consuming whole or
    partial quantities until the cancelled quantity is covered. A
    purchase dated after the cancellation is never used as a match. Any
    leftover (no prior purchase found at all) is reported as
    `cancellation_unmatched_quantity`.
    """
    stats = {
        "cancellation_rows_removed": int((df["Quantity"] < 0).sum()),
        "cancellation_netting_applied": "StockCode" in df.columns,
        "cancellation_netting_rows_removed": 0,
        "cancellation_netting_revenue_removed": 0.0,
        "cancellation_unmatched_quantity": 0.0,
    }

    if "StockCode" not in df.columns:
        stats["cancellation_unmatched_quantity"] = float(-df.loc[df["Quantity"] < 0, "Quantity"].sum())
        return df, stats

    df = df.reset_index(drop=True)
    qty = df["Quantity"].to_numpy(dtype=float)
    new_qty = qty.copy()
    ts = df["InvoiceTimestamp"].to_numpy()
    unit_price = df["UnitPrice"].to_numpy(dtype=float)

    groups = df.groupby(["CustomerID", "StockCode"], sort=False).indices
    unmatched_qty = 0.0

    for _, idx in groups.items():
        g_qty = qty[idx]
        if not (g_qty < 0).any():
            continue
        g_ts = ts[idx]
        time_order = np.argsort(g_ts.astype("int64"), kind="stable")
        remaining = g_qty.copy()

        stack = []  # local positions with remaining qty > 0, oldest-pushed-first
        for li in time_order:
            if g_qty[li] > 0:
                stack.append(li)
            elif g_qty[li] < 0:
                needed = -g_qty[li]
                while needed > 0 and stack:
                    top = stack[-1]
                    consume = min(remaining[top], needed)
                    remaining[top] -= consume
                    needed -= consume
                    if remaining[top] == 0:
                        stack.pop()
                if needed > 0:
                    unmatched_qty += needed
                remaining[li] = 0.0

        new_qty[idx] = remaining

    removed_revenue = float(((qty - new_qty) * unit_price)[qty > 0].sum())
    fully_dropped_positive_rows = int(((qty > 0) & (new_qty == 0)).sum())

    df = df.copy()
    df["Quantity"] = new_qty
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    stats["cancellation_netting_rows_removed"] = fully_dropped_positive_rows
    stats["cancellation_netting_revenue_removed"] = removed_revenue
    stats["cancellation_unmatched_quantity"] = float(unmatched_qty)
    return df, stats


def clean_and_assign_cohorts(df_raw: pd.DataFrame):
    """Returns (transactions_df, summary_dict).

    transactions_df has every original column plus Revenue, CohortMonth
    ('YYYY-MM') and PeriodIndex, for rows that survived cleaning.
    """
    original_columns = list(df_raw.columns)

    df = df_raw.copy()
    df["InvoiceTimestamp"] = parse_dates(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    con = duckdb.connect()
    con.register("raw_retail", df)

    raw_count = con.execute("SELECT COUNT(*) FROM raw_retail").fetchone()[0]

    unparseable_date_count = con.execute(
        "SELECT COUNT(*) FROM raw_retail WHERE InvoiceTimestamp IS NULL"
    ).fetchone()[0]
    con.execute("""
        CREATE OR REPLACE TABLE step0_has_date AS
        SELECT * FROM raw_retail WHERE InvoiceTimestamp IS NOT NULL
    """)

    missing_customer_count, missing_customer_revenue = con.execute("""
        SELECT COUNT(*), COALESCE(SUM(Revenue), 0)
        FROM step0_has_date WHERE CustomerID IS NULL
    """).fetchone()
    con.execute("""
        CREATE OR REPLACE TABLE step1_has_customer AS
        SELECT * FROM step0_has_date WHERE CustomerID IS NOT NULL
    """)

    partition_cols = ", ".join(f'"{c}"' for c in original_columns)
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

    deduped_df = con.execute("SELECT * FROM step2_deduped").fetchdf()
    netted_df, netting_stats = net_out_cancellations(deduped_df)
    con.register("step3_netted", netted_df)

    con.execute("""
        CREATE OR REPLACE TABLE step4_positive_qty AS
        SELECT * FROM step3_netted WHERE Quantity > 0
    """)

    non_positive_price_count = con.execute(
        "SELECT COUNT(*) FROM step4_positive_qty WHERE UnitPrice <= 0"
    ).fetchone()[0]

    con.execute("""
        CREATE OR REPLACE TABLE clean_retail AS
        SELECT *, date_trunc('month', InvoiceTimestamp) AS InvoiceMonth
        FROM step4_positive_qty
        WHERE UnitPrice > 0
    """)

    con.execute("""
        CREATE OR REPLACE TABLE customer_cohort AS
        SELECT CustomerID, MIN(InvoiceMonth) AS CohortMonth
        FROM clean_retail
        GROUP BY CustomerID
    """)

    select_original = ", ".join(f'r."{c}"' for c in original_columns)
    transactions_df = con.execute(f"""
        SELECT
            {select_original},
            r.Revenue AS Revenue,
            strftime(cc.CohortMonth, '%Y-%m') AS CohortMonth,
            date_diff('month', cc.CohortMonth, r.InvoiceMonth) AS PeriodIndex
        FROM clean_retail r
        JOIN customer_cohort cc ON r.CustomerID = cc.CustomerID
        ORDER BY cc.CohortMonth, r.CustomerID, r.InvoiceTimestamp
    """).fetchdf()

    max_invoice_row = con.execute("SELECT MAX(InvoiceTimestamp) FROM clean_retail").fetchone()[0]
    last_full_month, last_month_is_partial = _last_full_month(max_invoice_row)

    final_row_count = len(transactions_df)
    final_customer_count = transactions_df["CustomerID"].nunique()

    summary = {
        "raw_count": int(raw_count),
        "unparseable_date_count": int(unparseable_date_count),
        "missing_customer_count": int(missing_customer_count),
        "missing_customer_revenue": float(missing_customer_revenue),
        "duplicate_count": int(duplicate_count),
        "non_positive_price_count": int(non_positive_price_count),
        "final_row_count": int(final_row_count),
        "final_customer_count": int(final_customer_count),
        "last_full_month": last_full_month,
        "last_month_is_partial": last_month_is_partial,
        **netting_stats,
    }

    con.close()
    return transactions_df, summary


def _last_full_month(max_invoice_timestamp):
    """Returns (last_full_month 'YYYY-MM', is_partial) for the given max
    timestamp in the cleaned data. If the data's last calendar month
    doesn't reach that month's final day, it's treated as partial and the
    prior month is returned instead."""
    if max_invoice_timestamp is None:
        return None, False
    max_ts = pd.Timestamp(max_invoice_timestamp)
    last_day_of_month = (max_ts + pd.offsets.MonthEnd(0)).day
    is_partial = max_ts.day < last_day_of_month
    if is_partial:
        last_full = (max_ts.replace(day=1) - pd.Timedelta(days=1))
    else:
        last_full = max_ts
    return last_full.strftime("%Y-%m"), is_partial


# --------------------------------------------------------------------------
# Cohort matrices
# --------------------------------------------------------------------------

def build_matrices(transactions_df: pd.DataFrame, last_full_month: str):
    """Builds the 4 cohort matrices. Cells beyond a cohort's observed
    horizon (it hasn't existed that many months yet, or the period falls
    after `last_full_month`) are NaN; cells within the horizon with no
    activity are 0. A cohort whose first purchase falls entirely after
    `last_full_month` (e.g. a partial trailing month) gets an all-NaN row
    across every matrix, so it reads as excluded/greyed-out rather than
    "no activity"."""
    con = duckdb.connect()
    con.register("tx", transactions_df)

    cohort_summary = con.execute("""
        SELECT CohortMonth, PeriodIndex,
               COUNT(DISTINCT CustomerID) AS NumCustomers,
               SUM(Revenue) AS Revenue
        FROM tx
        GROUP BY CohortMonth, PeriodIndex
        ORDER BY CohortMonth, PeriodIndex
    """).fetchdf()

    horizon = con.execute(f"""
        SELECT CohortMonth, date_diff(
            'month', strptime(CohortMonth, '%Y-%m'), strptime('{last_full_month}', '%Y-%m')
        ) AS MaxValidPeriod
        FROM (SELECT DISTINCT CohortMonth FROM tx)
    """).fetchdf().set_index("CohortMonth")["MaxValidPeriod"]
    con.close()

    max_valid_period = max(int(horizon.max()), 0)
    max_period = min(int(cohort_summary["PeriodIndex"].max()), max_valid_period)
    all_periods = list(range(0, max_period + 1))
    cohorts = sorted(horizon.index)

    def pivot_masked(value_col):
        raw = cohort_summary.pivot(index="CohortMonth", columns="PeriodIndex", values=value_col)
        raw = raw.reindex(index=cohorts, columns=all_periods)
        for cohort in cohorts:
            max_p = int(horizon[cohort])
            in_range = [p for p in all_periods if p <= max_p]
            out_range = [p for p in all_periods if p > max_p]
            raw.loc[cohort, in_range] = raw.loc[cohort, in_range].fillna(0)
            raw.loc[cohort, out_range] = np.nan
        return raw

    customer_matrix = pivot_masked("NumCustomers")
    revenue_matrix = pivot_masked("Revenue").round(2)

    customer_retention = (customer_matrix.div(customer_matrix[0], axis=0) * 100).round(1)

    cumulative_revenue_per_customer = revenue_matrix.fillna(0).cumsum(axis=1).div(
        customer_matrix[0], axis=0
    )
    for cohort in cohorts:
        max_p = int(horizon[cohort])
        out_range = [p for p in all_periods if p > max_p]
        cumulative_revenue_per_customer.loc[cohort, out_range] = np.nan
    cumulative_revenue_per_customer = cumulative_revenue_per_customer.round(2)

    return {
        "customer_matrix": customer_matrix,
        "revenue_matrix": revenue_matrix,
        "customer_retention": customer_retention,
        "cumulative_revenue_per_customer": cumulative_revenue_per_customer,
    }


def apply_gross_margin(cumulative_revenue_per_customer: pd.DataFrame, margin_pct: float) -> pd.DataFrame:
    """Converts cumulative revenue per customer into cumulative gross
    profit per customer at the given margin assumption (0-100)."""
    return (cumulative_revenue_per_customer * (margin_pct / 100)).round(2)


def payback_periods(cumulative_value_per_customer: pd.DataFrame, cac: float):
    """For each cohort, the first period where the cumulative value
    (e.g. gross profit) per customer reaches the given CAC. None if it
    never does within the observed/included window; a cohort that's
    entirely excluded (all-NaN row) also returns None."""
    results = {}
    for cohort, row in cumulative_value_per_customer.iterrows():
        reached = row[row >= cac]
        results[cohort] = int(reached.index[0]) if not reached.empty else None
    return results


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_heatmap(matrix: pd.DataFrame, title: str, value_fmt, cbar_label: str):
    n_rows, n_cols = matrix.shape
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 0.8), max(5, n_rows * 0.5)), dpi=150)
    fig.patch.set_facecolor(CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    values = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = sequential_cmap()
    im = ax.imshow(masked, cmap=cmap, aspect="auto")

    vmin, vmax = np.nanmin(values), np.nanmax(values)
    span = (vmax - vmin) or 1.0
    for i in range(n_rows):
        for j in range(n_cols):
            val = values[i, j]
            if np.isnan(val):
                continue
            norm = (val - vmin) / span
            text_color = "#ffffff" if norm > 0.55 else INK_PRIMARY
            ax.text(j, i, value_fmt(val), ha="center", va="center",
                    color=text_color, fontsize=7)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(matrix.columns, color=INK_MUTED, fontsize=8)
    ax.set_yticks(range(n_rows))
    ytick_labels = [
        f"{cohort} (excluded)" if matrix.loc[cohort].isna().all() else str(cohort)
        for cohort in matrix.index
    ]
    ax.set_yticklabels(ytick_labels, color=INK_MUTED, fontsize=8)
    ax.set_xlabel("Period (months since first purchase)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Cohort", color=INK_SECONDARY, fontsize=10)
    ax.set_title(title, color=INK_PRIMARY, fontsize=13, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label, color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cbar.outline.set_visible(False)

    fig.text(0.01, 0.01,
              "Blank cells: cohort hasn't reached that period yet, or it's an excluded partial month.",
              color=INK_MUTED, fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def render_retention_chart(customer_retention: pd.DataFrame):
    cohorts = list(customer_retention.index)
    colors = cohort_line_colors(len(cohorts))

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    style_axes(ax)
    for cohort, color in zip(cohorts, colors):
        row = customer_retention.loc[cohort]
        if row.isna().all():
            continue
        ax.plot(customer_retention.columns, row,
                color=color, linewidth=1.8, solid_capstyle="round", label=cohort)

    ax.set_title("Customer retention by cohort", color=INK_PRIMARY, fontsize=13, pad=12)
    ax.set_xlabel("Period (months since first purchase)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Retention (% of cohort's period-0 customers)", color=INK_SECONDARY, fontsize=10)
    ax.set_xticks(customer_retention.columns)
    ax.set_ylim(0, None)
    legend = ax.legend(title="Cohort", loc="upper left", bbox_to_anchor=(1.02, 1.0),
                        fontsize=8, title_fontsize=9, frameon=False)
    plt.setp(legend.get_texts(), color=INK_SECONDARY)
    legend.get_title().set_color(INK_PRIMARY)
    fig.text(0.01, 0.01,
              "Each line stops at that cohort's last included period; excluded (partial-month) cohorts aren't plotted.",
              color=INK_MUTED, fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def render_cac_chart(cumulative_profit_per_customer: pd.DataFrame, cac: float):
    cohorts = list(cumulative_profit_per_customer.index)
    colors = cohort_line_colors(len(cohorts))

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    style_axes(ax)
    for cohort, color in zip(cohorts, colors):
        row = cumulative_profit_per_customer.loc[cohort]
        if row.isna().all():
            continue
        ax.plot(cumulative_profit_per_customer.columns, row,
                color=color, linewidth=1.8, solid_capstyle="round", label=cohort)

    ax.axhline(cac, color=STATUS_CRITICAL, linewidth=1.5, linestyle="--", zorder=3)
    ax.text(ax.get_xlim()[1], cac, f"  CAC = {CURRENCY_SYMBOL}{cac:,.0f}", color=STATUS_CRITICAL,
            fontsize=9, va="center", ha="left", fontweight="bold")

    ax.set_title("Cumulative gross profit per customer vs. CAC", color=INK_PRIMARY, fontsize=13, pad=12)
    ax.set_xlabel("Period (months since first purchase)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(f"Cumulative gross profit per customer ({CURRENCY_SYMBOL})", color=INK_SECONDARY, fontsize=10)
    ax.set_xticks(cumulative_profit_per_customer.columns)
    ax.set_ylim(0, None)
    legend = ax.legend(title="Cohort", loc="upper left", bbox_to_anchor=(1.02, 1.0),
                        fontsize=8, title_fontsize=9, frameon=False)
    plt.setp(legend.get_texts(), color=INK_SECONDARY)
    legend.get_title().set_color(INK_PRIMARY)
    fig.text(0.01, 0.01,
              "Each line stops at that cohort's last included period; excluded (partial-month) cohorts aren't plotted.",
              color=INK_MUTED, fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig
