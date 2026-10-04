import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

import cohort_core as cc

SAMPLE_DATA_PATH = Path(__file__).parent / "Online_Retail.csv"

st.set_page_config(page_title="Cohort & CAC Payback Analysis", layout="wide")

st.title("Cohort & CAC Payback Analysis")
st.write(
    "Upload a transaction-level CSV. Required columns: "
    f"`{'`, `'.join(cc.REQUIRED_COLUMNS)}`. Any other columns "
    "(e.g. `InvoiceNo`, `Description`, `Country`) are kept but not required "
    "— a `StockCode` column, if present, is used to net cancellations "
    "against the purchase they reversed."
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    st.session_state["use_sample"] = False
elif SAMPLE_DATA_PATH.exists() and st.button("Use sample dataset (Online Retail, UCI)"):
    st.session_state["use_sample"] = True

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
elif st.session_state.get("use_sample"):
    file_bytes = SAMPLE_DATA_PATH.read_bytes()
else:
    st.stop()


@st.cache_data(show_spinner="Cleaning data and building cohorts...")
def process(file_bytes: bytes):
    df_raw = cc.load_csv(file_bytes)
    missing = cc.validate_columns(df_raw)
    if missing:
        return {"missing": missing}
    tx, summary = cc.clean_and_assign_cohorts(df_raw)
    matrices = cc.build_matrices(tx, summary["last_full_month"])
    return {"missing": [], "transactions": tx, "summary": summary, "matrices": matrices}


result = process(file_bytes)

if result["missing"]:
    st.error(
        "This CSV is missing required column(s): "
        + ", ".join(f"`{c}`" for c in result["missing"])
    )
    st.stop()

tx = result["transactions"]
summary = result["summary"]
matrices = result["matrices"]
CCY = cc.CURRENCY_SYMBOL

st.header("Data cleaning summary")
col1, col2 = st.columns(2)
with col1:
    st.metric("Starting rows", f"{summary['raw_count']:,}")
    st.metric("Removed — unparseable InvoiceDate", f"{summary['unparseable_date_count']:,}")
    st.metric(
        "Removed — missing CustomerID", f"{summary['missing_customer_count']:,}",
        help=f"{CCY}{summary['missing_customer_revenue']:,.2f} revenue excluded",
    )
    st.metric("Removed — duplicate rows", f"{summary['duplicate_count']:,}")
with col2:
    st.metric(
        "Removed — cancelled sales (netted)",
        f"{summary['cancellation_rows_removed'] + summary['cancellation_netting_rows_removed']:,}",
        help=(
            f"{summary['cancellation_rows_removed']:,} return/cancellation rows + "
            f"{summary['cancellation_netting_rows_removed']:,} purchase rows they reversed "
            f"({CCY}{summary['cancellation_netting_revenue_removed']:,.2f} revenue netted out). "
            + (
                f"{summary['cancellation_unmatched_quantity']:,.0f} cancelled units had no matching "
                "purchase to net against." if summary["cancellation_unmatched_quantity"] else ""
            )
            if summary["cancellation_netting_applied"]
            else "No `StockCode` column — cancellations were dropped without netting against the original sale."
        ),
    )
    st.metric("Removed — unit price ≤ 0", f"{summary['non_positive_price_count']:,}")
    st.metric("Remaining rows", f"{summary['final_row_count']:,}")
    st.metric("Remaining unique customers", f"{summary['final_customer_count']:,}")

if summary["last_month_is_partial"]:
    st.caption(
        f"The data's final calendar month doesn't reach that month's last day, so it's "
        f"excluded from cohort matrices/charts below (analysis runs through "
        f"**{summary['last_full_month']}**). A cohort whose first purchase falls in that "
        "excluded month shows up as an empty/greyed-out row."
    )

st.header("CAC payback")
gross_margin_pct = st.number_input(
    "Gross margin assumption (%)", min_value=0.0, max_value=100.0, value=50.0, step=5.0,
    help="Applied to revenue per customer to estimate gross profit per customer.",
)
cumulative_profit = cc.apply_gross_margin(matrices["cumulative_revenue_per_customer"], gross_margin_pct)

cac = st.number_input("Assumed CAC", min_value=0.0, value=500.0, step=10.0,
                       help=f"Customer acquisition cost, in {CCY}.")

payback = cc.payback_periods(cumulative_profit, cac)
payback_rows = []
for cohort, period in payback.items():
    if cumulative_profit.loc[cohort].isna().all():
        label = "Excluded (partial month)"
    elif period is not None:
        label = f"Period {period}"
    else:
        label = "Not recovered within observed window"
    payback_rows.append({"CohortMonth": cohort, "PaybackPeriod": label})
payback_df = pd.DataFrame(payback_rows)
st.dataframe(payback_df, width="stretch", hide_index=True)

st.header("Cohort matrices")
heatmap_specs = [
    ("Customers", "customer_matrix", "customer_matrix.png", "Customers by cohort & period",
     lambda v: f"{int(v)}", "Distinct customers"),
    ("Revenue", "revenue_matrix", "revenue_matrix.png", "Revenue by cohort & period",
     lambda v: f"{CCY}{v:,.0f}", f"Revenue ({CCY})"),
    ("Retention %", "customer_retention", "retention_matrix.png", "Retention % by cohort & period",
     lambda v: f"{v:.0f}%", "Retention %"),
    ("Cumulative revenue / customer", "cumulative_revenue_per_customer", "cumulative_revenue_matrix.png",
     "Cumulative revenue per customer", lambda v: f"{CCY}{v:,.0f}", f"Cumulative {CCY} / customer"),
]

heatmap_pngs = {}
tabs = st.tabs([spec[0] for spec in heatmap_specs])
for tab, (label, key, filename, title, fmt, cbar_label) in zip(tabs, heatmap_specs):
    with tab:
        fig = cc.render_heatmap(matrices[key], title, fmt, cbar_label)
        st.pyplot(fig)
        png_bytes = cc.fig_to_png_bytes(fig)
        heatmap_pngs[filename] = png_bytes
        st.download_button(f"Download {filename}", png_bytes, file_name=filename, mime="image/png")
        st.dataframe(matrices[key], width="stretch")

st.header("Charts")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    fig_retention = cc.render_retention_chart(matrices["customer_retention"])
    st.pyplot(fig_retention)
    retention_chart_png = cc.fig_to_png_bytes(fig_retention)
    st.download_button(
        "Download retention_curves.png", retention_chart_png,
        file_name="retention_curves.png", mime="image/png",
    )
with chart_col2:
    fig_cac = cc.render_cac_chart(cumulative_profit, cac)
    st.pyplot(fig_cac)
    cac_chart_png = cc.fig_to_png_bytes(fig_cac)
    st.download_button(
        "Download cumulative_gross_profit_cac.png", cac_chart_png,
        file_name="cumulative_gross_profit_cac.png", mime="image/png",
    )

st.header("Cleaned table")
cleaned_csv_bytes = tx.to_csv(index=False).encode("utf-8")
st.caption(
    f"{len(tx):,} rows, original columns plus `Revenue`, `CohortMonth` and `PeriodIndex`."
)
st.download_button(
    "Download clean_retail.csv", cleaned_csv_bytes,
    file_name="clean_retail.csv", mime="text/csv",
)

st.header("Download everything")
zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("clean_retail.csv", cleaned_csv_bytes)
    for filename, png_bytes in heatmap_pngs.items():
        zf.writestr(filename, png_bytes)
    zf.writestr("retention_curves.png", retention_chart_png)
    zf.writestr("cumulative_gross_profit_cac.png", cac_chart_png)
zip_buffer.seek(0)
st.download_button(
    "Download all as ZIP", zip_buffer.getvalue(),
    file_name="cohort_analysis.zip", mime="application/zip",
)

st.caption(
    "Sample data: Online Retail dataset, Daqing Chen, UCI Machine Learning "
    "Repository (London South Bank University)."
)
