import zipfile
from io import BytesIO

import pandas as pd
import streamlit as st

import cohort_core as cc

st.set_page_config(page_title="Cohort & CAC Payback Analysis", layout="wide")

st.title("Cohort & CAC Payback Analysis")
st.write(
    "Upload a transaction-level CSV. Required columns: "
    f"`{'`, `'.join(cc.REQUIRED_COLUMNS)}`. Any other columns "
    "(e.g. `InvoiceNo`, `Description`, `Country`) are kept but not required."
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.stop()

file_bytes = uploaded_file.getvalue()


@st.cache_data(show_spinner="Cleaning data and building cohorts...")
def process(file_bytes: bytes):
    df_raw = cc.load_csv(file_bytes)
    missing = cc.validate_columns(df_raw)
    if missing:
        return {"missing": missing}
    tx, summary = cc.clean_and_assign_cohorts(df_raw)
    matrices = cc.build_matrices(tx)
    cac_suggestion = cc.suggested_cac(tx)
    return {
        "missing": [],
        "transactions": tx,
        "summary": summary,
        "matrices": matrices,
        "cac_suggestion": cac_suggestion,
    }


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

st.header("Data cleaning summary")
col1, col2 = st.columns(2)
with col1:
    st.metric("Starting rows", f"{summary['raw_count']:,}")
    st.metric("Removed — unparseable InvoiceDate", f"{summary['unparseable_date_count']:,}")
    st.metric(
        "Removed — missing CustomerID", f"{summary['missing_customer_count']:,}",
        help=f"${summary['missing_customer_revenue']:,.2f} revenue excluded",
    )
    st.metric("Removed — duplicate rows", f"{summary['duplicate_count']:,}")
with col2:
    st.metric("Removed — negative quantity", f"{summary['negative_qty_count']:,}")
    st.metric("Removed — unit price ≤ 0", f"{summary['non_positive_price_count']:,}")
    st.metric("Remaining rows", f"{summary['final_row_count']:,}")
    st.metric("Remaining unique customers", f"{summary['final_customer_count']:,}")

st.header("CAC payback")
default_cac = float(result["cac_suggestion"])
st.caption(f"Suggested CAC based on this data: ${default_cac:,.0f}")
cac = st.number_input("Assumed CAC ($)", min_value=0.0, value=default_cac, step=10.0)

payback = cc.payback_periods(matrices["cumulative_revenue_per_customer"], cac)
payback_df = pd.DataFrame(
    [
        {
            "CohortMonth": k,
            "PaybackPeriod": f"Period {v}" if v is not None else "Not recovered within observed window",
        }
        for k, v in payback.items()
    ]
)
st.dataframe(payback_df, width="stretch", hide_index=True)

st.header("Cohort matrices")
heatmap_specs = [
    ("Customers", "customer_matrix", "customer_matrix.png", "Customers by cohort & period",
     lambda v: f"{int(v)}", "Distinct customers"),
    ("Revenue", "revenue_matrix", "revenue_matrix.png", "Revenue by cohort & period",
     lambda v: f"${v:,.0f}", "Revenue ($)"),
    ("Retention %", "customer_retention", "retention_matrix.png", "Retention % by cohort & period",
     lambda v: f"{v:.0f}%", "Retention %"),
    ("Cumulative revenue / customer", "cumulative_revenue_per_customer", "cumulative_revenue_matrix.png",
     "Cumulative revenue per customer", lambda v: f"${v:,.0f}", "Cumulative $ / customer"),
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
    fig_cac = cc.render_cac_chart(matrices["cumulative_revenue_per_customer"], cac)
    st.pyplot(fig_cac)
    cac_chart_png = cc.fig_to_png_bytes(fig_cac)
    st.download_button(
        "Download cumulative_revenue_cac.png", cac_chart_png,
        file_name="cumulative_revenue_cac.png", mime="image/png",
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
    zf.writestr("cumulative_revenue_cac.png", cac_chart_png)
zip_buffer.seek(0)
st.download_button(
    "Download all as ZIP", zip_buffer.getvalue(),
    file_name="cohort_analysis.zip", mime="application/zip",
)

st.caption(
    "Sample data: Online Retail dataset, Daqing Chen, UCI Machine Learning "
    "Repository (London South Bank University)."
)
