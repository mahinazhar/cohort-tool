# Cohort & CAC Payback Analysis

A Streamlit app for cohort retention and CAC-payback analysis on retail
transaction data.

Upload a transaction-level CSV with at least `CustomerID`, `InvoiceDate`,
`Quantity` and `UnitPrice` columns. The app cleans the data (drops rows
with missing `CustomerID`, exact duplicates, negative quantities, and
non-positive unit prices), assigns each customer a cohort (their first
purchase month), computes each transaction's period index, and builds:

- Customer count, revenue, retention %, and cumulative-revenue-per-customer
  matrices (as heatmaps and tables)
- Retention curves by cohort
- A cumulative-revenue-per-customer chart with a configurable CAC line and
  payback period per cohort

All matrices, charts, and the cleaned table are downloadable individually
or as a single ZIP.

## Running locally

```
pip install -r requirements.txt   # or: streamlit duckdb pandas numpy matplotlib
streamlit run app.py
```

## Secrets

This app doesn't currently call any external API and has no secrets to
configure. If that changes, put credentials in `.streamlit/secrets.toml`
(already gitignored) rather than hardcoding them.

## Data

`Online_Retail.csv` / `Online_Retail.xlsx` are included as sample data you
can upload to try the app.

Data: Online Retail dataset, Daqing Chen, UCI Machine Learning Repository
(London South Bank University).
