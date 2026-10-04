# Cohort & CAC Payback Analysis

A Streamlit app that turns raw e-commerce transactions into cohort
retention, revenue, and CAC-payback analysis. Upload any transaction CSV
with `CustomerID`, `InvoiceDate`, `Quantity` and `UnitPrice`, enter a CAC
and a gross margin assumption, and get cohort matrices, retention curves,
and a payback chart — all downloadable as PNGs/CSV/ZIP.

Built with Claude Code in 6 hours; I specified the analysis, checked the
outputs and made the methodology calls.

**Try it live:** https://mahinazhar-cohort-tool-app-o7dfwn.streamlit.app
**Run locally:** see [Running locally](#running-locally) below.

![Retention and CAC payback charts](app_screenshot.png)
_Screenshot: the app's retention-curve and CAC-payback charts, run against
the sample Online Retail dataset with a £500 CAC and 50% gross margin._

## How it works

The app cleans the data in a fixed order, tracking what's removed at each
step:

1. Drop rows with an unparseable `InvoiceDate`.
2. Drop rows with a missing `CustomerID` (tracked with the revenue this
   excludes).
3. Drop exact duplicate rows.
4. **Net cancellations against the purchases they reverse** (needs a
   `StockCode` column — matches each return, in chronological order, to
   the customer's most recent *prior* purchase of the same product,
   consuming partial quantities where needed; a purchase dated after the
   cancellation is never used as a match. Without `StockCode`,
   cancellations are just dropped).
5. Drop rows with `UnitPrice <= 0`.

Each surviving customer gets a cohort = the month of their first
purchase. Each transaction gets a period index = months since that
cohort month. If the data's last calendar month is partial (doesn't reach
that month's last day), it's excluded from all matrices/charts so it
doesn't understate every active cohort's most recent period — a cohort
whose first purchase falls in that excluded month shows up as a
greyed-out row instead.

From there the app builds: a customer-count matrix, a revenue matrix, a
retention % matrix, and a cumulative-revenue-per-customer matrix — each
as a heatmap and a table — plus a retention-curve chart and a
cumulative-gross-profit-vs-CAC chart with per-cohort payback periods.

## Running locally

```
pip install -r requirements.txt
streamlit run app.py
```

Click **"Use sample dataset"** to try it against the bundled
`Online_Retail.csv` without uploading anything.

## Data

`Online_Retail.csv` is included as sample data — Online Retail dataset,
Daqing Chen, UCI Machine Learning Repository (London South Bank
University).
