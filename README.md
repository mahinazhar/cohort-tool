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

## Methodology calls and caveats

Calls I made:

- **Cancellations are netted, not dropped.** Dropping return rows leaves
  the original sale counted in full, which overstates revenue. Each
  return is matched to an earlier purchase of the same product by the
  same customer.
- **A partial final month is excluded.** Otherwise every cohort's most
  recent period looks like a drop in retention when it is just
  incomplete data.
- **Rows with no customer ID are removed.** They can't be assigned to a
  cohort. The revenue this excludes is shown in the cleaning summary
  so the size of the gap is visible.

Caveats when reading the output:

- **The earliest cohort is not a true cohort.** The data has a start
  date, so the first month contains every existing customer who
  happened to buy then, not just new ones. It will look stronger than
  later cohorts on retention and payback, and should be read
  separately.
- **Retention curves pick up calendar seasonality.** Each cohort's
  latest point falls in the same calendar month. If that month is a
  seasonal peak, every curve turns up at the end, which is timing and
  not improving retention.
- **Payback uses the average customer.** Revenue per customer is a
  mean, so a few very large accounts can pull a whole cohort past
  payback while the typical customer in it is not there yet.
- **CAC is an assumption, not data.** The dataset has no acquisition
  spend, so the payback period is driven by the CAC and gross margin
  entered. Treat it as a sensitivity to those inputs, not as a finding
  about the business.

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
