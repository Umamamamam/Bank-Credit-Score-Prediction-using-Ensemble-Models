# Bank GoodCredit — Credit Risk Scoring (PR-0015)

Predicts whether a credit card customer will become a **bad customer** (falls 30+ days past due), so the bank can catch risky customers early and reduce default losses.

## Business Context
Bank GoodCredit wants a credit score for current customers. Target: `Bad_label` (0 = good credit history, 1 = bad credit history). Existing benchmark model scores **Gini = 37.9** — this project tries to match or beat it.

## Files
- `bank_credit_risk.py` — full pipeline
- `Cust_Account.csv`, `Cust_Enquiry.csv`, `Cust_Demographics.csv` — source data (place in same folder)

## Requirements
```
pip install pandas numpy scikit-learn xgboost --break-system-packages
```

## The 3 Source Tables

| Table | What it holds | Rows per customer |
|---|---|---|
| `Cust_Account` | Credit account history — balances, limits, payment dates | Multiple (avg ~7.8) |
| `Cust_Enquiry` | Loan/credit applications the customer made | Multiple (avg ~17) |
| `Cust_Demographics` | Demographic features (`feature_1`–`feature_79`) + target `Bad_label` | One (base table) |

All 3 link on `customer_no`.

## Pipeline Overview

1. **Load** the 3 CSVs.
2. **Fix mixed-type columns** — some numeric columns (`rateofinterest`, `high_credit_amt`, `feature_10/18/61`, etc.) load as text due to stray/blank values. Force to numeric with `pd.to_numeric(errors="coerce")` before any aggregation, or `groupby().agg()` throws a `TypeError`.
3. **Parse date columns** (`opened_dt`, `last_paymt_dt`, `enquiry_dt`, etc.).
4. **Aggregate `Cust_Account` per customer** — since one customer has many accounts, roll up into one row: totals, means, and time-gap features (e.g. `mean_diff_lastpaymt_opened_dt`).
5. **Aggregate `Cust_Enquiry` per customer** — same idea: total/mean enquiry amounts, enquiry counts, and recency counts (last 90/365 days).
6. **Merge everything onto `Cust_Demographics`** (the one-row-per-customer base table) using `customer_no`.
7. **Handle missing values:**
   - `feature_1`–`feature_79` → median
   - Count/total columns (`num_accounts`, `total_enq_amt`, etc.) → 0 (missing = customer has none, not unknown)
   - Ratio/mean columns → median (missing = nothing to average, not zero)
   - Rows with missing `Bad_label` → dropped (can't train without a label)
8. **Encode categoricals** (`max_freq_enquiry`) with `LabelEncoder`.
9. **Build feature matrix** — drop IDs/dates/target, keep numeric columns only.
10. **Train/test split** — 80/20, stratified on `Bad_label`.
11. **Train two models:**
    - **Logistic Regression** — interpretable baseline (scaled features, `class_weight="balanced"`)
    - **XGBoost** — main model, matches the benchmark's "Ensemble" approach
12. **Evaluate with Gini score** (`2 × AUC − 1`) — the banking-standard metric, compared directly against the 37.9 benchmark.
13. **Decile rank-ordering table** — splits customers into 10 risk groups by predicted probability; decile 10 (riskiest) should show a much higher bad-customer rate than decile 1 (safest). This is how banks sanity-check that a risk model actually separates good from bad customers.
14. **Feature importance (gain)** — shows which features drove the XGBoost predictions, for the feature-matrix deliverable.

## Deliverables (per the project brief)
- Data exploration insights — what was found, what decisions were made (missing values, aggregation choices)
- Feature matrix — features used, each with a gain score
- Model evaluation — Gini score + decile rank ordering, compared to the 37.9 benchmark

## Notes / Things to Double-Check
- Column names in the actual CSVs may not perfectly match what's assumed here (based on the project PDF) — verify with `df.columns` if something doesn't line up.
- `feature_1`–`feature_79` are anonymized on purpose — treat them as unlabeled numeric inputs, no need to guess their real-world meaning.
- Gini, not accuracy/F1, is the metric that matters here — this is different from the ITSM ticket-priority project, which prioritized recall/F1.