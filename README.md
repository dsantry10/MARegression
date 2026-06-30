# M&A Deal Timing Regression

OLS model predicting M&A deal completion time from deal characteristics, using
the `Deal Sheet` tab of the M&A statistics workbook.

## What it does

`ma_deal_timing_regression.py`:

1. **Data prep** — filters to `Deal Status == "Completed"`, drops rows with
   nulls in any regression variable, and folds the single `Stock Tender`
   observation into `Stock` (one obs is too few for a standalone level).
2. **Model** — OLS via `statsmodels`:
   - Dependent: `log(Days to Complete)`
   - Predictors: `Consideration` (Cash = reference), `log(Announced Equity
     Value)`, `Strategic/Sponsor` (Strategic = 1), and `Target Industry Group`
     (Health Care = reference).
3. **Output** — full regression summary, an exponentiated-coefficient table
   (multiplicative effects on days with 95% CIs and p-values), and significance
   flags at p<0.05 / p<0.10.
4. **Diagnostics** — R²/adj-R², residuals-vs-fitted and Q-Q plots, Breusch-Pagan
   test, VIFs, and a flag for industry groups with <5 completed deals.
5. **Files** — `ma_regression_diagnostics.pdf` (plots) and
   `ma_regression_coefficients.xlsx` (coefficient table + diagnostics).

## Run

```bash
pip install pandas numpy statsmodels openpyxl matplotlib scipy
python ma_deal_timing_regression.py
```

## Key findings (n = 152 completed deals)

The model explains ~58.9% of variation in log days-to-complete
(adj. R² = 52.2%). Because the outcome is logged, each `exp(beta)` is a
multiplier on completion time.

Significant at **p < 0.05**:

| Driver | exp(beta) | Effect |
|---|---|---|
| Cash Tender (vs Cash) | 0.47 | ~53% **shorter** |
| Financial Services (vs Health Care) | 1.65 | ~65% **longer** |
| Media (vs Health Care) | 1.89 | ~89% **longer** |
| log(Equity Value) | 1.05 | larger deals take slightly longer |

Marginal (p < 0.10): Banking, Consumer Discretionary Products, Insurance (+);
Oil & Gas, Renewable Energy (−). `Strategic/Sponsor` and `Stock` (vs Cash) are
not significant.

Breusch-Pagan shows no significant heteroskedasticity (p ≈ 0.29); VIFs ≈ 1.0
(no multicollinearity among continuous predictors).

## Random Forest companion

`ma_deal_timing_random_forest.py` fits a straightforward Random Forest (500
trees) on the **same 152 deals and the same predictors** as the OLS, with
categoricals one-hot encoded (reference levels dropped to match). It serves as
a non-parametric cross-check that captures any non-linearities/interactions
without us specifying them.

Outputs: `ma_random_forest_diagnostics.pdf` (importance, predicted-vs-actual,
residuals) and `ma_random_forest_importances.xlsx` (importances + performance).

**Performance (honest, out-of-sample):** OOB R² ≈ 0.33, 5-fold CV R² ≈ 0.36,
typical error ≈ 30 days (median deal = 96 days). In-sample R² is 0.91 — that
gap is the usual RF overfitting, which is why OOB/CV are the numbers to read.
The OLS in-sample R² (0.59) isn't directly comparable to the RF's OOB/CV figures.

**Most predictive features (permutation importance):** Cash Tender, then
log(equity value), then Strategic and Financial Services — the same signals the
OLS flagged. With only 152 rows and many one-hot industry columns, the RF does
not beat the simpler OLS here; it's a baseline, not a tuned model.

## Random Forest on the richer dataset (1,170 deals)

`ma_days_to_complete_random_forest.py` runs a Random Forest on
`New_Training_Sheet_2.xlsx` to find which characteristics correlate with
`Days To Complete` (raw days). Feature engineering: median-imputed numerics with
missingness flags (Premium coerced from text; raw Sales dropped as it equals
log Revenue), Yes/No flags mapped to 1/0, one-hot categoricals, Acquirer country
bucketed to top-8 + Other, and the non-redundant Deal-Attribute tokens. Because
RF importance is unsigned, the output adds a **direction** column (Spearman for
numerics, mean-day difference for flags).

Outputs: `ma_days_rf_diagnostics.pdf` and `ma_days_rf_importances.xlsx`
(importance + direction + performance).

**Fit (honest, out-of-sample):** OOB R² ≈ 0.12, 5-fold CV R² ≈ 0.14, MAE ≈ 47
days (median deal = 81 days). Modest — completion time is largely driven by
factors not captured here, and a 1,078-day outlier inflates RMSE.

**Strongest correlates of Days To Complete:**

| Feature | Direction |
|---|---|
| Log Equity Value | larger deals → **longer** (ρ ≈ +0.21) |
| Log Revenue | larger targets → **longer** (ρ ≈ +0.31) |
| Tender Offer | **~62 days shorter** |
| Banking (industry) | **~52 days longer** |
| Operating Margin | higher → slightly longer (ρ ≈ +0.19) |
| Announced Premium | higher → slightly shorter (ρ ≈ −0.17) |

These line up with the OLS story (tender offers close fast; bigger
deals/financials drag on). High-importance rows with tiny n (e.g. the 3
log-equity-missing rows) are artifacts — read low-n directions with caution.

## Profile simulation

`ma_simulate_profile.py` asks the forest what a specific deal implies:
**Health Care target, US strategic buyer, $1B equity (Log Equity = log10(1000) =
3.0), all-cash, 0% premium, PE Buyout = No.** It fixes those attributes and
marginalizes over everything unspecified by applying the overrides to all 1,170
real rows and predicting (partial-dependence style), plus a single synthetic-deal
point estimate with per-tree spread. Outputs `ma_profile_simulation.xlsx` and
`ma_profile_simulation.pdf`.

**Result:** ~**76-day median** (mean ~84), typical **68–106 day** interquartile
range, slightly below the dataset median of 81 days. The synthetic point estimate
is ~65 days. A real-data check — 121 actual Health Care / US-buyer / Cash deals —
shows mean 75 / median 55 days, corroborating the model. Given OOB R² ≈ 0.12,
read this as a centre-of-mass expectation, not a precise forecast.

### Same profile on the prior (smaller, 2025-present) dataset

`ma_simulate_profile_prior.py` runs the same profile through the **first** RF
(152 completed deals; features = log(equity), strategic, consideration dummies,
industry dummies; target = log days). Two caveats are structural: that model has
**no acquirer-country or premium feature**, so "US buyer" and "no premium" can't
be set; and Health Care + Cash are the model's *reference levels*, so the profile
pins every feature — the simulation is a point prediction plus the 500-tree spread.

**Result:** point ≈ **70 days** (tree IQR ~57–83). The 11 real Health Care / Cash
/ strategic deals averaged 95 days (median 91), but the 3 near-$1B ones averaged
77 — so ~70 days lines up with comparable real deals. Announcing 2026-06-29, that
implies a close around **early September 2026**. OOB R² ≈ 0.33, so still a central
expectation rather than a precise date.

`ma_simulate_profile_prior2.py` runs a second profile on the same prior model:
**Consumer Discretionary, US strategic buyer, $2.431B (log_equity = ln(2431) =
7.796), Stock consideration, 18% premium.** Here Stock *is* a feature (Cash is the
reference) and is the main driver; US buyer and the 18% premium are again not
features of this model. The dataset splits Consumer Discretionary into Products
and Services, so both are simulated.

**Result:** ~**105–107 days** for both sub-groups (close ~ mid-October 2026 if
announced 2026-06-29) — materially longer than the all-cash Health Care case
(~70 days), driven by **stock consideration** plus larger size. Real comparables
(7 CD+Stock completed deals) median ~130 days, so the model sits a touch below
the noisy small-sample actuals. Same OOB R² ≈ 0.33 caveat.

`ma_simulate_profile_large2.py` runs the same kind of profile through the
**larger** model: **Consumer Discretionary (Automobiles & Components), US strategic
buyer, $2.431B (Log Equity = log10(2431) = 3.386), Stock, 18% premium.** This
model *does* use premium and acquirer country, so the full profile is
representable (the dataset has no autos-specific group, so autos map to Consumer
Discretionary Products). Marginalized partial-dependence over all 1,170 rows.

**Result:** Products (autos) ≈ **94-day median** (IQR 85–109, close ~ early Oct
2026); Services alternate ≈ 109 days. Consistent with the prior model's ~105–107
days for a comparable stock deal. Real comparables are tiny/noisy (CD Products +
Stock + US buyer n=1 at 213 days), so lean on the model's central range. OOB
R² ≈ 0.12.

> **Caveats:** Industry groups with <5 completed deals — Insurance (1),
> Renewable Energy (1), Media (2), Retail & Wholesale - Staples (2),
> Utilities (2), Consumer Staple Products (4) — have wide confidence intervals;
> interpret with caution.
