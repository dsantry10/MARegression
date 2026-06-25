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

> **Caveats:** Industry groups with <5 completed deals — Insurance (1),
> Renewable Energy (1), Media (2), Retail & Wholesale - Staples (2),
> Utilities (2), Consumer Staple Products (4) — have wide confidence intervals;
> interpret with caution.
