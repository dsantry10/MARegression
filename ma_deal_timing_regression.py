"""
M&A Deal Timing Regression
===========================
OLS model predicting deal completion time from the "Deal Sheet" tab of the
M&A statistics workbook.

Dependent variable:   log(Days to Complete)
Independent variables:
    - Consideration            (categorical; Cash = reference)
    - log(Announced Equity Value (mil.))
    - Strategic/Sponsor        (binary; Strategic = 1, Sponsor = 0)
    - Target Industry Group    (categorical; Health Care = reference)

Outputs:
    - Full regression summary (console)
    - Exponentiated-coefficient table with 95% CIs (console + .xlsx)
    - Diagnostics: R^2, residual plots, Q-Q plot, Breusch-Pagan, VIF
    - All plots -> single PDF
    - Plain-English summary of key findings
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_FILE = "MA_Statistics_2025-Present__Claude_Code_.xlsx"
SHEET = "Deal Sheet"
PLOTS_PDF = "ma_regression_diagnostics.pdf"
COEF_XLSX = "ma_regression_coefficients.xlsx"

DEP = "Days to Complete"
EQUITY = "Announced Equity Value (mil.)"
CONSIDERATION = "Consideration"
STRAT = "Strategic/Sponsor"
INDUSTRY = "Target Industry Group"

CONSIDERATION_BASE = "Cash"
INDUSTRY_BASE = "Health Care"
SMALL_SAMPLE_THRESHOLD = 5

# ---------------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------------
print("=" * 78)
print("M&A DEAL TIMING REGRESSION")
print("=" * 78)

raw = pd.read_excel(INPUT_FILE, sheet_name=SHEET)
print(f"\nRaw rows read from '{SHEET}': {len(raw)}")

# Filter to completed deals
df = raw[raw["Deal Status"] == "Completed"].copy()
print(f"Rows with Deal Status == 'Completed': {len(df)}")

# Combine the lone 'Stock Tender' observation into 'Stock'.
# Stock Tender has only one completed observation, so an isolated dummy would
# be statistically meaningless (perfectly fit, infinite-variance estimate).
# Folding it into 'Stock' keeps the row and treats it as stock consideration.
n_stock_tender = (df[CONSIDERATION] == "Stock Tender").sum()
if n_stock_tender:
    df[CONSIDERATION] = df[CONSIDERATION].replace({"Stock Tender": "Stock"})
    print(
        f"Combined {n_stock_tender} 'Stock Tender' deal(s) into 'Stock' "
        f"(too few observations for a standalone level)."
    )

regression_vars = [DEP, CONSIDERATION, EQUITY, STRAT, INDUSTRY]
before = len(df)
df = df.dropna(subset=regression_vars)
print(f"Dropped {before - len(df)} row(s) with nulls in regression variables.")
print(f"Final modeling sample: {len(df)} deals")

# ---------------------------------------------------------------------------
# 2. Build model variables
# ---------------------------------------------------------------------------
df["log_days"] = np.log(df[DEP])
df["log_equity"] = np.log(df[EQUITY])
# Strategic = 1, Sponsor = 0
df["strategic"] = (df[STRAT] == "Strategic").astype(int)

# Order categoricals so the reference level is first (statsmodels uses the
# first category as the base when using Treatment coding via C(...)).
consideration_levels = [CONSIDERATION_BASE] + sorted(
    [c for c in df[CONSIDERATION].unique() if c != CONSIDERATION_BASE]
)
industry_levels = [INDUSTRY_BASE] + sorted(
    [g for g in df[INDUSTRY].unique() if g != INDUSTRY_BASE]
)

# ---------------------------------------------------------------------------
# 3. Fit OLS
# ---------------------------------------------------------------------------
formula = (
    "log_days ~ "
    f"C(Q('{CONSIDERATION}'), Treatment(reference='{CONSIDERATION_BASE}')) "
    "+ log_equity "
    "+ strategic "
    f"+ C(Q('{INDUSTRY}'), Treatment(reference='{INDUSTRY_BASE}'))"
)

model = smf.ols(formula=formula, data=df).fit()


def clean_name(name: str) -> str:
    """Turn statsmodels' verbose term names into readable labels."""
    n = name
    n = n.replace(
        f"C(Q('{CONSIDERATION}'), Treatment(reference='{CONSIDERATION_BASE}'))",
        "Consideration",
    )
    n = n.replace(
        f"C(Q('{INDUSTRY}'), Treatment(reference='{INDUSTRY_BASE}'))",
        "Industry",
    )
    n = n.replace("[T.", "[").replace("]", "]")
    return n


print("\n" + "=" * 78)
print("FULL OLS REGRESSION SUMMARY")
print("=" * 78)
print(model.summary())

# ---------------------------------------------------------------------------
# 4. Exponentiated coefficient table (multiplicative effects on days)
# ---------------------------------------------------------------------------
params = model.params
conf = model.conf_int(alpha=0.05)
conf.columns = ["ci_low", "ci_high"]
pvals = model.pvalues

coef_tbl = pd.DataFrame(
    {
        "term": [clean_name(i) for i in params.index],
        "beta": params.values,
        "exp_beta": np.exp(params.values),
        "exp_ci_low": np.exp(conf["ci_low"].values),
        "exp_ci_high": np.exp(conf["ci_high"].values),
        "p_value": pvals.values,
    }
)


def signif_flag(p: float) -> str:
    if p < 0.05:
        return "** p<0.05"
    if p < 0.10:
        return "*  p<0.10"
    return ""


coef_tbl["significance"] = coef_tbl["p_value"].apply(signif_flag)
# pct change interpretation (only meaningful for non-intercept terms)
coef_tbl["pct_change_in_days"] = (coef_tbl["exp_beta"] - 1) * 100

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

print("\n" + "=" * 78)
print("EXPONENTIATED COEFFICIENTS  (multiplicative effect on Days to Complete)")
print("exp(beta) > 1  => longer completion time;  < 1 => shorter")
print("=" * 78)
print(
    coef_tbl[
        [
            "term",
            "beta",
            "exp_beta",
            "exp_ci_low",
            "exp_ci_high",
            "p_value",
            "pct_change_in_days",
            "significance",
        ]
    ].to_string(index=False)
)

print("\nSignificant coefficients (p < 0.05):")
sig05 = coef_tbl[(coef_tbl["p_value"] < 0.05) & (coef_tbl["term"] != "Intercept")]
if len(sig05):
    for _, r in sig05.iterrows():
        print(f"  - {r['term']}: exp(beta)={r['exp_beta']:.3f}, p={r['p_value']:.4f}")
else:
    print("  (none)")

print("\nMarginally significant coefficients (0.05 <= p < 0.10):")
sig10 = coef_tbl[
    (coef_tbl["p_value"] >= 0.05)
    & (coef_tbl["p_value"] < 0.10)
    & (coef_tbl["term"] != "Intercept")
]
if len(sig10):
    for _, r in sig10.iterrows():
        print(f"  - {r['term']}: exp(beta)={r['exp_beta']:.3f}, p={r['p_value']:.4f}")
else:
    print("  (none)")

# ---------------------------------------------------------------------------
# 5. Diagnostics
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("DIAGNOSTICS")
print("=" * 78)
print(f"R-squared:          {model.rsquared:.4f}")
print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
print(f"F-statistic:        {model.fvalue:.4f}  (p = {model.f_pvalue:.4g})")
print(f"N observations:     {int(model.nobs)}")

# Breusch-Pagan test
bp = het_breuschpagan(model.resid, model.model.exog)
bp_labels = ["LM stat", "LM p-value", "F stat", "F p-value"]
print("\nBreusch-Pagan test for heteroskedasticity:")
for lab, val in zip(bp_labels, bp):
    print(f"  {lab:12s}: {val:.4f}")
if bp[1] < 0.05:
    print("  => Reject homoskedasticity (p<0.05): evidence of heteroskedasticity.")
    print("     Consider robust (HC) standard errors when interpreting p-values.")
else:
    print("  => Fail to reject homoskedasticity (p>=0.05): no strong evidence.")

# VIF for non-dummy (continuous) variables.
# Computed on the continuous regressors plus the strategic dummy is excluded
# from "non-dummy"; the spec asks for VIF on non-dummy variables. The only
# truly continuous regressor is log_equity, but we report VIF for the full
# numeric design among continuous + binary predictors to check collinearity.
print("\nVariance Inflation Factors (continuous / non-categorical predictors):")
vif_data = df[["log_equity", "strategic"]].copy()
vif_data = sm.add_constant(vif_data)
for i, col in enumerate(vif_data.columns):
    if col == "const":
        continue
    vif = variance_inflation_factor(vif_data.values, i)
    tag = " (binary dummy)" if col == "strategic" else ""
    print(f"  {col:12s}: VIF = {vif:.4f}{tag}")
print("  (Rule of thumb: VIF > 5 suggests problematic multicollinearity.)")

# Small-sample industry groups
print("\nIndustry groups with fewer than "
      f"{SMALL_SAMPLE_THRESHOLD} completed deals (estimates may be unreliable):")
ind_counts = df[INDUSTRY].value_counts()
small = ind_counts[ind_counts < SMALL_SAMPLE_THRESHOLD]
if len(small):
    for grp, n in small.items():
        print(f"  - {grp}: {n} deal(s)")
else:
    print("  (none)")

# ---------------------------------------------------------------------------
# 6. Plots -> single PDF
# ---------------------------------------------------------------------------
fitted = model.fittedvalues
resid = model.resid
std_resid = model.get_influence().resid_studentized_internal

with PdfPages(PLOTS_PDF) as pdf:
    # Residuals vs fitted
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(fitted, resid, alpha=0.6, edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    # lowess trend
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        sm_line = lowess(resid, fitted, frac=0.6)
        ax.plot(sm_line[:, 0], sm_line[:, 1], color="blue", linewidth=1.2,
                label="LOWESS")
        ax.legend()
    except Exception:
        pass
    ax.set_xlabel("Fitted values (log days)")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted Values")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Q-Q plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.qqplot(std_resid, line="45", ax=ax)
    ax.set_title("Normal Q-Q Plot of Standardized Residuals")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Histogram of residuals (extra context)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(resid, bins=20, edgecolor="k", alpha=0.7)
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Residuals")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"\nSaved diagnostic plots to: {PLOTS_PDF}")

# ---------------------------------------------------------------------------
# 7. Save coefficient table to xlsx
# ---------------------------------------------------------------------------
out_tbl = coef_tbl[
    [
        "term",
        "beta",
        "exp_beta",
        "exp_ci_low",
        "exp_ci_high",
        "p_value",
        "pct_change_in_days",
        "significance",
    ]
].copy()
out_tbl.columns = [
    "Term",
    "Beta (log scale)",
    "exp(Beta)",
    "95% CI low (exp)",
    "95% CI high (exp)",
    "p-value",
    "% change in days",
    "Significance",
]

with pd.ExcelWriter(COEF_XLSX, engine="openpyxl") as writer:
    out_tbl.to_excel(writer, sheet_name="Coefficients", index=False)

    diag = pd.DataFrame(
        {
            "Metric": [
                "R-squared",
                "Adjusted R-squared",
                "F-statistic",
                "F p-value",
                "N observations",
                "Breusch-Pagan LM p-value",
                "Dependent variable",
                "Consideration reference",
                "Industry reference",
            ],
            "Value": [
                round(model.rsquared, 4),
                round(model.rsquared_adj, 4),
                round(model.fvalue, 4),
                f"{model.f_pvalue:.4g}",
                int(model.nobs),
                round(bp[1], 4),
                "log(Days to Complete)",
                CONSIDERATION_BASE,
                INDUSTRY_BASE,
            ],
        }
    )
    diag.to_excel(writer, sheet_name="Diagnostics", index=False)

    if len(small):
        small_df = small.reset_index()
        small_df.columns = ["Industry Group", "Completed Deals"]
        small_df.to_excel(writer, sheet_name="Small Sample Groups", index=False)

print(f"Saved coefficient table to:  {COEF_XLSX}")

# ---------------------------------------------------------------------------
# 8. Plain-English summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PLAIN-ENGLISH SUMMARY OF KEY FINDINGS")
print("=" * 78)

print(
    f"\nThe model explains {model.rsquared*100:.1f}% of the variation in (log) "
    f"days-to-complete\n(adjusted R-squared = {model.rsquared_adj*100:.1f}%), "
    f"based on {int(model.nobs)} completed deals.\n"
)
print(
    "Because the outcome is logged, each exp(beta) is a MULTIPLIER on completion\n"
    "time: 1.20 means ~20% longer, 0.85 means ~15% shorter, holding all else equal.\n"
)

drivers = coef_tbl[
    (coef_tbl["term"] != "Intercept") & (coef_tbl["p_value"] < 0.10)
].copy()
drivers = drivers.sort_values("p_value")

if len(drivers):
    print("Statistically notable drivers of completion time:")
    for _, r in drivers.iterrows():
        direction = "LONGER" if r["exp_beta"] > 1 else "SHORTER"
        pct = abs(r["pct_change_in_days"])
        level = "p<0.05" if r["p_value"] < 0.05 else "p<0.10 (marginal)"
        if r["term"] == "log_equity":
            print(
                f"  - Deal size (log equity value): a 1% increase in equity value is\n"
                f"    associated with ~{r['beta']:.3f}% change in days "
                f"(exp(beta)={r['exp_beta']:.3f}). [{level}]"
            )
        else:
            print(
                f"  - {r['term']}: associated with ~{pct:.1f}% {direction} "
                f"completion time\n    (exp(beta)={r['exp_beta']:.3f}, "
                f"p={r['p_value']:.4f}). [{level}]"
            )
else:
    print("No predictors reached significance at the p<0.10 level.")

print(
    "\nCaveats:\n"
    "  - 'Stock Tender' (1 deal) was merged into 'Stock'.\n"
    "  - Industry groups with <5 completed deals (listed above) have wide\n"
    "    confidence intervals; treat their coefficients with caution.\n"
)
if bp[1] < 0.05:
    print(
        "  - Breusch-Pagan indicates heteroskedasticity; consider HC robust\n"
        "    standard errors before drawing firm inferential conclusions."
    )
print("\nDone.")
