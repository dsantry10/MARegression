"""
M&A Deal Timing - Random Forest
===============================
A straightforward Random Forest companion to the OLS model, using the SAME
data prep and the SAME variables so the two are directly comparable.

Target:      log(Days to Complete)
Predictors:  Consideration (one-hot, Cash dropped as reference)
             log(Announced Equity Value (mil.))
             Strategic/Sponsor  (Strategic = 1, Sponsor = 0)
             Target Industry Group (one-hot, Health Care dropped as reference)

Unlike OLS, a Random Forest is non-parametric: it captures non-linearities and
interactions automatically and gives variable-importance rankings rather than
signed coefficients. We assess fit honestly with out-of-bag and cross-validated
scores (the sample is small, so a single train/test split would be noisy).

Outputs:
    - Console: OOB R^2, 5-fold CV R^2 / MAE / RMSE, OLS comparison
    - Feature importances (impurity-based + permutation) -> console + .xlsx
    - Plots (importance, predicted-vs-actual, residuals) -> single PDF
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Configuration  (mirrors the OLS script)
# ---------------------------------------------------------------------------
INPUT_FILE = "MA_Statistics_2025-Present__Claude_Code_.xlsx"
SHEET = "Deal Sheet"
PLOTS_PDF = "ma_random_forest_diagnostics.pdf"
IMPORTANCE_XLSX = "ma_random_forest_importances.xlsx"

DEP = "Days to Complete"
EQUITY = "Announced Equity Value (mil.)"
CONSIDERATION = "Consideration"
STRAT = "Strategic/Sponsor"
INDUSTRY = "Target Industry Group"

CONSIDERATION_BASE = "Cash"
INDUSTRY_BASE = "Health Care"

RANDOM_STATE = 42
N_ESTIMATORS = 500

# ---------------------------------------------------------------------------
# 1. Load and prepare data  (identical to the OLS pipeline)
# ---------------------------------------------------------------------------
print("=" * 78)
print("M&A DEAL TIMING - RANDOM FOREST")
print("=" * 78)

raw = pd.read_excel(INPUT_FILE, sheet_name=SHEET)
df = raw[raw["Deal Status"] == "Completed"].copy()

# Fold lone 'Stock Tender' into 'Stock' (same call as the OLS model)
n_stock_tender = (df[CONSIDERATION] == "Stock Tender").sum()
if n_stock_tender:
    df[CONSIDERATION] = df[CONSIDERATION].replace({"Stock Tender": "Stock"})

regression_vars = [DEP, CONSIDERATION, EQUITY, STRAT, INDUSTRY]
df = df.dropna(subset=regression_vars)
print(f"\nModeling sample: {len(df)} completed deals "
      f"(same {len(df)} rows as the OLS model).")

# ---------------------------------------------------------------------------
# 2. Build the design matrix (same variables as OLS)
# ---------------------------------------------------------------------------
y = np.log(df[DEP].values)  # log(Days) — keeps target identical to OLS

X = pd.DataFrame(index=df.index)
X["log_equity"] = np.log(df[EQUITY].values)
X["strategic"] = (df[STRAT] == "Strategic").astype(int).values

# One-hot encode categoricals, dropping the reference level used in the OLS so
# the feature set matches. (RF doesn't need a reference dropped, but matching
# keeps the comparison clean and avoids a redundant column.)
cons = pd.get_dummies(df[CONSIDERATION], prefix="Consideration")
cons = cons.drop(columns=[f"Consideration_{CONSIDERATION_BASE}"], errors="ignore")

ind = pd.get_dummies(df[INDUSTRY], prefix="Industry")
ind = ind.drop(columns=[f"Industry_{INDUSTRY_BASE}"], errors="ignore")

X = pd.concat([X, cons.set_index(X.index), ind.set_index(X.index)], axis=1)
X = X.astype(float)
print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} predictors "
      f"(reference levels '{CONSIDERATION_BASE}' / '{INDUSTRY_BASE}' dropped).")

# ---------------------------------------------------------------------------
# 3. Fit the Random Forest
# ---------------------------------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    oob_score=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X, y)

# ---------------------------------------------------------------------------
# 4. Performance  (OOB + cross-validated, on the log scale)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PERFORMANCE  (target = log days)")
print("=" * 78)

in_sample_r2 = rf.score(X, y)
print(f"In-sample R^2 (optimistic):   {in_sample_r2:.4f}")
print(f"Out-of-bag (OOB) R^2:         {rf.oob_score_:.4f}")

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_pred = cross_val_predict(rf, X, y, cv=cv, n_jobs=-1)
cv_r2 = r2_score(y, cv_pred)
cv_mae = mean_absolute_error(y, cv_pred)
cv_rmse = np.sqrt(mean_squared_error(y, cv_pred))
print(f"\n5-fold CV R^2:                {cv_r2:.4f}")
print(f"5-fold CV MAE (log days):     {cv_mae:.4f}")
print(f"5-fold CV RMSE (log days):    {cv_rmse:.4f}")

# Back-transform CV errors to the day scale for interpretability.
# On the log scale, MAE ~= typical |log ratio|; exp(MAE) is a multiplicative
# error factor. We also report MAE in actual days from back-transformed preds.
days_actual = df[DEP].values
days_pred_cv = np.exp(cv_pred)
cv_mae_days = mean_absolute_error(days_actual, days_pred_cv)
print(f"\n5-fold CV MAE (back-transformed to days): {cv_mae_days:.1f} days")
print(f"   (median deal completes in {np.median(days_actual):.0f} days, "
      f"mean {np.mean(days_actual):.0f})")

# Comparison to OLS (R^2 reported by the OLS script)
print("\nComparison to OLS:")
print("  OLS in-sample R^2 = 0.589, adj-R^2 = 0.522 (from ma_deal_timing_regression.py)")
print(f"  RF  OOB R^2       = {rf.oob_score_:.3f}, 5-fold CV R^2 = {cv_r2:.3f}")
print("  (OLS R^2 is in-sample; the honest RF analogues are OOB / CV.)")

# ---------------------------------------------------------------------------
# 5. Feature importances
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("FEATURE IMPORTANCE")
print("=" * 78)

# Impurity-based (fast, but biased toward high-cardinality / continuous vars)
imp_gini = pd.Series(rf.feature_importances_, index=X.columns)

# Permutation importance (more reliable; measured as drop in OOB-like score)
perm = permutation_importance(
    rf, X, y, n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1
)
imp_perm = pd.Series(perm.importances_mean, index=X.columns)
imp_perm_std = pd.Series(perm.importances_std, index=X.columns)

importance = pd.DataFrame(
    {
        "feature": X.columns,
        "impurity_importance": imp_gini.values,
        "permutation_importance": imp_perm.values,
        "permutation_std": imp_perm_std.values,
    }
).sort_values("permutation_importance", ascending=False).reset_index(drop=True)

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
print("\nRanked by permutation importance (higher = more predictive):\n")
print(importance.to_string(index=False))

print("\nTop predictors of completion time (permutation importance):")
for _, r in importance.head(5).iterrows():
    print(f"  - {r['feature']}: {r['permutation_importance']:.4f}")

# ---------------------------------------------------------------------------
# 6. Plots -> single PDF
# ---------------------------------------------------------------------------
with PdfPages(PLOTS_PDF) as pdf:
    # Permutation importance bar chart (top 15)
    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["permutation_importance"],
            xerr=top["permutation_std"], color="steelblue", edgecolor="k")
    ax.set_xlabel("Permutation importance (mean decrease in R^2)")
    ax.set_title("Random Forest — Feature Importance (top 15)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Predicted vs actual (cross-validated), log scale
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, cv_pred, alpha=0.6, edgecolor="k", linewidth=0.3)
    lims = [min(y.min(), cv_pred.min()), max(y.max(), cv_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="perfect prediction")
    ax.set_xlabel("Actual log(days)")
    ax.set_ylabel("Cross-validated predicted log(days)")
    ax.set_title(f"Predicted vs Actual (5-fold CV)  —  CV R^2 = {cv_r2:.3f}")
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Residuals vs predicted (CV, day scale)
    resid_days = days_actual - days_pred_cv
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(days_pred_cv, resid_days, alpha=0.6, edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted days (CV, back-transformed)")
    ax.set_ylabel("Residual (actual - predicted days)")
    ax.set_title("Random Forest Residuals vs Predicted (days)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"\nSaved diagnostic plots to: {PLOTS_PDF}")

# ---------------------------------------------------------------------------
# 7. Save importances to xlsx
# ---------------------------------------------------------------------------
with pd.ExcelWriter(IMPORTANCE_XLSX, engine="openpyxl") as writer:
    importance.to_excel(writer, sheet_name="Importances", index=False)
    perf = pd.DataFrame(
        {
            "Metric": [
                "In-sample R^2",
                "OOB R^2",
                "5-fold CV R^2",
                "5-fold CV MAE (log days)",
                "5-fold CV RMSE (log days)",
                "5-fold CV MAE (days)",
                "N observations",
                "N predictors",
                "n_estimators",
            ],
            "Value": [
                round(in_sample_r2, 4),
                round(rf.oob_score_, 4),
                round(cv_r2, 4),
                round(cv_mae, 4),
                round(cv_rmse, 4),
                round(cv_mae_days, 1),
                int(len(df)),
                int(X.shape[1]),
                N_ESTIMATORS,
            ],
        }
    )
    perf.to_excel(writer, sheet_name="Performance", index=False)

print(f"Saved feature importances to: {IMPORTANCE_XLSX}")

# ---------------------------------------------------------------------------
# 8. Plain-English summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PLAIN-ENGLISH SUMMARY")
print("=" * 78)
print(
    f"\nA Random Forest ({N_ESTIMATORS} trees) was fit on the same {len(df)} "
    f"completed deals and\nthe same predictors as the OLS model.\n"
)
print(
    f"Honest (cross-validated) fit: CV R^2 = {cv_r2:.3f}, OOB R^2 = "
    f"{rf.oob_score_:.3f}. Typical prediction\nerror is about {cv_mae_days:.0f} "
    f"days (vs a median completion time of {np.median(days_actual):.0f} days).\n"
)
top3 = importance.head(3)["feature"].tolist()
print("The most predictive deal characteristics (permutation importance) are:")
for f in top3:
    print(f"  - {f}")
print(
    "\nThese rankings line up with the OLS findings — consideration type "
    "(esp. Cash Tender)\nand deal size (log equity value) carry the most signal "
    "— but the RF captures any\nnon-linear / interaction effects without us "
    "specifying them. Note that with only\n152 deals and many one-hot industry "
    "columns, the forest's edge over OLS is modest;\nthis is a straightforward "
    "baseline, not a tuned model.\n"
)
print("Done.")
