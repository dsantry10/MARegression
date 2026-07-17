"""
M&A Days-to-Complete - Random Forest feature analysis
=====================================================
New, richer dataset (New_Training_Sheet_2.xlsx, 1,170 deals). Goal: identify
which deal characteristics correlate with / predict "Days To Complete" using a
Random Forest, and give a sense of DIRECTION (what lengthens vs shortens a deal).

Target:  Days To Complete  (raw days)

Feature handling
----------------
- Numeric (median-imputed + missingness flag): Announced Premium (coerced from
  text; blanks -> missing), Operating Margin, Log Revenue, Log Equity Value.
  Raw "Target Sales/Revenue/Turnover" is dropped: Log Revenue == log(Sales)
  (corr = 1.00), so it is perfectly redundant.
- Binary flags (Yes/No -> 1/0): Additional Stake Purchase, Competing Bid,
  Cross Border, Going Private, PE Buyout, Tender Offer.
- One-hot: Payment Type, Nature of Bid, Target Country/Region,
  Target Industry Group.
- Acquirer Country/Region: top 8 + "Other", one-hot.
- Deal Attributes: most tokens duplicate the binary flags above; we add only
  the non-redundant tokens occurring >= 10 times (Squeeze out, Majority
  purchase, Reverse Merger).

Outputs
-------
- Console: OOB R^2, 5-fold CV R^2 / MAE / RMSE, importance + direction table
- ma_days_rf_importances.xlsx : importances, direction, performance
- ma_days_rf_diagnostics.pdf  : importance bar, predicted-vs-actual, residuals
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INPUT_FILE = "New_Training_Sheet_2.xlsx"
SHEET = "Sheet1"
PLOTS_PDF = "ma_days_rf_diagnostics.pdf"
OUT_XLSX = "ma_days_rf_importances.xlsx"
TARGET = "Days To Complete"
RANDOM_STATE = 42
N_ESTIMATORS = 500

print("=" * 78)
print("M&A DAYS-TO-COMPLETE  -  RANDOM FOREST FEATURE ANALYSIS")
print("=" * 78)

df = pd.read_excel(INPUT_FILE, sheet_name=SHEET)
print(f"\nRows: {len(df)}   Columns: {df.shape[1]}")

y = df[TARGET].astype(float).values
print(f"Target '{TARGET}': min={y.min():.0f}  median={np.median(y):.0f}  "
      f"mean={y.mean():.0f}  max={y.max():.0f}")

# ---------------------------------------------------------------------------
# Build features
# ---------------------------------------------------------------------------
X = pd.DataFrame(index=df.index)
feature_kind = {}  # name -> 'numeric' | 'binary'  (for directional readout)

# --- Numeric (with median imputation + missingness flag) ---
numeric_src = {
    "Announced Premium": pd.to_numeric(df["Announced Premium"], errors="coerce"),
    "Operating Margin": df["Target Trailg 12 Mth Operating Margin"],
    "Log Revenue": df["Log Revenue"],
    "Log Equity Value": df["Log Equity Value"],
}
for name, col in numeric_src.items():
    col = pd.to_numeric(col, errors="coerce")
    n_missing = col.isna().sum()
    if n_missing:
        X[f"{name}_missing"] = col.isna().astype(int).values
        feature_kind[f"{name}_missing"] = "binary"
    X[name] = col.fillna(col.median()).values
    feature_kind[name] = "numeric"

# --- Binary Yes/No flags ---
binary_cols = [
    "Additional Stake Purchase", "Competing Bid", "Cross Border",
    "Going Private", "PE Buyout", "Tender Offer",
]
for c in binary_cols:
    X[c] = (df[c].astype(str).str.strip().str.lower() == "yes").astype(int).values
    feature_kind[c] = "binary"

# --- One-hot categoricals ---
def add_onehot(series, prefix):
    d = pd.get_dummies(series.astype(str).str.strip(), prefix=prefix).astype(int)
    for col in d.columns:
        X[col] = d[col].values
        feature_kind[col] = "binary"

add_onehot(df["Payment Type"], "Pay")
add_onehot(df["Nature of Bid"], "Bid")
add_onehot(df["Target Country/Region"], "TgtCtry")
add_onehot(df["Target Industry Group"], "Ind")

# Acquirer country: top 8 + Other
acq = df["Acquirer Country/Region"].astype(str).str.strip()
top_acq = acq.value_counts().head(8).index
acq_bucketed = acq.where(acq.isin(top_acq), other="Other")
add_onehot(acq_bucketed, "AcqCtry")

# --- Non-redundant Deal Attribute tokens (>=10 occurrences) ---
extra_tokens = ["Squeeze out", "Majority purchase", "Reverse Merger"]
attr = df["Deal Attributes"].astype(str)
for tok in extra_tokens:
    col = f"Attr_{tok}"
    X[col] = attr.str.contains(tok, case=False, regex=False).astype(int).values
    feature_kind[col] = "binary"

X = X.astype(float)
print(f"Engineered feature matrix: {X.shape[0]} rows x {X.shape[1]} features")

# ---------------------------------------------------------------------------
# Fit Random Forest
# ---------------------------------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=N_ESTIMATORS, oob_score=True,
    random_state=RANDOM_STATE, n_jobs=-1,
)
rf.fit(X, y)

print("\n" + "=" * 78)
print("PERFORMANCE  (target = days)")
print("=" * 78)
print(f"In-sample R^2 (optimistic): {rf.score(X, y):.4f}")
print(f"Out-of-bag (OOB) R^2:       {rf.oob_score_:.4f}")

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_pred = cross_val_predict(rf, X, y, cv=cv, n_jobs=-1)
cv_r2 = r2_score(y, cv_pred)
cv_mae = mean_absolute_error(y, cv_pred)
cv_rmse = np.sqrt(mean_squared_error(y, cv_pred))
print(f"\n5-fold CV R^2:   {cv_r2:.4f}")
print(f"5-fold CV MAE:   {cv_mae:.1f} days")
print(f"5-fold CV RMSE:  {cv_rmse:.1f} days")
print(f"(For reference, median completion time is {np.median(y):.0f} days.)")

# ---------------------------------------------------------------------------
# Importance + direction
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("FEATURE IMPORTANCE  +  DIRECTION")
print("=" * 78)

perm = permutation_importance(
    rf, X, y, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1
)


def direction(col):
    """Signed association with Days: numeric -> Spearman rho;
    binary -> mean(days | flag=1) - mean(days | flag=0)."""
    kind = feature_kind.get(col, "binary")
    if kind == "numeric":
        rho, _ = spearmanr(X[col].values, y)
        sign = "+ (longer)" if rho > 0 else "- (shorter)"
        return rho, f"Spearman rho={rho:+.3f} {sign}"
    mask = X[col].values == 1
    if mask.sum() == 0 or (~mask).sum() == 0:
        return 0.0, "n/a (constant)"
    diff = y[mask].mean() - y[~mask].mean()
    sign = "+ (longer)" if diff > 0 else "- (shorter)"
    return diff, f"{diff:+.1f} days vs base {sign} (n={int(mask.sum())})"


rows = []
for col in X.columns:
    dval, dtxt = direction(col)
    rows.append({
        "feature": col,
        "permutation_importance": perm.importances_mean[X.columns.get_loc(col)],
        "perm_std": perm.importances_std[X.columns.get_loc(col)],
        "impurity_importance": rf.feature_importances_[X.columns.get_loc(col)],
        "direction_value": dval,
        "direction": dtxt,
    })

imp = pd.DataFrame(rows).sort_values(
    "permutation_importance", ascending=False
).reset_index(drop=True)

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
print("\nTop 20 features by permutation importance "
      "(higher = more predictive of Days To Complete):\n")
print(imp.head(20)[
    ["feature", "permutation_importance", "perm_std", "impurity_importance",
     "direction"]
].to_string(index=False))

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
with PdfPages(PLOTS_PDF) as pdf:
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["permutation_importance"],
            xerr=top["perm_std"], color="seagreen", edgecolor="k")
    ax.set_xlabel("Permutation importance (mean decrease in R^2)")
    ax.set_title("Random Forest — Top 15 Predictors of Days To Complete")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, cv_pred, alpha=0.4, edgecolor="k", linewidth=0.2)
    lims = [min(y.min(), cv_pred.min()), max(y.max(), cv_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="perfect prediction")
    ax.set_xlabel("Actual days")
    ax.set_ylabel("Cross-validated predicted days")
    ax.set_title(f"Predicted vs Actual (5-fold CV)  —  CV R^2 = {cv_r2:.3f}")
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    resid = y - cv_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cv_pred, resid, alpha=0.4, edgecolor="k", linewidth=0.2)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted days (CV)")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Random Forest Residuals vs Predicted (days)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
print(f"\nSaved diagnostic plots to: {PLOTS_PDF}")

# ---------------------------------------------------------------------------
# Save to xlsx
# ---------------------------------------------------------------------------
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    imp.to_excel(writer, sheet_name="Importance_and_Direction", index=False)
    perf = pd.DataFrame({
        "Metric": ["In-sample R^2", "OOB R^2", "5-fold CV R^2",
                   "5-fold CV MAE (days)", "5-fold CV RMSE (days)",
                   "N observations", "N features", "n_estimators"],
        "Value": [round(rf.score(X, y), 4), round(rf.oob_score_, 4),
                  round(cv_r2, 4), round(cv_mae, 1), round(cv_rmse, 1),
                  int(len(df)), int(X.shape[1]), N_ESTIMATORS],
    })
    perf.to_excel(writer, sheet_name="Performance", index=False)
print(f"Saved importances + direction to: {OUT_XLSX}")

# ---------------------------------------------------------------------------
# Plain-English summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PLAIN-ENGLISH SUMMARY")
print("=" * 78)
print(
    f"\nRandom Forest ({N_ESTIMATORS} trees) on {len(df)} deals and "
    f"{X.shape[1]} engineered features.\nHonest out-of-sample fit: OOB R^2 = "
    f"{rf.oob_score_:.3f}, 5-fold CV R^2 = {cv_r2:.3f}; typical error "
    f"~{cv_mae:.0f} days.\n"
)
print("Strongest correlates of Days To Complete (with direction):")
for _, r in imp.head(8).iterrows():
    print(f"  - {r['feature']:32s} importance={r['permutation_importance']:.4f}"
          f"   {r['direction']}")
print(
    "\nImportance is unsigned (it measures predictive contribution); the "
    "'direction'\ncolumn shows whether higher values / the flag being set are "
    "associated with\nLONGER (+) or SHORTER (-) completion. Read low-importance "
    "directions with caution.\n"
)
print("Done.")
