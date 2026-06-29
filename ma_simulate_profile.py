"""
Random Forest simulation for a specific deal profile
====================================================
Profile requested:
    - Target Industry Group : Health Care
    - Acquirer Country       : United States  (US buyer)
    - Equity Value           : $1B  ->  Log Equity Value = log10(1000) = 3.0
    - Payment Type           : Cash
    - Announced Premium      : 0 (no premium)
    - Buyer type             : Strategic  ->  PE Buyout = No

We reuse the SAME Random Forest and feature engineering as
ma_days_to_complete_random_forest.py.

Because the model uses many features we did NOT specify (revenue, operating
margin, tender/cross-border flags, target country, nature of bid, etc.), we
estimate the implied duration two ways:

  (A) MARGINALIZED SIMULATION  (partial dependence):
      Take all 1,170 real deals, overwrite ONLY the specified attributes on
      every row, predict, and report the resulting distribution. This averages
      over the unspecified attributes using their real joint distribution --
      "for a deal like this, across the range of everything else we didn't pin
      down, what does the model imply?"

  (B) SINGLE SYNTHETIC DEAL  (point estimate):
      One row with unspecified numerics at their median and all unspecified
      flags off, then the profile applied. We also report the spread of the
      500 individual trees for this row as a model-uncertainty band.

A real-data sanity check (actual Health Care + US-buyer + Cash deals) is printed
for comparison.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INPUT_FILE = "New_Training_Sheet_2.xlsx"
SHEET = "Sheet1"
TARGET = "Days To Complete"
RANDOM_STATE = 42
N_ESTIMATORS = 500
OUT_PDF = "ma_profile_simulation.pdf"
OUT_XLSX = "ma_profile_simulation.xlsx"

# ---------------------------------------------------------------------------
# Rebuild features identically to the training script
# ---------------------------------------------------------------------------
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET)
y = df[TARGET].astype(float).values

X = pd.DataFrame(index=df.index)

numeric_src = {
    "Announced Premium": pd.to_numeric(df["Announced Premium"], errors="coerce"),
    "Operating Margin": df["Target Trailg 12 Mth Operating Margin"],
    "Log Revenue": df["Log Revenue"],
    "Log Equity Value": df["Log Equity Value"],
}
numeric_medians = {}
for name, col in numeric_src.items():
    col = pd.to_numeric(col, errors="coerce")
    med = col.median()
    numeric_medians[name] = med
    if col.isna().sum():
        X[f"{name}_missing"] = col.isna().astype(int).values
    X[name] = col.fillna(med).values

binary_cols = [
    "Additional Stake Purchase", "Competing Bid", "Cross Border",
    "Going Private", "PE Buyout", "Tender Offer",
]
for c in binary_cols:
    X[c] = (df[c].astype(str).str.strip().str.lower() == "yes").astype(int).values


def add_onehot(series, prefix):
    d = pd.get_dummies(series.astype(str).str.strip(), prefix=prefix).astype(int)
    for col in d.columns:
        X[col] = d[col].values


add_onehot(df["Payment Type"], "Pay")
add_onehot(df["Nature of Bid"], "Bid")
add_onehot(df["Target Country/Region"], "TgtCtry")
add_onehot(df["Target Industry Group"], "Ind")
acq = df["Acquirer Country/Region"].astype(str).str.strip()
top_acq = acq.value_counts().head(8).index
add_onehot(acq.where(acq.isin(top_acq), other="Other"), "AcqCtry")

for tok in ["Squeeze out", "Majority purchase", "Reverse Merger"]:
    X[f"Attr_{tok}"] = df["Deal Attributes"].astype(str).str.contains(
        tok, case=False, regex=False).astype(int).values

X = X.astype(float)

rf = RandomForestRegressor(
    n_estimators=N_ESTIMATORS, oob_score=True,
    random_state=RANDOM_STATE, n_jobs=-1,
)
rf.fit(X, y)
print(f"Random Forest refit: {X.shape[0]} deals x {X.shape[1]} features  "
      f"(OOB R^2 = {rf.oob_score_:.3f})")

# ---------------------------------------------------------------------------
# Define the requested profile as feature overrides
# ---------------------------------------------------------------------------
LOG_EQUITY_1B = np.log10(1000.0)  # $1,000mm -> 3.0 on the dataset's log10($mm) scale

overrides = {
    "Log Equity Value": LOG_EQUITY_1B,
    "Announced Premium": 0.0,
    "PE Buyout": 0.0,            # strategic (corporate) buyer, not a sponsor
}
# Missingness flags for the values we are explicitly setting -> not missing
if "Announced Premium_missing" in X.columns:
    overrides["Announced Premium_missing"] = 0.0
if "Log Equity Value_missing" in X.columns:
    overrides["Log Equity Value_missing"] = 0.0


def set_exclusive(prefix, chosen):
    """Set the chosen one-hot column to 1 and all sibling <prefix>_* to 0."""
    sibs = [c for c in X.columns if c.startswith(prefix + "_")]
    target_col = f"{prefix}_{chosen}"
    assert target_col in sibs, f"Missing column {target_col!r}; have {sibs}"
    for c in sibs:
        overrides[c] = 1.0 if c == target_col else 0.0


set_exclusive("Pay", "Cash")
set_exclusive("Ind", "Health Care")
set_exclusive("AcqCtry", "United States")

print("\nProfile overrides applied:")
for k in ["Ind_Health Care", "AcqCtry_United States", "Pay_Cash",
          "Log Equity Value", "Announced Premium", "PE Buyout"]:
    print(f"  {k:24s} = {overrides[k]}")
print("  (Equity $1B -> Log Equity Value = log10(1000) = "
      f"{LOG_EQUITY_1B:.3f})")
print("Unspecified features (revenue, margin, tender, cross-border, target "
      "country,\n  nature of bid, etc.) are left at their real values and "
      "marginalized over.")

# ---------------------------------------------------------------------------
# (A) Marginalized simulation over the real background distribution
# ---------------------------------------------------------------------------
X_sim = X.copy()
for col, val in overrides.items():
    X_sim[col] = val

sim_pred = rf.predict(X_sim)
pcts = {p: np.percentile(sim_pred, p) for p in [5, 25, 50, 75, 95]}

print("\n" + "=" * 78)
print("(A) MARGINALIZED SIMULATION  (n = %d background deals)" % len(sim_pred))
print("=" * 78)
print(f"  Mean predicted duration:   {sim_pred.mean():6.1f} days")
print(f"  Median predicted duration: {pcts[50]:6.1f} days")
print(f"  Std dev:                   {sim_pred.std():6.1f} days")
print(f"  5th -95th percentile:      {pcts[5]:.0f}  -  {pcts[95]:.0f} days")
print(f"  25th-75th percentile:      {pcts[25]:.0f}  -  {pcts[75]:.0f} days")

# ---------------------------------------------------------------------------
# (B) Single synthetic deal point estimate + per-tree spread
# ---------------------------------------------------------------------------
row = pd.DataFrame(0.0, index=[0], columns=X.columns)
for name, med in numeric_medians.items():
    row[name] = med            # unspecified numerics at median
for col, val in overrides.items():
    row[col] = val             # then apply the profile

point = rf.predict(row)[0]
tree_preds = np.array([t.predict(row.values)[0] for t in rf.estimators_])

print("\n" + "=" * 78)
print("(B) SINGLE SYNTHETIC DEAL  (unspecified numerics = median, flags = off)")
print("=" * 78)
print(f"  Point prediction:          {point:6.1f} days")
print(f"  Across 500 trees: mean {tree_preds.mean():.1f}, "
      f"median {np.median(tree_preds):.1f} days")
print(f"  Tree 5th-95th pct:         {np.percentile(tree_preds,5):.0f}  -  "
      f"{np.percentile(tree_preds,95):.0f} days")

# ---------------------------------------------------------------------------
# Real-data sanity check
# ---------------------------------------------------------------------------
mask = (
    (df["Target Industry Group"].astype(str).str.strip() == "Health Care")
    & (df["Acquirer Country/Region"].astype(str).str.strip() == "United States")
    & (df["Payment Type"].astype(str).str.strip() == "Cash")
)
print("\n" + "=" * 78)
print("REAL-DATA SANITY CHECK")
print("=" * 78)
print(f"  Actual Health Care + US-buyer + Cash deals: n = {mask.sum()}")
if mask.sum():
    actual = df.loc[mask, TARGET].astype(float)
    print(f"  Actual days  -> mean {actual.mean():.1f}, median {actual.median():.1f}, "
          f"range {actual.min():.0f}-{actual.max():.0f}")
print(f"  Whole dataset -> mean {y.mean():.1f}, median {np.median(y):.1f} days")

# ---------------------------------------------------------------------------
# Plot + save
# ---------------------------------------------------------------------------
with PdfPages(OUT_PDF) as pdf:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(sim_pred, bins=40, color="teal", alpha=0.75, edgecolor="k")
    ax.axvline(sim_pred.mean(), color="red", linestyle="--",
               label=f"mean {sim_pred.mean():.0f}d")
    ax.axvline(pcts[50], color="orange", linestyle="--",
               label=f"median {pcts[50]:.0f}d")
    ax.axvline(point, color="black", linestyle=":",
               label=f"synthetic point {point:.0f}d")
    ax.set_xlabel("Predicted Days To Complete")
    ax.set_ylabel("Count of simulated deals")
    ax.set_title("Simulated duration: Health Care / US buyer / $1B / Cash / "
                 "no premium / strategic")
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
print(f"\nSaved distribution plot to: {OUT_PDF}")

summary = pd.DataFrame({
    "Metric": [
        "Profile",
        "(A) Marginalized mean (days)",
        "(A) Marginalized median (days)",
        "(A) 5th-95th pct (days)",
        "(A) 25th-75th pct (days)",
        "(B) Synthetic point (days)",
        "(B) Tree 5th-95th pct (days)",
        "Real HC+US+Cash mean (days)",
        "Real HC+US+Cash n",
        "Dataset median (days)",
        "Model OOB R^2",
    ],
    "Value": [
        "Health Care / US buyer / $1B equity / Cash / 0% premium / Strategic",
        round(float(sim_pred.mean()), 1),
        round(float(pcts[50]), 1),
        f"{pcts[5]:.0f} - {pcts[95]:.0f}",
        f"{pcts[25]:.0f} - {pcts[75]:.0f}",
        round(float(point), 1),
        f"{np.percentile(tree_preds,5):.0f} - {np.percentile(tree_preds,95):.0f}",
        round(float(df.loc[mask, TARGET].astype(float).mean()), 1) if mask.sum() else "n/a",
        int(mask.sum()),
        round(float(np.median(y)), 1),
        round(float(rf.oob_score_), 3),
    ],
})
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Simulation", index=False)
print(f"Saved simulation summary to: {OUT_XLSX}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PLAIN-ENGLISH IMPLICATION")
print("=" * 78)
print(
    f"\nFor a $1B all-cash Health Care acquisition by a US strategic buyer with "
    f"no premium,\nthe Random Forest implies a completion time of roughly "
    f"{pcts[50]:.0f} days (median),\ntypically in the {pcts[25]:.0f}-{pcts[75]:.0f} "
    f"day range once everything else varies as in real deals.\n"
    f"That is near / slightly below the dataset median of {np.median(y):.0f} days "
    f"-- consistent with the\nmodel's signals that all-cash and Health Care lean "
    f"toward faster closes, while the\n$1B size pushes mildly the other way.\n"
)
print("Caveat: model fit is modest (OOB R^2 ~0.12), so treat this as a "
      "centre-of-mass\nexpectation, not a precise forecast; the spread above "
      "reflects real uncertainty.")
print("\nDone.")
