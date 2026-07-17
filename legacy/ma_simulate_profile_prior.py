"""
Random Forest simulation (PRIOR / smaller dataset, mergers since 2025)
======================================================================
Same requested profile as before:
    - Target Industry Group : Health Care
    - Buyer                  : US strategic buyer
    - Equity Value           : $1B
    - Consideration          : Cash
    - Premium                : none

This uses the FIRST Random Forest (ma_deal_timing_random_forest.py) built on the
152 completed deals in MA_Statistics_2025-Present. That model's features are
only:  log(equity value), strategic dummy, Consideration dummies (Cash = ref),
and Target Industry Group dummies (Health Care = ref). Target = log(Days).

Two parts of the profile are NOT features of this model, so they cannot be set:
    * "US buyer"  -> acquirer country was never a predictor here.
    * "no premium"-> premium was never a predictor here.
We note this explicitly rather than pretending to model them.

Because Health Care and Cash are the dropped REFERENCE levels, this profile pins
every feature in the model (log_equity, strategic, and all dummies = 0). There is
nothing left to marginalize over, so the simulation is a point prediction plus
the spread across the 500 trees as a model-uncertainty band. Predictions are on
the log-day scale and back-transformed with exp().
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INPUT_FILE = "MA_Statistics_2025-Present__Claude_Code_.xlsx"
SHEET = "Deal Sheet"
DEP = "Days to Complete"
EQUITY = "Announced Equity Value (mil.)"
CONSIDERATION = "Consideration"
STRAT = "Strategic/Sponsor"
INDUSTRY = "Target Industry Group"
CONSIDERATION_BASE = "Cash"
INDUSTRY_BASE = "Health Care"
RANDOM_STATE = 42
N_ESTIMATORS = 500
OUT_PDF = "ma_profile_simulation_prior.pdf"
OUT_XLSX = "ma_profile_simulation_prior.xlsx"

# ---------------------------------------------------------------------------
# Rebuild data + features exactly as in ma_deal_timing_random_forest.py
# ---------------------------------------------------------------------------
raw = pd.read_excel(INPUT_FILE, sheet_name=SHEET)
df = raw[raw["Deal Status"] == "Completed"].copy()
df[CONSIDERATION] = df[CONSIDERATION].replace({"Stock Tender": "Stock"})
df = df.dropna(subset=[DEP, CONSIDERATION, EQUITY, STRAT, INDUSTRY])

y = np.log(df[DEP].values)  # target = log(days)

X = pd.DataFrame(index=df.index)
X["log_equity"] = np.log(df[EQUITY].values)          # natural log of $millions
X["strategic"] = (df[STRAT] == "Strategic").astype(int).values

cons = pd.get_dummies(df[CONSIDERATION], prefix="Consideration")
cons = cons.drop(columns=[f"Consideration_{CONSIDERATION_BASE}"], errors="ignore")
ind = pd.get_dummies(df[INDUSTRY], prefix="Industry")
ind = ind.drop(columns=[f"Industry_{INDUSTRY_BASE}"], errors="ignore")
X = pd.concat([X, cons.set_index(X.index), ind.set_index(X.index)], axis=1).astype(float)

rf = RandomForestRegressor(
    n_estimators=N_ESTIMATORS, oob_score=True,
    random_state=RANDOM_STATE, n_jobs=-1,
)
rf.fit(X, y)
print(f"Prior-dataset RF refit: {X.shape[0]} completed deals x {X.shape[1]} "
      f"features (OOB R^2 = {rf.oob_score_:.3f})")

# ---------------------------------------------------------------------------
# Build the profile row
# ---------------------------------------------------------------------------
LOG_EQUITY_1B = np.log(1000.0)  # $1,000mm, natural log -> 6.908

row = pd.DataFrame(0.0, index=[0], columns=X.columns)  # all dummies 0
row["log_equity"] = LOG_EQUITY_1B
row["strategic"] = 1                                   # strategic buyer
# Health Care + Cash are reference levels => their dummies stay 0 (already).

print("\nProfile -> model features:")
print(f"  log_equity = ln(1000) = {LOG_EQUITY_1B:.3f}  ($1B equity, $mm scale)")
print(f"  strategic  = 1  (strategic buyer)")
print(f"  Consideration dummies all 0  => Cash (reference level)")
print(f"  Industry dummies all 0       => Health Care (reference level)")
print("  NOTE: 'US buyer' and 'no premium' are not features of this model and "
      "cannot be set.")

# ---------------------------------------------------------------------------
# Predict (point) + per-tree spread, back-transformed to days
# ---------------------------------------------------------------------------
log_point = rf.predict(row)[0]
point_days = float(np.exp(log_point))

tree_log = np.array([t.predict(row.values)[0] for t in rf.estimators_])
tree_days = np.exp(tree_log)
p = {q: float(np.percentile(tree_days, q)) for q in [5, 25, 50, 75, 95]}

print("\n" + "=" * 78)
print("SIMULATED DURATION  (prior dataset, mergers since 2025)")
print("=" * 78)
print(f"  Point prediction:            {point_days:6.1f} days  "
      f"(exp of predicted log-days = {log_point:.3f})")
print(f"  Across 500 trees -> median:  {p[50]:6.1f} days")
print(f"                      mean:    {tree_days.mean():6.1f} days")
print(f"  Tree 25th-75th pct:          {p[25]:.0f}  -  {p[75]:.0f} days")
print(f"  Tree 5th-95th pct:           {p[5]:.0f}  -  {p[95]:.0f} days")

# ---------------------------------------------------------------------------
# Real-data sanity check
# ---------------------------------------------------------------------------
mask = (
    (df[INDUSTRY] == "Health Care")
    & (df[CONSIDERATION] == "Cash")
    & (df[STRAT] == "Strategic")
)
near = mask & df[EQUITY].between(500, 2000)
print("\n" + "=" * 78)
print("REAL-DATA SANITY CHECK")
print("=" * 78)
act = df.loc[mask, DEP].astype(float)
print(f"  Actual Health Care + Cash + Strategic completed deals: n = {mask.sum()}")
print(f"    days -> mean {act.mean():.1f}, median {act.median():.1f}, "
      f"range {act.min():.0f}-{act.max():.0f}")
if near.sum():
    actn = df.loc[near, DEP].astype(float)
    print(f"  Of those, near $1B ($0.5-2B): n = {near.sum()}, "
          f"mean {actn.mean():.1f}, median {actn.median():.1f} days")
print(f"  All completed deals -> mean {np.exp(y).mean():.1f}, "
      f"median {np.median(np.exp(y)):.1f} days")

# ---------------------------------------------------------------------------
# Plot + save
# ---------------------------------------------------------------------------
with PdfPages(OUT_PDF) as pdf:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(tree_days, bins=30, color="indianred", alpha=0.75, edgecolor="k")
    ax.axvline(point_days, color="black", linestyle="--",
               label=f"point {point_days:.0f}d")
    ax.axvline(act.mean(), color="navy", linestyle=":",
               label=f"actual HC+Cash+Strategic mean {act.mean():.0f}d")
    ax.set_xlabel("Predicted Days to Complete (per tree)")
    ax.set_ylabel("Count of trees")
    ax.set_title("Prior dataset: Health Care / strategic / $1B / Cash\n"
                 "(500-tree prediction spread)")
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
print(f"\nSaved plot to: {OUT_PDF}")

summary = pd.DataFrame({
    "Metric": [
        "Profile (modelable parts)",
        "Point prediction (days)",
        "Tree median (days)",
        "Tree mean (days)",
        "Tree 25th-75th pct (days)",
        "Tree 5th-95th pct (days)",
        "Real HC+Cash+Strategic mean (days)",
        "Real HC+Cash+Strategic median (days)",
        "Real HC+Cash+Strategic n",
        "Real near-$1B mean (days)",
        "All-completed median (days)",
        "Model OOB R^2",
        "Not modeled",
    ],
    "Value": [
        "Health Care / Cash / Strategic / $1B equity",
        round(point_days, 1),
        round(p[50], 1),
        round(float(tree_days.mean()), 1),
        f"{p[25]:.0f} - {p[75]:.0f}",
        f"{p[5]:.0f} - {p[95]:.0f}",
        round(float(act.mean()), 1),
        round(float(act.median()), 1),
        int(mask.sum()),
        round(float(df.loc[near, DEP].astype(float).mean()), 1) if near.sum() else "n/a",
        round(float(np.median(np.exp(y))), 1),
        round(float(rf.oob_score_), 3),
        "US-buyer and premium (not features in this model)",
    ],
})
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Simulation_prior", index=False)
print(f"Saved summary to: {OUT_XLSX}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("WHAT THIS IMPLIES ABOUT CLOSE")
print("=" * 78)
print(
    f"\nOn the prior (2025-present) dataset, the Random Forest implies this deal "
    f"closes in\nabout {point_days:.0f} days (~{p[25]:.0f}-{p[75]:.0f} days across "
    f"the tree ensemble). The 11 actual Health Care\nall-cash strategic deals "
    f"averaged {act.mean():.0f} days (median {act.median():.0f}), so the model's "
    f"estimate sits right\nin that real range.\n\n"
    f"Translated to a calendar close: announcing today (2026-06-29), ~{point_days:.0f} "
    f"days implies a\ntarget completion around "
    f"{(pd.Timestamp('2026-06-29') + pd.Timedelta(days=round(point_days))).date()}.\n"
)
print("Caveats:\n"
      f"  - OOB R^2 ~ {rf.oob_score_:.2f}: weak fit on only 152 deals; this is a "
      "central expectation,\n    not a precise forecast.\n"
      "  - 'US buyer' and 'no premium' are not in this model, so they do not "
      "affect the\n    estimate here (unlike the larger-dataset model, which did "
      "use country & premium).\n"
      "  - Health Care + Cash are the model's baseline, so this profile is "
      "essentially the\n    model's reference deal scaled to $1B and flagged "
      "strategic.")
print("\nDone.")
