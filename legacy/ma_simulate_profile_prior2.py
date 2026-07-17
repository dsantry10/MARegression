"""
Random Forest simulation (PRIOR / smaller dataset) - profile #2
===============================================================
Requested profile:
    - Target Industry Group : Consumer Discretionary
    - Buyer                  : US strategic buyer
    - Equity Value           : $2.431B  -> log_equity = ln(2431) = 7.796
    - Consideration          : Stock
    - Premium                : 18%
    - Buyer type             : Strategic

Model: the first Random Forest (ma_deal_timing_random_forest.py), 152 completed
deals since 2025. Features = log(equity), strategic, Consideration dummies
(Cash = ref), Industry dummies (Health Care = ref). Target = log(Days).

Notes / structural limits of this model:
  * "US buyer" and "18% premium" are NOT features here, so they cannot be set
    (the larger-dataset model does use country & premium; this one does not).
  * Unlike the Health Care/Cash profile, Stock and Consumer Discretionary are
    NOT reference levels, so the relevant dummies are switched on. Stock is the
    key driver and pushes the estimate UP (stock deals close slower).
  * The dataset splits Consumer Discretionary into "Products" and "Services";
    we simulate BOTH (the spec did not disambiguate).

All features are pinned, so each scenario is a point prediction plus the spread
across the 500 trees. Predictions are log-days, back-transformed with exp().
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
OUT_PDF = "ma_profile_simulation_prior2.pdf"
OUT_XLSX = "ma_profile_simulation_prior2.xlsx"

EQUITY_MM = 2431.0
LOG_EQUITY = np.log(EQUITY_MM)         # natural log of $mm -> 7.796
ANNOUNCE_DATE = pd.Timestamp("2026-06-29")

# ---------------------------------------------------------------------------
# Rebuild data + features exactly as in ma_deal_timing_random_forest.py
# ---------------------------------------------------------------------------
raw = pd.read_excel(INPUT_FILE, sheet_name=SHEET)
df = raw[raw["Deal Status"] == "Completed"].copy()
df[CONSIDERATION] = df[CONSIDERATION].replace({"Stock Tender": "Stock"})
df = df.dropna(subset=[DEP, CONSIDERATION, EQUITY, STRAT, INDUSTRY])

y = np.log(df[DEP].values)

X = pd.DataFrame(index=df.index)
X["log_equity"] = np.log(df[EQUITY].values)
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
print(f"$2.431B equity -> log_equity = ln({EQUITY_MM:.0f}) = {LOG_EQUITY:.3f}")
print("NOTE: 'US buyer' and '18% premium' are not features in this model.\n")


def simulate(industry_group):
    """Build the profile row for a given CD industry group and predict."""
    row = pd.DataFrame(0.0, index=[0], columns=X.columns)
    row["log_equity"] = LOG_EQUITY
    row["strategic"] = 1
    stock_col = f"Consideration_Stock"
    assert stock_col in X.columns, "Stock dummy missing"
    row[stock_col] = 1                               # Stock consideration
    ind_col = f"Industry_{industry_group}"
    assert ind_col in X.columns, f"{ind_col} missing; have {[c for c in X.columns if c.startswith('Industry_')]}"
    row[ind_col] = 1

    log_point = rf.predict(row)[0]
    point_days = float(np.exp(log_point))
    tree_days = np.exp(np.array([t.predict(row.values)[0] for t in rf.estimators_]))
    p = {q: float(np.percentile(tree_days, q)) for q in [5, 25, 50, 75, 95]}
    return point_days, tree_days, p


scenarios = ["Consumer Discretionary Products", "Consumer Discretionary Services"]
results = {}

for grp in scenarios:
    point, trees, p = simulate(grp)
    results[grp] = (point, trees, p)
    close_date = (ANNOUNCE_DATE + pd.Timedelta(days=round(point))).date()
    print("=" * 78)
    print(f"SCENARIO: {grp}  (Stock / Strategic / $2.431B)")
    print("=" * 78)
    print(f"  Point prediction:     {point:6.1f} days   -> close ~ {close_date}")
    print(f"  Tree median / mean:   {p[50]:.1f} / {trees.mean():.1f} days")
    print(f"  Tree 25th-75th pct:   {p[25]:.0f} - {p[75]:.0f} days")
    print(f"  Tree 5th-95th pct:    {p[5]:.0f} - {p[95]:.0f} days")

    # real comparable
    m = ((df[INDUSTRY] == grp) & (df[CONSIDERATION] == "Stock")
         & (df[STRAT] == "Strategic"))
    if m.sum():
        act = df.loc[m, DEP].astype(float)
        print(f"  Real {grp} + Stock + Strategic: n={m.sum()}, "
              f"mean {act.mean():.0f}, median {act.median():.0f} days "
              f"(range {act.min():.0f}-{act.max():.0f})")
    print()

# Broad comparable: any Consumer Discretionary + Stock
mall = (df[INDUSTRY].str.contains("Consumer Discretionary")
        & (df[CONSIDERATION] == "Stock"))
print("Broad real comparable -- all Consumer Discretionary + Stock completed: "
      f"n={mall.sum()}, mean {df.loc[mall, DEP].astype(float).mean():.0f}, "
      f"median {df.loc[mall, DEP].astype(float).median():.0f} days")
print(f"All completed deals -> median {np.median(np.exp(y)):.0f} days")

# ---------------------------------------------------------------------------
# Plot + save
# ---------------------------------------------------------------------------
with PdfPages(OUT_PDF) as pdf:
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"Consumer Discretionary Products": "steelblue",
              "Consumer Discretionary Services": "darkorange"}
    for grp in scenarios:
        point, trees, p = results[grp]
        ax.hist(trees, bins=30, alpha=0.55, edgecolor="k",
                color=colors[grp], label=f"{grp} (point {point:.0f}d)")
        ax.axvline(point, color=colors[grp], linestyle="--")
    ax.set_xlabel("Predicted Days to Complete (per tree)")
    ax.set_ylabel("Count of trees")
    ax.set_title("Prior dataset: Consumer Discretionary / Stock / Strategic / "
                 "$2.431B\n(500-tree prediction spread)")
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
print(f"\nSaved plot to: {OUT_PDF}")

rows_out = []
for grp in scenarios:
    point, trees, p = results[grp]
    m = ((df[INDUSTRY] == grp) & (df[CONSIDERATION] == "Stock")
         & (df[STRAT] == "Strategic"))
    act = df.loc[m, DEP].astype(float)
    rows_out.append({
        "Scenario": grp,
        "Point (days)": round(point, 1),
        "Close date (announce 2026-06-29)":
            str((ANNOUNCE_DATE + pd.Timedelta(days=round(point))).date()),
        "Tree median (days)": round(p[50], 1),
        "Tree 25-75 pct": f"{p[25]:.0f}-{p[75]:.0f}",
        "Tree 5-95 pct": f"{p[5]:.0f}-{p[95]:.0f}",
        "Real comparable n": int(m.sum()),
        "Real comparable mean (days)": round(float(act.mean()), 1) if m.sum() else "n/a",
    })
summary = pd.DataFrame(rows_out)
meta = pd.DataFrame({
    "Note": [
        "Equity $2.431B -> log_equity = ln(2431) = %.3f" % LOG_EQUITY,
        "Stock consideration set (Cash is reference)",
        "Strategic = 1",
        "NOT modeled: US buyer (no country feature), 18%% premium (no premium feature)",
        "Model OOB R^2 = %.3f" % rf.oob_score_,
        "All-completed median = %.0f days" % np.median(np.exp(y)),
    ]
})
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Scenarios", index=False)
    meta.to_excel(writer, sheet_name="Notes", index=False)
print(f"Saved summary to: {OUT_XLSX}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("WHAT THIS IMPLIES ABOUT CLOSE")
print("=" * 78)
pp, _, ppd = results["Consumer Discretionary Products"]
ps, _, psd = results["Consumer Discretionary Services"]
lo, hi = sorted([pp, ps])
print(
    f"\nOn the prior (2025-present) dataset, this Stock-funded, $2.431B strategic "
    f"Consumer\nDiscretionary deal is implied to take roughly {lo:.0f}-{hi:.0f} days "
    f"depending on sub-group:\n"
    f"  - Consumer Discretionary Products : ~{pp:.0f} days "
    f"(close ~ {(ANNOUNCE_DATE + pd.Timedelta(days=round(pp))).date()})\n"
    f"  - Consumer Discretionary Services : ~{ps:.0f} days "
    f"(close ~ {(ANNOUNCE_DATE + pd.Timedelta(days=round(ps))).date()})\n\n"
    f"This is materially LONGER than the $1B Health-Care/all-cash case (~70 days) "
    f"-- the\ndifference is driven almost entirely by STOCK consideration (stock "
    f"deals need a stock\nregistration / shareholder vote and historically close "
    f"slower) plus the larger size.\n"
)
print("Caveats:\n"
      f"  - OOB R^2 ~ {rf.oob_score_:.2f}; small-sample industry (CD Products n=8, "
      "Services n=5 completed),\n    so treat as a central expectation with wide "
      "uncertainty.\n"
      "  - 'US buyer' and '18% premium' do not enter this model; if you want them "
      "to matter,\n    use the larger-dataset simulation (ma_simulate_profile.py), "
      "which includes both.")
print("\nDone.")
