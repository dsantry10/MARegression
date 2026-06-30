"""
Random Forest simulation (LARGER dataset, 1,170 deals) - profile #2
===================================================================
Requested profile:
    - Target Industry Group : Consumer Discretionary (Automobiles & Components)
    - Buyer                  : US strategic buyer
    - Equity Value           : $2.431B  -> Log Equity Value = log10(2431) = 3.386
    - Consideration          : Stock
    - Premium                : 18%
    - Buyer type             : Strategic  (PE Buyout = No)

Model: same Random Forest / feature engineering as
ma_days_to_complete_random_forest.py (1,170 deals; target = Days, raw).

Unlike the prior (smaller) model, this one DOES use Announced Premium and
Acquirer Country, so US-buyer and the 18% premium are real inputs here.

Industry note: the dataset has no "Automobiles & Components" group. Autos are a
consumer DISCRETIONARY PRODUCT, so we map the profile to
"Consumer Discretionary Products" (primary) and also report
"Consumer Discretionary Services" as an alternate.

Method (as in ma_simulate_profile.py):
  (A) Marginalized simulation: overwrite ONLY the specified attributes on all
      1,170 real rows, predict, summarize the distribution (averages over the
      unspecified features -- revenue, margin, tender, target country, etc.).
  (B) Single synthetic deal: unspecified numerics at median, flags off, profile
      applied; report point + per-tree spread.
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
OUT_PDF = "ma_profile_simulation_large2.pdf"
OUT_XLSX = "ma_profile_simulation_large2.xlsx"
ANNOUNCE_DATE = pd.Timestamp("2026-06-29")

EQUITY_MM = 2431.0
LOG_EQUITY = np.log10(EQUITY_MM)   # dataset uses log10 of $mm -> 3.386
PREMIUM = 18.0
PRIMARY_IND = "Consumer Discretionary Products"   # autos -> products
ALT_IND = "Consumer Discretionary Services"

# ---------------------------------------------------------------------------
# Rebuild features identically to ma_days_to_complete_random_forest.py
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

binary_cols = ["Additional Stake Purchase", "Competing Bid", "Cross Border",
               "Going Private", "PE Buyout", "Tender Offer"]
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

rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, oob_score=True,
                           random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X, y)
print(f"Larger-dataset RF refit: {X.shape[0]} deals x {X.shape[1]} features "
      f"(OOB R^2 = {rf.oob_score_:.3f})")
print(f"$2.431B -> Log Equity Value = log10(2431) = {LOG_EQUITY:.3f}; "
      f"Premium = {PREMIUM}%")
print("This model DOES use premium and acquirer country (unlike the prior one).")
print(f"Industry: dataset has no 'Automobiles & Components'; mapping autos -> "
      f"'{PRIMARY_IND}'.\n")


def build_overrides(industry_group):
    ov = {
        "Log Equity Value": LOG_EQUITY,
        "Announced Premium": PREMIUM,
        "PE Buyout": 0.0,               # strategic buyer
    }
    if "Announced Premium_missing" in X.columns:
        ov["Announced Premium_missing"] = 0.0
    if "Log Equity Value_missing" in X.columns:
        ov["Log Equity Value_missing"] = 0.0

    def set_exclusive(prefix, chosen):
        sibs = [c for c in X.columns if c.startswith(prefix + "_")]
        tgt = f"{prefix}_{chosen}"
        assert tgt in sibs, f"Missing {tgt}; have {sibs}"
        for c in sibs:
            ov[c] = 1.0 if c == tgt else 0.0

    set_exclusive("Pay", "Stock")
    set_exclusive("Ind", industry_group)
    set_exclusive("AcqCtry", "United States")
    return ov


def simulate(industry_group):
    ov = build_overrides(industry_group)
    # (A) marginalized over real background rows
    X_sim = X.copy()
    for col, val in ov.items():
        X_sim[col] = val
    marg = rf.predict(X_sim)
    pm = {q: float(np.percentile(marg, q)) for q in [5, 25, 50, 75, 95]}
    # (B) single synthetic deal
    row = pd.DataFrame(0.0, index=[0], columns=X.columns)
    for name, med in numeric_medians.items():
        row[name] = med
    for col, val in ov.items():
        row[col] = val
    point = float(rf.predict(row)[0])
    tree = np.array([t.predict(row.values)[0] for t in rf.estimators_])
    return marg, pm, point, tree


results = {}
for grp in [PRIMARY_IND, ALT_IND]:
    marg, pm, point, tree = simulate(grp)
    results[grp] = (marg, pm, point, tree)
    close = (ANNOUNCE_DATE + pd.Timedelta(days=round(pm[50]))).date()
    tag = "PRIMARY (autos)" if grp == PRIMARY_IND else "alternate"
    print("=" * 78)
    print(f"SCENARIO [{tag}]: {grp}  (Stock / Strategic / US / $2.431B / 18% prem)")
    print("=" * 78)
    print(f"  (A) Marginalized mean / median:  {marg.mean():.1f} / {pm[50]:.1f} days"
          f"   -> close ~ {close}")
    print(f"      25th-75th pct:               {pm[25]:.0f} - {pm[75]:.0f} days")
    print(f"      5th-95th pct:                {pm[5]:.0f} - {pm[95]:.0f} days")
    print(f"  (B) Synthetic point:             {point:.1f} days "
          f"(per-tree 5-95: {np.percentile(tree,5):.0f}-{np.percentile(tree,95):.0f})")

    # real comparable
    m = ((df["Target Industry Group"].astype(str).str.strip() == grp)
         & (df["Payment Type"].astype(str).str.strip() == "Stock")
         & (df["Acquirer Country/Region"].astype(str).str.strip() == "United States"))
    if m.sum():
        act = df.loc[m, TARGET].astype(float)
        print(f"  Real {grp} + Stock + US buyer: n={m.sum()}, "
              f"mean {act.mean():.0f}, median {act.median():.0f} days")
    print()

print(f"Whole dataset -> mean {y.mean():.0f}, median {np.median(y):.0f} days")

# ---------------------------------------------------------------------------
# Plot + save
# ---------------------------------------------------------------------------
with PdfPages(OUT_PDF) as pdf:
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {PRIMARY_IND: "steelblue", ALT_IND: "darkorange"}
    for grp in [PRIMARY_IND, ALT_IND]:
        marg, pm, point, tree = results[grp]
        ax.hist(marg, bins=40, alpha=0.55, edgecolor="k", color=colors[grp],
                label=f"{grp} (median {pm[50]:.0f}d)")
        ax.axvline(pm[50], color=colors[grp], linestyle="--")
    ax.set_xlabel("Predicted Days To Complete")
    ax.set_ylabel("Count of simulated deals")
    ax.set_title("Larger dataset: Consumer Discretionary / Stock / US / Strategic\n"
                 "$2.431B / 18% premium  (marginalized simulation)")
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
print(f"\nSaved plot to: {OUT_PDF}")

rows_out = []
for grp in [PRIMARY_IND, ALT_IND]:
    marg, pm, point, tree = results[grp]
    m = ((df["Target Industry Group"].astype(str).str.strip() == grp)
         & (df["Payment Type"].astype(str).str.strip() == "Stock")
         & (df["Acquirer Country/Region"].astype(str).str.strip() == "United States"))
    act = df.loc[m, TARGET].astype(float)
    rows_out.append({
        "Scenario": grp,
        "Marginalized mean (days)": round(float(marg.mean()), 1),
        "Marginalized median (days)": round(pm[50], 1),
        "Close (announce 2026-06-29)":
            str((ANNOUNCE_DATE + pd.Timedelta(days=round(pm[50]))).date()),
        "25-75 pct": f"{pm[25]:.0f}-{pm[75]:.0f}",
        "5-95 pct": f"{pm[5]:.0f}-{pm[95]:.0f}",
        "Synthetic point (days)": round(point, 1),
        "Real comp n": int(m.sum()),
        "Real comp mean (days)": round(float(act.mean()), 1) if m.sum() else "n/a",
    })
summary = pd.DataFrame(rows_out)
meta = pd.DataFrame({"Note": [
    "Equity $2.431B -> Log Equity Value = log10(2431) = %.3f" % LOG_EQUITY,
    "Announced Premium = 18.0 (this model uses premium)",
    "Payment Type = Stock; Acquirer Country = United States; PE Buyout = No",
    "No 'Automobiles & Components' group; autos mapped to Consumer Discretionary Products",
    "Model OOB R^2 = %.3f" % rf.oob_score_,
    "Dataset median = %.0f days" % np.median(y),
]})
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Scenarios", index=False)
    meta.to_excel(writer, sheet_name="Notes", index=False)
print(f"Saved summary to: {OUT_XLSX}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("WHAT THIS IMPLIES ABOUT CLOSE")
print("=" * 78)
mp, pmp, ptp, _ = results[PRIMARY_IND]
close_p = (ANNOUNCE_DATE + pd.Timedelta(days=round(pmp[50]))).date()
print(
    f"\nOn the larger dataset, this $2.431B Stock-funded US strategic Consumer "
    f"Discretionary\n(autos -> Products) deal is implied to take ~{pmp[50]:.0f} days "
    f"(mean {mp.mean():.0f}), typically\n{pmp[25]:.0f}-{pmp[75]:.0f} days. Announcing "
    f"{ANNOUNCE_DATE.date()}, that points to a close around {close_p}.\n\n"
    f"Cross-model read: the prior smaller model put a similar (Stock, $2.431B, "
    f"strategic)\nConsumer Discretionary deal at ~105-107 days; the larger model "
    f"-- which also accounts\nfor the US buyer and 18% premium -- lands in a "
    f"comparable {pmp[25]:.0f}-{pmp[75]:.0f} day range.\n"
)
print("Caveats:\n"
      f"  - OOB R^2 ~ {rf.oob_score_:.2f}: modest fit; central expectation, not a "
      "precise date.\n"
      "  - No autos-specific industry; 'Consumer Discretionary Products' is the "
      "closest proxy.\n"
      "  - Higher premium and stock funding both lean toward longer timelines in "
      "this data.")
print("\nDone.")
