"""Regenerate a portable ensemble pickle and full-feature interpretability plots.

Loading the pickle from any context (not just `__main__`) requires the ensemble to be
saved with its class resolving to this module, so we refit+resave here in module context.
Then we render importance and SHAP over ALL features for every ensemble component.
"""
import pickle, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, shap

from xgboost_ma_completion_model import build_features, CATEGORICAL_FEATURES, FILE_PATH, SHEET_NAME, TARGET
from ma_completion_twostage_ensemble import TwoStageEnsemble, DEFAULT_THRESHOLD, ENSEMBLE_PATH

warnings.filterwarnings("ignore")

raw = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
X = build_features(raw)
for c in CATEGORICAL_FEATURES:
    X[c] = X[c].astype("category")
y = raw[TARGET].astype(float)
N = X.shape[1]

# ---- Refit + resave a PORTABLE pickle (class __module__ is this package, not __main__) ----
ens = TwoStageEnsemble(threshold=DEFAULT_THRESHOLD).fit(X, y)
with open(ENSEMBLE_PATH, "wb") as f:
    pickle.dump(ens, f)
print(f"Re-saved portable ensemble -> {ENSEMBLE_PATH}")

te = pd.to_datetime(raw["Announce Date"]).dt.year >= 2024
Xte = X[te]
keys = list(X.columns)

# ---- 1. ALL-feature regime-weighted gain importance ----
def gain(mdl):
    return pd.Series({k: mdl.get_booster().get_score(importance_type="gain").get(k, 0.0) for k in keys})
ws, wl = 1 - ens._fitted_long_rate, ens._fitted_long_rate
agg = (ws * gain(ens.expert_short["xgb_log"]) + wl * gain(ens.expert_long["xgb_log"])).sort_values(ascending=False)
plt.figure(figsize=(10, 10))
agg.iloc[::-1].plot(kind="barh", color="#2c7fb8")
plt.xlabel("Regime-weighted gain")
plt.title(f"ALL {N} Feature Importances (two-stage ensemble)")
plt.tight_layout(); plt.savefig("all_features_importance.png", dpi=150); plt.close()
zero = [k for k in keys if agg[k] == 0]
print("Saved all_features_importance.png | features with 0 gain:", zero or "none")

# ---- 2/3/4. SHAP over ALL features for each key component ----
def shap_all(model, title, path):
    sv = shap.TreeExplainer(model).shap_values(Xte)
    plt.figure(); shap.summary_plot(sv, Xte, max_display=N, show=False, plot_size=(11, 12))
    plt.title(title, fontsize=13); plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(); print("Saved", path)

shap_all(ens.expert_short["xgb_log"], f"SHAP - all {N} features - SHORT-regime expert (bulk of deals)", "all_features_shap_short.png")
shap_all(ens.expert_long["xgb_log"],  f"SHAP - all {N} features - LONG-regime expert (regulatory/slow deals)", "all_features_shap_long.png")
shap_all(ens.clf_xgb,                 f"SHAP - all {N} features - Stage-1 P(long) router", "all_features_shap_router.png")

print("\nFull feature ranking (regime-weighted gain):")
for i, (k, v) in enumerate(agg.items(), 1):
    print(f"  {i:2}. {k:<40} {v:8.3f}")
