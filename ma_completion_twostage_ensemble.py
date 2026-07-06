"""
Two-Stage Blended-Ensemble model for M&A `Business Days To Complete`.
====================================================================

Note: the target is in BUSINESS days (weekdays). Convert to calendar time with ~1.4x
(7/5), or ~21.7 business days per month.

This is the R2-maximizing successor to `xgboost_ma_completion_model.py`. It keeps that
module's strict, leakage-free feature schema (imported directly, single source of truth)
and layers on the methodology chosen after empirical benchmarking:

  * MIXTURE-OF-EXPERTS (two-stage).  A blended classifier estimates P(long) -- the
    probability a deal falls into the slow / regulatory-review regime (Days > threshold).
    Two regime experts (short / long) each predict a duration; the final prediction is the
    probability-weighted blend  yhat = (1-p)*yhat_short + p*yhat_long.  Soft routing (not a
    hard if/else) is used because it degrades gracefully when the classifier is unsure and
    generalizes better out-of-time than hard assignment.

  * BLENDED ENSEMBLE per expert.  Each expert averages three complementary learners:
      - XGBoost, squared-error on log1p(days)      (conditional-mean, tail-damped)
      - HistGradientBoosting, squared-error on log  (decorrelated 2nd implementation)
      - XGBoost, absolute-error on raw days          (conditional-median, outlier-robust)
    Empirically the MAE learner is what protects out-of-time R2 on this long-tailed target,
    so it carries the largest blend weight.

  * WHY NOT plain squared-error on raw days?  The target is extremely right-skewed
    (skew ~3.3, max 1078d). A raw squared-error objective chases a handful of mega-deals
    and posts a NEGATIVE out-of-time R2. Log transform + robust losses are what make R2
    stable here.

EVALUATION (both reported, per the agreed methodology):
  * Repeated K-Fold CV R2 (mean +/- std)  -- the metric the model is tuned against.
  * Honest out-of-time R2 (train <=2023, test >=2024) -- the headline forecast metric.
  * A single-blend baseline is reported alongside for context.

Realistic expectation: R2 ~ 0.26-0.29 out-of-time. The dataset has a genuine signal
ceiling; ~2% of deals run >1 year and dominate the residual sum-of-squares that R2 is
built on. The two-stage design squeezes out the CV edge without inflating variance.

Run:
    python ma_completion_twostage_ensemble.py
"""

# --------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------
import pickle
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedKFold

# Reuse the vetted, leakage-free preprocessing from the original deliverable so both
# scripts share exactly one feature definition.
from xgboost_ma_completion_model import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FILE_PATH,
    SHEET_NAME,
    TARGET,
    build_features,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42

# ---- Methodology hyper-parameters (threshold is re-confirmed against CV at runtime) ----
CANDIDATE_THRESHOLDS = [150, 180, 240]  # days; "long / regulatory-review" regime cutoffs
DEFAULT_THRESHOLD = 180                 # 6 months -- business-meaningful default
# Blend weights: (xgb_log_squared, hgb_log_squared, xgb_raw_mae). MAE-heavy: the robust
# median learner best protects out-of-time R2 on this long-tailed target.
EXPERT_WEIGHTS = (0.30, 0.20, 0.50)

# ---- Regulatory routing floor -------------------------------------------------------
# The learned Stage-1 router under-reacts to regulatory flags (only ~14% of training
# deals carry any flag, so it leans on year/size/industry instead). Empirically, deals
# requiring BOTH SAMR (China) and EC (EU) clearance land in the long regime far more
# often than the router predicts -- this profile ran a median of 140d / mean 184d in the
# training data. We therefore floor P(long) for these deals so the long-regime expert
# receives meaningful weight. This is a deliberate, documented business override of the
# learned probability, not a fitted parameter.
SAMR_EC_FLAGS = ("SAMR", "EC")   # both must be 1 for the floor to apply
SAMR_EC_MIN_P_LONG = 0.20        # minimum routing probability when both flags are on

# ---- Artifact paths ----
ENSEMBLE_PATH = "ma_twostage_ensemble.pkl"
PREPROCESSOR_PATH = "twostage_preprocessor.pkl"
IMPORTANCE_PLOT_PATH = "twostage_feature_importance.png"
SHAP_PLOT_PATH = "twostage_shap_summary.png"


# --------------------------------------------------------------------------------------
# Model factories (well-regularized; selected via repeated-CV comparison)
# --------------------------------------------------------------------------------------
def _xgb_log_squared():
    """XGBoost, squared error on log1p(days) -- conditional mean, tail-damped."""
    return xgb.XGBRegressor(
        tree_method="hist", enable_categorical=True, objective="reg:squarederror",
        n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=3.0, min_child_weight=6,
        random_state=RANDOM_STATE, n_jobs=-1,
    )


def _hgb_log_squared():
    """HistGradientBoosting on log target -- a decorrelated 2nd implementation."""
    return HistGradientBoostingRegressor(
        loss="squared_error", max_depth=4, learning_rate=0.05, max_iter=400,
        l2_regularization=3.0, min_samples_leaf=20,
        categorical_features="from_dtype", random_state=RANDOM_STATE,
    )


def _xgb_raw_mae():
    """XGBoost, absolute error on raw days -- conditional median, outlier-robust."""
    return xgb.XGBRegressor(
        tree_method="hist", enable_categorical=True, objective="reg:absoluteerror",
        n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )


def _xgb_classifier():
    return xgb.XGBClassifier(
        tree_method="hist", enable_categorical=True, n_estimators=300, max_depth=4,
        learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
        min_child_weight=6, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
    )


def _hgb_classifier():
    return HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=3.0,
        min_samples_leaf=20, categorical_features="from_dtype", random_state=RANDOM_STATE,
    )


# --------------------------------------------------------------------------------------
# The two-stage blended ensemble
# --------------------------------------------------------------------------------------
class TwoStageEnsemble:
    """Mixture-of-experts: blended P(long) classifier gates two blended regime experts."""

    def __init__(self, threshold=DEFAULT_THRESHOLD, weights=EXPERT_WEIGHTS,
                 reg_floor=SAMR_EC_MIN_P_LONG):
        self.threshold = threshold
        self.weights = weights
        # Minimum P(long) enforced for deals carrying both SAMR and EC flags.
        self.reg_floor = reg_floor

    # ---- experts ----------------------------------------------------------------------
    @staticmethod
    def _fit_expert(X, y):
        """Fit the three-learner blend for one regime. Returns a dict of fitted models."""
        logy = np.log1p(y)
        return {
            "xgb_log": _xgb_log_squared().fit(X, logy),
            "hgb_log": _hgb_log_squared().fit(X, logy),
            "xgb_mae": _xgb_raw_mae().fit(X, y),  # raw scale
        }

    def _predict_expert(self, expert, X):
        w = self.weights
        return (
            w[0] * np.expm1(expert["xgb_log"].predict(X))
            + w[1] * np.expm1(expert["hgb_log"].predict(X))
            + w[2] * expert["xgb_mae"].predict(X)
        )

    # ---- stage 1 ----------------------------------------------------------------------
    def _apply_reg_floor(self, p, X):
        """Floor P(long) for deals requiring BOTH SAMR and EC regulatory clearance."""
        floor = getattr(self, "reg_floor", SAMR_EC_MIN_P_LONG)
        if floor and all(f in X.columns for f in SAMR_EC_FLAGS):
            both_on = np.logical_and.reduce([X[f].to_numpy() == 1 for f in SAMR_EC_FLAGS])
            p = np.where(both_on, np.maximum(p, floor), p)
        return p

    def _p_long(self, X):
        """Blended probability of the long / regulatory-review regime (with reg floor)."""
        p = 0.5 * self.clf_xgb.predict_proba(X)[:, 1] + 0.5 * self.clf_hgb.predict_proba(X)[:, 1]
        return self._apply_reg_floor(p, X)

    # ---- API --------------------------------------------------------------------------
    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        L = (y > self.threshold).astype(int)
        # Guard: if a fold has too few long deals, widen slightly so the expert can train.
        if L.sum() < 10:
            L = (y > np.quantile(y, 0.85)).astype(int)

        self.clf_xgb = _xgb_classifier().fit(X, L)
        self.clf_hgb = _hgb_classifier().fit(X, L)
        self.expert_short = self._fit_expert(X[L == 0], y[L == 0])
        self.expert_long = self._fit_expert(X[L == 1], y[L == 1])
        self._fitted_long_rate = float(L.mean())
        return self

    def predict(self, X):
        p = self._p_long(X)
        y_short = self._predict_expert(self.expert_short, X)
        y_long = self._predict_expert(self.expert_long, X)
        pred = (1.0 - p) * y_short + p * y_long
        return np.clip(pred, 1.0, None)  # durations are strictly positive


class SingleBlend:
    """Baseline: one blended expert over all deals (no routing). For context only."""

    def __init__(self, weights=EXPERT_WEIGHTS):
        self.weights = weights

    def fit(self, X, y):
        self.expert = TwoStageEnsemble._fit_expert(X, np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        w = self.weights
        e = self.expert
        pred = (
            w[0] * np.expm1(e["xgb_log"].predict(X))
            + w[1] * np.expm1(e["hgb_log"].predict(X))
            + w[2] * e["xgb_mae"].predict(X)
        )
        return np.clip(pred, 1.0, None)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_xy():
    raw = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
    X = build_features(raw)  # strict schema, NaNs preserved, categoricals native
    y = raw[TARGET].astype(float)
    announce_year = pd.to_datetime(raw["Announce Date"], errors="coerce").dt.year

    # Leakage guard (mirrors the base module's assertions).
    assert TARGET not in X.columns
    assert list(X.columns) == FEATURE_COLUMNS
    # Ensure plain category dtype for HistGradientBoosting's from_dtype detection.
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    return raw, X, y, announce_year


# --------------------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------------------
def repeated_cv_r2(model_factory, X, y, n_splits=5, n_repeats=3):
    """Mean +/- std R2 over RepeatedKFold -- the metric we optimize against."""
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    scores = []
    for tr, te in rkf.split(X):
        model = model_factory().fit(X.iloc[tr], y.iloc[tr])
        scores.append(r2_score(y.iloc[te], model.predict(X.iloc[te])))
    return float(np.mean(scores)), float(np.std(scores))


def select_threshold_by_cv(X, y):
    """Pick the long/short regime threshold that maximizes repeated-CV R2."""
    print("Selecting regime threshold by repeated-CV R2 ...")
    best_t, best_mean, best_std = DEFAULT_THRESHOLD, -np.inf, 0.0
    for t in CANDIDATE_THRESHOLDS:
        mean, std = repeated_cv_r2(lambda t=t: TwoStageEnsemble(threshold=t), X, y)
        flag = ""
        if mean > best_mean:
            best_t, best_mean, best_std, flag = t, mean, std, "  <-- best"
        print(f"  threshold={t:>3}d :  CV R2 = {mean:.3f} +/- {std:.3f}{flag}")
    print(f"Chosen threshold: {best_t}d  (CV R2 {best_mean:.3f} +/- {best_std:.3f})")
    return best_t


def report_out_of_time(model, X, y, announce_year):
    tr = announce_year <= 2023
    te = announce_year >= 2024
    fitted = model.fit(X[tr], y[tr])
    pred = fitted.predict(X[te])
    mae = mean_absolute_error(y[te], pred)
    mape = mean_absolute_percentage_error(y[te], pred) * 100
    r2 = r2_score(y[te], pred)
    print(f"  rows: train(<=2023)={tr.sum()}  test(>=2024)={te.sum()}")
    print(f"  R2   = {r2:.3f}   (headline forecast metric)")
    print(f"  MAE  = {mae:.1f} days")
    print(f"  MAPE = {mape:.1f} %")
    # Stage-1 diagnostic: how well did we route?
    if isinstance(fitted, TwoStageEnsemble):
        L_te = (y[te] > fitted.threshold).astype(int)
        if L_te.nunique() > 1:
            auc = roc_auc_score(L_te, fitted._p_long(X[te]))
            print(f"  Stage-1 routing AUC (long regime) = {auc:.3f}")
    return fitted, {"R2": r2, "MAE": mae, "MAPE": mape}


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------
def plot_importance(ensemble, top_n=15, path=IMPORTANCE_PLOT_PATH):
    """Aggregate gain importance across both regime XGB (log-squared) experts."""
    short = ensemble.expert_short["xgb_log"].get_booster().get_score(importance_type="gain")
    long = ensemble.expert_long["xgb_log"].get_booster().get_score(importance_type="gain")
    ws = 1.0 - ensemble._fitted_long_rate
    wl = ensemble._fitted_long_rate
    agg = {}
    for k in set(short) | set(long):
        agg[k] = ws * short.get(k, 0.0) + wl * long.get(k, 0.0)
    imp = pd.Series(agg).sort_values(ascending=False).head(top_n)

    print(f"\nTop {top_n} features (regime-weighted gain):")
    for name, val in imp.items():
        print(f"  {name:<42} {val:10.2f}")

    plt.figure(figsize=(9, 7))
    imp.iloc[::-1].plot(kind="barh", color="#2c7fb8")
    plt.xlabel("Regime-weighted gain")
    plt.title(f"Top {top_n} Feature Importances (two-stage ensemble)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved -> {path}")


def plot_shap(ensemble, X_sample, top_n=10, path=SHAP_PLOT_PATH):
    """SHAP beeswarm for the short-regime expert (the bulk of deals)."""
    print("\nComputing SHAP values (short-regime expert, majority of deals) ...")
    try:
        explainer = shap.TreeExplainer(ensemble.expert_short["xgb_log"])
        sv = explainer.shap_values(X_sample)
        plt.figure()
        shap.summary_plot(sv, X_sample, max_display=top_n, show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved -> {path}")
    except Exception as exc:
        print(f"  [warning] SHAP skipped: {exc}")


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------
def predict_days(new_deal_dict: dict) -> float:
    """Predict Business Days To Complete for a single raw deal dict (original Excel columns)."""
    with open(PREPROCESSOR_PATH, "rb") as fh:
        preproc = pickle.load(fh)
    with open(ENSEMBLE_PATH, "rb") as fh:
        ensemble = pickle.load(fh)

    X_new = build_features(pd.DataFrame([new_deal_dict]), categories=preproc["categories"])
    X_new = X_new[preproc["feature_columns"]]
    for c in preproc["categorical_features"]:
        X_new[c] = X_new[c].astype(preproc["categories"][c])
    return float(ensemble.predict(X_new)[0])


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    raw, X, y, announce_year = load_xy()
    print(f"Loaded {len(X)} deals, {X.shape[1]} leakage-free features.")
    print(f"Target skew={y.skew():.2f}  median={y.median():.0f}d  max={y.max():.0f}d\n")

    # 1. Optimize the regime threshold against repeated CV (the chosen tuning metric).
    threshold = select_threshold_by_cv(X, y)

    # 2. Report BOTH metrics for the tuned two-stage model and the single-blend baseline.
    print("\n" + "=" * 72)
    print("REPEATED 3x5-FOLD CV R2 (tuning metric)")
    two_mean, two_std = repeated_cv_r2(lambda: TwoStageEnsemble(threshold=threshold), X, y)
    base_mean, base_std = repeated_cv_r2(SingleBlend, X, y)
    print(f"  Two-stage blended ensemble : {two_mean:.3f} +/- {two_std:.3f}")
    print(f"  Single-blend baseline      : {base_mean:.3f} +/- {base_std:.3f}")

    print("\n" + "=" * 72)
    print("HONEST OUT-OF-TIME (train <=2023, test >=2024)")
    print("Two-stage blended ensemble:")
    ensemble, _ = report_out_of_time(
        TwoStageEnsemble(threshold=threshold), X, y, announce_year
    )
    print("Single-blend baseline:")
    report_out_of_time(SingleBlend(), X, y, announce_year)

    # 3. Refit the tuned ensemble on ALL data for the shipped artifact + diagnostics.
    print("\n" + "=" * 72)
    print("Fitting final two-stage ensemble on the full dataset for deployment ...")
    ensemble = TwoStageEnsemble(threshold=threshold).fit(X, y)

    # 4. Diagnostics.
    plot_importance(ensemble)
    te = announce_year >= 2024
    shap_sample = X[te]
    if len(shap_sample) > 400:
        shap_sample = shap_sample.sample(400, random_state=RANDOM_STATE)
    plot_shap(ensemble, shap_sample)

    # 5. Persist artifacts.
    with open(ENSEMBLE_PATH, "wb") as fh:
        pickle.dump(ensemble, fh)
    preprocessor = {
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "categories": {c: X[c].dtype for c in CATEGORICAL_FEATURES},
        "threshold": threshold,
        "weights": EXPERT_WEIGHTS,
    }
    with open(PREPROCESSOR_PATH, "wb") as fh:
        pickle.dump(preprocessor, fh)
    print(f"\nSaved ensemble -> {ENSEMBLE_PATH}")
    print(f"Saved preprocessor -> {PREPROCESSOR_PATH}")

    # 6. Smoke-test inference.
    sample = raw[pd.to_datetime(raw["Announce Date"]).dt.year >= 2024].iloc[0].to_dict()
    print("\nSmoke-test predict_days():")
    print(f"  predicted={predict_days(sample):.1f}d  actual={sample[TARGET]}d")

    print("\nDone. Two-stage blended ensemble trained, evaluated, and saved.")


if __name__ == "__main__":
    main()
