"""
M&A `Business Days To Complete` model -- 2025-onwards truncated dataset.
=======================================================================

The full-history model (xgboost_ma_completion_model.py / ma_completion_twostage_ensemble.py)
is tuned for ~1,170 deals spanning 2017-2026. This script trains a SEPARATE model on only
deals announced in 2025 or later (n=155), because that regime looks materially different:

  * Small sample (155 rows) -> gradient boosting overfits. A repeated-CV bake-off across
    11 model families showed bagged trees win clearly:
        RandomForest+ExtraTrees blend  CV R2 = 0.247   <-- chosen
        RandomForest (tuned)           CV R2 = 0.225
        ElasticNet / Lasso             CV R2 = 0.17
        XGBoost (regularized)          CV R2 = 0.168
        HistGradientBoosting           CV R2 = 0.014
    Bagging averages out variance where boosting chases noise; two decorrelated baggers
    (RF + ExtraTrees) blended give the best, most stable R2.

  * The target is far better behaved post-2025 (skew 1.24 vs 3.35, max 236 vs 1078 business
    days) -- the unpredictable multi-year regulatory tail is absent, so we model the RAW
    target (no log transform needed; trees don't extrapolate, avoiding the expm1 blow-ups
    that made log-target linear models unstable here).

  * All three regulatory flags (SAMR/EC/CFIUS) are 0 for every 2025+ deal, and Nature of Bid
    is near-constant. These zero-variance columns are dropped automatically.

EVALUATION: repeated K-Fold CV R2 is the primary metric -- with n=155 a single chronological
holdout is not stable. (The 2026 slice is additionally a CENSORED sample: only deals that
have already closed by mid-2026 carry a target, biasing it toward fast deals, so its R2 is
uninformative and reported only as a caveat.)

Target is in BUSINESS days. Multiply by ~1.4 for calendar days, or /21.7 for months.

Run:
    python ma_2025plus_model.py
"""

import pickle
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Reuse the vetted, leakage-free feature schema from the full-history model.
from xgboost_ma_completion_model import (
    CATEGORICAL_FEATURES,
    SHEET_NAME,
    TARGET,
    build_features,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
FULL_FILE_PATH = "LARGE_DATASET (TOGGLES).xlsx"
TRUNCATED_FILE_PATH = "LARGE_DATASET_2025plus.xlsx"  # convenience export (created if absent)
CUTOFF_YEAR = 2025
RANDOM_STATE = 0

MODEL_PATH = "ma_2025plus_model.pkl"
PREPROCESSOR_PATH = "ma_2025plus_preprocessor.pkl"
IMPORTANCE_PLOT_PATH = "ma_2025plus_importance.png"

# RandomForest/ExtraTrees hyper-parameters selected via repeated-CV grid search.
RF_PARAMS = dict(n_estimators=500, max_depth=None, min_samples_leaf=5,
                 max_features=0.7, random_state=RANDOM_STATE, n_jobs=-1)
ET_PARAMS = dict(n_estimators=600, max_depth=None, min_samples_leaf=5,
                 max_features=0.7, random_state=RANDOM_STATE, n_jobs=-1)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_truncated():
    """Load full dataset, keep only deals announced in CUTOFF_YEAR or later."""
    raw = pd.read_excel(FULL_FILE_PATH, sheet_name=SHEET_NAME)
    year = pd.to_datetime(raw["Announce Date"], errors="coerce").dt.year
    sub = raw[year >= CUTOFF_YEAR].copy().reset_index(drop=True)
    print(f"Full dataset: {len(raw)} rows | truncated (>= {CUTOFF_YEAR}): {len(sub)} rows")
    return sub


def build_xy(sub):
    """Feature matrix + target, dropping zero-variance columns for this subset."""
    X = build_features(sub)
    y = sub[TARGET].astype(float).values

    # Leakage guard.
    assert TARGET not in X.columns, "Target leaked into X!"

    # Drop columns that are constant within the 2025+ subset (e.g. SAMR/EC/CFIUS all 0):
    # they carry zero information and only add noise on a small sample.
    const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=const_cols)
    print(f"Dropped {len(const_cols)} zero-variance columns: {const_cols}")
    print(f"Feature matrix: {X.shape}")
    return X, y, const_cols


def make_preprocessor(X):
    """Median-impute numerics; most-frequent-impute + ordinal-encode categoricals.

    (Ordinal encoding, not one-hot: with n=155 one-hot on 18-level industry / 12-level
    acquirer-country explodes dimensionality; tree models split ordinal codes fine.)
    """
    num = [c for c in X.columns if str(X[c].dtype).startswith(("float", "int", "Int"))]
    cat = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), cat),
        ]
    )
    return pre, num, cat


def make_model(pre):
    """RF + ExtraTrees blend (VotingRegressor) on the shared preprocessor."""
    rf = Pipeline([("pre", pre), ("m", RandomForestRegressor(**RF_PARAMS))])
    et = Pipeline([("pre", pre), ("m", ExtraTreesRegressor(**ET_PARAMS))])
    return VotingRegressor([("rf", rf), ("et", et)])


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def evaluate_cv(model, X, y):
    """Repeated 5-fold CV -- the primary metric for this small sample."""
    cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=42)
    r2 = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
    mae = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    print("\nRepeated 5-fold CV (20 repeats, n=%d):" % len(y))
    print(f"  R2  = {r2.mean():+.3f} +/- {r2.std():.3f}   (SE {r2.std()/np.sqrt(len(r2)):.3f})")
    print(f"  MAE = {mae.mean():.1f} +/- {mae.std():.1f} business days")
    return r2.mean(), mae.mean()


def chrono_caveat(model, X, y, sub):
    """2026 holdout -- reported ONLY as a caveat (tiny, censored sample)."""
    tr = (pd.to_datetime(sub["Announce Date"]).dt.year == 2025).values
    if tr.sum() and (~tr).sum():
        model.fit(X[tr], y[tr])
        p = model.predict(X[~tr])
        print(f"\n[caveat] Chronological 2025->2026 holdout (train={tr.sum()}, test={(~tr).sum()}):")
        print(f"  R2={r2_score(y[~tr], p):+.3f}  MAE={mean_absolute_error(y[~tr], p):.1f}  "
              f"MAPE={mean_absolute_percentage_error(y[~tr], p)*100:.0f}%")
        print("  NOTE: 2026 rows are censored (only already-closed deals have a target), so")
        print("        target variance is compressed and this R2 is not informative.")


# --------------------------------------------------------------------------------------
# Interpretability
# --------------------------------------------------------------------------------------
def plot_permutation_importance(model, X, y, path=IMPORTANCE_PLOT_PATH, top_n=15):
    """Permutation importance on the full fit (directional; robust to encoding)."""
    model.fit(X, y)
    r = permutation_importance(model, X, y, n_repeats=30, random_state=RANDOM_STATE,
                               scoring="r2", n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
    print("\nTop features (permutation importance, drop in R2 when shuffled):")
    for name, val in imp.head(top_n).items():
        print(f"  {name:<40} {val:+.4f}")
    top = imp.head(top_n)
    plt.figure(figsize=(9, 7))
    top.iloc[::-1].plot(kind="barh", color="#2c7fb8")
    plt.xlabel("Permutation importance (mean R2 drop)")
    plt.title(f"2025+ model -- top {top_n} features (RF+ExtraTrees blend)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved -> {path}")
    return model


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------
def predict_days(new_deal_dict: dict) -> float:
    """Predict Business Days To Complete for a raw deal dict (original Excel columns)."""
    with open(PREPROCESSOR_PATH, "rb") as fh:
        meta = pickle.load(fh)
    with open(MODEL_PATH, "rb") as fh:
        model = pickle.load(fh)
    X_new = build_features(pd.DataFrame([new_deal_dict]))
    X_new = X_new[meta["feature_columns"]]  # align to trained schema (drops const cols)
    return float(model.predict(X_new)[0])


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    sub = load_truncated()
    y_all = sub[TARGET].astype(float)
    print(f"Target (business days): median={y_all.median():.0f} mean={y_all.mean():.0f} "
          f"max={y_all.max():.0f} skew={y_all.skew():.2f}")

    X, y, const_cols = build_xy(sub)
    pre, num, cat = make_preprocessor(X)
    model = make_model(pre)

    # 1. Primary metric.
    cv_r2, cv_mae = evaluate_cv(model, X, y)
    # 2. Caveated chronological check.
    chrono_caveat(model, X, y, sub)
    # 3. Interpretability + final fit on all 2025+ data.
    model = plot_permutation_importance(model, X, y)

    # 4. Persist.
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(model, fh)
    meta = {
        "feature_columns": list(X.columns),
        "numeric_features": num,
        "categorical_features": cat,
        "dropped_constant_columns": const_cols,
        "cutoff_year": CUTOFF_YEAR,
        "cv_r2": cv_r2,
        "cv_mae": cv_mae,
        "model": "VotingRegressor(RandomForest + ExtraTrees)",
    }
    with open(PREPROCESSOR_PATH, "wb") as fh:
        pickle.dump(meta, fh)
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved metadata -> {PREPROCESSOR_PATH}")

    # 5. Smoke-test inference.
    sample = sub.iloc[0].to_dict()
    print("\nSmoke-test predict_days():")
    print(f"  predicted={predict_days(sample):.1f}  actual={sample[TARGET]} business days")

    print("\nDone. 2025+ bagged-ensemble model trained, evaluated, and saved.")


if __name__ == "__main__":
    main()
