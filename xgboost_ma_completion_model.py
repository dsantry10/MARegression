"""
XGBoost model for predicting M&A deal "Days To Complete"
========================================================

Senior-quant-grade pipeline that trains a gradient-boosted tree model to
predict how many days an M&A deal takes to close, using deal characteristics
as features.

Data source
-----------
``New_Training_Sheet_2.xlsx`` (sheet "Sheet1"), 1,170 completed deals.

SPLIT STRATEGY  (chronological when possible, seeded-random fallback)
---------------------------------------------------------------------
The brief mandates a **chronological** split on ``Announce Date``: train on
deals announced before 2024-01-01, test on deals announced on/after that date.
This script does exactly that **whenever an ``Announce Date`` column is
present**.

The file we were pointed at (``New_Training_Sheet_2.xlsx``), however, contains
**no date column of any kind** (verified: 18 columns, none temporal). A
chronological split is therefore impossible against it, so the script falls
back to a **seeded random split** (``RANDOM_STATE``) and prints a loud notice
when it does. The moment the data is supplied with an ``Announce Date`` column,
the chronological path activates automatically -- no code change needed.

The feature schema is likewise resolved **dynamically against whatever columns
are present**: every feature from the brief that exists is treated exactly as
specified, and any that is absent is skipped with a printed note.

Everything else follows the brief:
  * XGBoost native NaN handling (we do NOT impute numeric NaNs).
  * No scaling/standardisation (tree model).
  * Native categorical handling via pandas ``category`` dtype.
  * Target leakage guarded by an explicit assertion.
  * MAE-loss primary model, optional P75 quantile model.
  * Feature importance (gain), SHAP summary, MAE / MAPE / R^2 reporting.
  * Saved model + preprocessor artifacts and a reusable ``predict_days``.

Dependencies: pandas, numpy, xgboost, scikit-learn, shap, matplotlib, openpyxl
"""

from __future__ import annotations

import pickle
import warnings

import matplotlib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    TimeSeriesSplit,
    train_test_split,
)

matplotlib.use("Agg")  # headless: save plots to disk instead of a GUI window
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# 0. CONFIGURATION  (the only thing you may need to edit)
# ---------------------------------------------------------------------------
INPUT_FILE = "New_Training_Sheet_2.xlsx"
SHEET_NAME = "Sheet1"
TARGET = "Days To Complete"
RANDOM_STATE = 42
SPLIT_DATE = "2024-01-01"  # chronological cutoff: train < this, test >= this
TEST_SIZE = 0.20          # random-fallback fraction held out as the test set
VALID_SIZE = 0.20         # fraction of the *training* set used for early stopping
N_SEARCH_ITER = 25        # RandomizedSearchCV candidates
EARLY_STOPPING_ROUNDS = 20

MODEL_PATH = "xgboost_ma_model.json"
P75_MODEL_PATH = "xgboost_p75_model.json"
PREPROCESSOR_PATH = "preprocessor.pkl"
IMPORTANCE_PLOT = "feature_importance_top15.png"
SHAP_PLOT = "shap_summary_top10.png"

# ---------------------------------------------------------------------------
# FEATURE SCHEMA  (candidate lists from the brief; resolved against the file)
# ---------------------------------------------------------------------------
# A. Numeric features -- kept as float, NaNs are LEFT IN PLACE (no imputation).
#    Raw "Target Sales/Revenue/Turnover" is intentionally NOT here: per the
#    brief we model on its pre-computed log ("Log Revenue") and never use the
#    raw and log columns together (raw revenue is in DROP_CANDIDATES below).
NUMERIC_CANDIDATES = [
    "Announced Premium",
    "Target Trailg 12 Mth Operating Margin",
    "Log Revenue",
    "Log Equity Value",
]

# B. Binary Yes/No flags -- coerced to {0, 1} integers.
BINARY_CANDIDATES = [
    "Additional Stake Purchase",
    "Competing Bid",
    "Cross Border",
    "Going Private",
    "PE Buyout",
    "Tender Offer",
]

# C+D. Categorical / country features -- pandas 'category' dtype, handled
#      natively by XGBoost (NO one-hot encoding).
CATEGORICAL_CANDIDATES = [
    "Payment Type",
    "Nature of Bid",
    "Target Industry Group",
    "Target Country/Region",
    "Acquirer Country/Region",
]

# D. Raw date column to derive Announce_Year / Announce_Month from (if present).
DATE_COL = "Announce Date"

# E. New binary flags parsed out of the free-text "Deal Attributes" column.
#    (We deliberately do NOT re-extract flags that already exist as columns,
#     e.g. Cross Border / Going Private.)
DEAL_ATTR_COL = "Deal Attributes"
TEXT_FLAG_PATTERNS = {
    "is_Reverse_Merger": "Reverse Merger",
    "is_Management_Buyout": "Management Buyout",
    "is_Squeeze_Out": "Squeeze out",
    "is_Secondary_Transaction": "Secondary Transaction",
    "is_Bankruptcy_Liquidation": "Bankruptcy/Liquidation",
}

# G. Columns to drop completely (identifiers + raw cols superseded by logs).
DROP_CANDIDATES = [
    "Target Ticker",                      # identifier (drop if present)
    "Acquirer Ticker",                    # identifier (drop if present)
    "Target Sales/Revenue/Turnover",      # use Log Revenue instead
    "Announced Total Value (mil.)",       # raw value column, if present
    "Announced Equity Value (mil.)",      # use Log Equity Value instead
]


# ===========================================================================
# PREPROCESSING
# ===========================================================================
def _coerce_binary(series: pd.Series) -> pd.Series:
    """Map Yes/No (or 1/0/true/false) to {0, 1} ints; anything else -> NaN->0."""
    s = series.astype(str).str.strip().str.lower()
    mapped = s.map(
        {"yes": 1, "y": 1, "true": 1, "1": 1,
         "no": 0, "n": 0, "false": 0, "0": 0}
    )
    return mapped.fillna(0).astype(int)


def build_schema(df: pd.DataFrame) -> dict:
    """Resolve the candidate feature lists against the columns actually present.

    Returns a 'preprocessor' dict that fully describes how to turn a raw frame
    into the model's feature matrix -- this is what we pickle so that
    ``predict_days`` can reproduce the transformation exactly.
    """
    present = set(df.columns)

    numeric = [c for c in NUMERIC_CANDIDATES if c in present]
    binary = [c for c in BINARY_CANDIDATES if c in present]
    categorical = [c for c in CATEGORICAL_CANDIDATES if c in present]
    has_date = DATE_COL in present
    has_attrs = DEAL_ATTR_COL in present

    # Report what we found vs. what the brief expected.
    def _report(name, found, candidates):
        missing = [c for c in candidates if c not in found]
        print(f"  {name:12s}: using {len(found)}/{len(candidates)} -> {found}")
        if missing:
            print(f"               (absent in this file, skipped: {missing})")

    print("\nResolving feature schema against the file:")
    _report("numeric", numeric, NUMERIC_CANDIDATES)
    _report("binary", binary, BINARY_CANDIDATES)
    _report("categorical", categorical, CATEGORICAL_CANDIDATES)
    print(f"  date col    : {'present' if has_date else 'ABSENT (no temporal features)'}")
    print(f"  deal attrs  : {'present' if has_attrs else 'ABSENT (no text flags)'}")

    schema = {
        "numeric": numeric,
        "binary": binary,
        "categorical": categorical,
        "has_date": has_date,
        "has_attrs": has_attrs,
        "text_flags": TEXT_FLAG_PATTERNS,
        # Filled in after the first transform so inference matches training:
        "feature_order": None,
        "categories": {},  # {categorical_col: [ordered category labels]}
    }
    return schema


def transform(df: pd.DataFrame, schema: dict, *, fit: bool) -> pd.DataFrame:
    """Turn a raw deal frame into the model-ready feature matrix.

    ``fit=True`` learns the category vocabularies and final column order from
    the data; ``fit=False`` reuses the vocabularies/order stored in ``schema``
    so a single new deal lines up exactly with the training matrix.
    """
    X = pd.DataFrame(index=df.index)

    # --- A. Numeric: coerce to float, KEEP NaNs (XGBoost handles them) -------
    for col in schema["numeric"]:
        X[col] = pd.to_numeric(df.get(col), errors="coerce").astype(float)

    # --- B. Binary Yes/No flags ---------------------------------------------
    for col in schema["binary"]:
        X[col] = _coerce_binary(df[col]) if col in df else 0

    # --- D. Temporal features from the raw date (if a date column exists) ----
    if schema["has_date"] and DATE_COL in df:
        dt = pd.to_datetime(df[DATE_COL], errors="coerce")
        X["Announce_Year"] = dt.dt.year.astype("Int64").astype(float)
        X["Announce_Month"] = dt.dt.month.astype("Int64").astype(float)

    # --- E. Text-parsed binary flags from "Deal Attributes" ------------------
    if schema["has_attrs"] and DEAL_ATTR_COL in df:
        attr = df[DEAL_ATTR_COL].astype(str)
        for flag, pattern in schema["text_flags"].items():
            X[flag] = attr.str.contains(pattern, case=False, regex=False).astype(int)

    # --- C. Categorical: pandas 'category' dtype (native XGBoost handling) ---
    for col in schema["categorical"]:
        raw = df[col].astype("object") if col in df else pd.Series(index=df.index, dtype="object")
        # Treat blanks/None as a genuine NaN category, not the literal "nan".
        raw = raw.where(~raw.isna(), other=np.nan)
        if fit:
            cat = pd.Categorical(raw)
            schema["categories"][col] = list(cat.categories)
            X[col] = cat
        else:
            X[col] = pd.Categorical(raw, categories=schema["categories"][col])

    # --- Lock in / enforce the exact training column order -------------------
    if fit:
        schema["feature_order"] = list(X.columns)
    else:
        X = X.reindex(columns=schema["feature_order"])
        # Re-assert category dtypes after reindex (reindex can drop dtype info).
        for col in schema["categorical"]:
            X[col] = pd.Categorical(X[col], categories=schema["categories"][col])

    return X


# ===========================================================================
# METRICS
# ===========================================================================
def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE in %, ignoring rows where the actual is 0 (avoids div-by-zero)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def report_metrics(label: str, y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{label}")
    print(f"  MAE : {mae:8.2f} days")
    print(f"  MAPE: {mape:8.2f} %")
    print(f"  R^2 : {r2:8.4f}")
    return {"MAE": mae, "MAPE": mape, "R2": r2}


# ===========================================================================
# MODEL ARTIFACTS (loaded lazily so predict_days works standalone)
# ===========================================================================
_LOADED_MODEL: xgb.XGBRegressor | None = None
_LOADED_SCHEMA: dict | None = None


def _ensure_loaded():
    """Load the saved model + preprocessor from disk on first prediction."""
    global _LOADED_MODEL, _LOADED_SCHEMA
    if _LOADED_MODEL is None:
        _LOADED_MODEL = xgb.XGBRegressor(enable_categorical=True)
        _LOADED_MODEL.load_model(MODEL_PATH)
    if _LOADED_SCHEMA is None:
        with open(PREPROCESSOR_PATH, "rb") as fh:
            _LOADED_SCHEMA = pickle.load(fh)


def predict_days(new_deal_dict: dict) -> float:
    """Predict Days-To-Complete for one raw deal.

    Parameters
    ----------
    new_deal_dict : dict
        Raw feature values keyed by the ORIGINAL column names (same schema as
        the source spreadsheet). Missing keys are tolerated -- they become NaN
        and are handled natively by XGBoost.

    Returns
    -------
    float
        Predicted number of days from announcement to completion.
    """
    _ensure_loaded()
    raw = pd.DataFrame([new_deal_dict])
    X = transform(raw, _LOADED_SCHEMA, fit=False)
    return float(_LOADED_MODEL.predict(X)[0])


# ===========================================================================
# TRAIN / VALIDATION / TEST SPLIT
# ===========================================================================
def make_splits(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, schema: dict):
    """Build train / validation / test splits and the matching CV strategy.

    * If an ``Announce Date`` column exists -> **chronological** split exactly
      as the brief mandates (train < ``SPLIT_DATE``, test >= ``SPLIT_DATE``),
      the validation slice is the *latest* 20% of the training period, and CV
      is a 3-fold ``TimeSeriesSplit`` (no look-ahead).
    * Otherwise -> **seeded random** split (loud notice), with a random
      validation slice and a shuffled 3-fold ``KFold``.

    Returns
    -------
    (X_tr, X_val, y_tr, y_val, X_test, y_test, cv, mode)
    """
    if schema["has_date"] and DATE_COL in df:
        dates = pd.to_datetime(df[DATE_COL], errors="coerce")
        cutoff = pd.Timestamp(SPLIT_DATE)
        train_mask = (dates < cutoff).to_numpy()
        test_mask = (dates >= cutoff).to_numpy()

        # Order the training rows by date so the validation slice / TimeSeries
        # folds respect chronology.
        train_order = (dates[train_mask]
                       .sort_values()
                       .index)
        X_train = X.loc[train_order]
        y_train = y.loc[train_order]
        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]

        n_val = max(1, int(round(len(X_train) * VALID_SIZE)))
        X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
        y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]

        cv = TimeSeriesSplit(n_splits=3)
        mode = f"chronological (Announce Date, cutoff {SPLIT_DATE})"
        print(f"\nSplit — {mode}:")
    else:
        print("\n" + "!" * 78)
        print(f"!! '{DATE_COL}' column NOT found -> chronological split is "
              f"impossible.")
        print(f"!! Falling back to a SEEDED RANDOM split (state={RANDOM_STATE}). "
              f"Supply an")
        print(f"!! '{DATE_COL}' column to activate the chronological split "
              f"automatically.")
        print("!" * 78)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=VALID_SIZE, random_state=RANDOM_STATE
        )
        cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        mode = f"seeded random (state={RANDOM_STATE})"
        print(f"\nSplit — {mode}:")

    print(f"  train     : {len(X_tr):4d} rows")
    print(f"  validation: {len(X_val):4d} rows  (latest period / for early stopping)")
    print(f"  test      : {len(X_test):4d} rows")
    if len(X_test) == 0:
        raise ValueError(
            "Test set is empty — no deals on/after the cutoff. Check SPLIT_DATE "
            "against the date range in the data."
        )
    return X_tr, X_val, y_tr, y_val, X_test, y_test, cv, mode


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================
def main() -> None:
    print("=" * 78)
    print("XGBoost  —  M&A Days-To-Complete model")
    print("=" * 78)

    # --- 1. Load -------------------------------------------------------------
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    print(f"\nLoaded {len(df)} deals x {df.shape[1]} columns from "
          f"{INPUT_FILE!r} [{SHEET_NAME}]")

    # Drop identifiers / superseded raw columns that happen to be present.
    to_drop = [c for c in DROP_CANDIDATES if c in df.columns]
    if to_drop:
        print(f"Dropping identifier/redundant columns: {to_drop}")
        df = df.drop(columns=to_drop)

    # --- 2. Target + feature matrix -----------------------------------------
    df = df[df[TARGET].notna()].reset_index(drop=True)  # need a known target to train
    y = df[TARGET].astype(float)
    print(f"\nTarget '{TARGET}': n={len(y)}  min={y.min():.0f}  "
          f"median={y.median():.0f}  mean={y.mean():.0f}  max={y.max():.0f}")

    schema = build_schema(df)
    X = transform(df, schema, fit=True)

    # --- 3. Leakage guard ----------------------------------------------------
    assert TARGET not in X.columns, "Target leaked into features!"
    print(f"\nLeakage check passed: '{TARGET}' is NOT in the {X.shape[1]} "
          f"feature columns.")
    print(f"Feature columns: {list(X.columns)}")

    # --- 4. Split (chronological if a date column exists, else random) -------
    X_tr, X_val, y_tr, y_val, X_test, y_test, cv, split_mode = make_splits(
        df, X, y, schema
    )

    # --- 5. Hyperparameter search (MAE objective) ----------------------------
    base = xgb.XGBRegressor(
        objective="reg:absoluteerror",      # MAE loss -> robust to outliers
        tree_method="hist",                  # fast histogram algorithm
        enable_categorical=True,             # native pandas 'category' handling
        eval_metric="mae",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    param_dist = {
        "n_estimators": [100, 300, 500],
        "max_depth": [4, 6, 8],              # shallow -> guards against overfit
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
    }

    # cv comes from make_splits: TimeSeriesSplit (chronological) or KFold.
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=N_SEARCH_ITER,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print(f"\nRunning RandomizedSearchCV (3-fold {type(cv).__name__}) with "
          f"early stopping ...")
    search.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    best = search.best_estimator_
    print(f"\nBest CV MAE: {-search.best_score_:.2f} days")
    print(f"Best params: {search.best_params_}")
    if getattr(best, "best_iteration", None) is not None:
        print(f"Early-stopped best_iteration: {best.best_iteration}")

    # --- 6. Evaluate on the held-out test set --------------------------------
    print("\n" + "=" * 78)
    print("TEST-SET PERFORMANCE")
    print("=" * 78)
    pred_test = best.predict(X_test)
    primary_metrics = report_metrics("Primary model (MAE loss):", y_test, pred_test)

    # --- 7. Feature importance (gain) ---------------------------------------
    print("\n" + "=" * 78)
    print("FEATURE IMPORTANCE  (top 15 by gain)")
    print("=" * 78)
    booster = best.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp = (pd.Series(gain, name="gain")
           .sort_values(ascending=False)
           .head(15))
    for feat, val in imp.items():
        print(f"  {feat:42s} {val:12.2f}")

    fig, ax = plt.subplots(figsize=(9, 7))
    imp.iloc[::-1].plot.barh(ax=ax, color="steelblue", edgecolor="k")
    ax.set_xlabel("Importance (gain)")
    ax.set_title("XGBoost — Top 15 Features by Gain")
    fig.tight_layout()
    fig.savefig(IMPORTANCE_PLOT, dpi=120)
    plt.close(fig)
    print(f"\nSaved importance plot -> {IMPORTANCE_PLOT}")

    # --- 8. SHAP summary plot (top 10 features) ------------------------------
    print("\nComputing SHAP values for the test set ...")
    try:
        explainer = shap.TreeExplainer(best)
        shap_values = explainer.shap_values(X_test)
        plt.figure()
        shap.summary_plot(
            shap_values, X_test, max_display=10, show=False, plot_size=(9, 7)
        )
        plt.tight_layout()
        plt.savefig(SHAP_PLOT, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP summary plot -> {SHAP_PLOT}")
    except Exception as exc:  # SHAP + native categoricals can be finicky
        print(f"  [warn] SHAP plot skipped: {exc}")

    # --- 9. Optional P75 (worst-case) quantile model -------------------------
    print("\n" + "=" * 78)
    print("P75 QUANTILE MODEL  (worst-case timeline, quantile_alpha=0.75)")
    print("=" * 78)
    p75_params = {k: v for k, v in search.best_params_.items()}
    p75 = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.75,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **p75_params,
    )
    p75.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    pred_p75 = p75.predict(X_test)
    p75_metrics = report_metrics("P75 quantile model:", y_test, pred_p75)
    print(f"\nMAE comparison  ->  primary: {primary_metrics['MAE']:.2f} days   "
          f"P75: {p75_metrics['MAE']:.2f} days")
    print("(The P75 model intentionally over-predicts to give a conservative, "
          "worst-case\n timeline; a higher MAE vs. the median model is expected.)")

    # --- 10. Persist artifacts ----------------------------------------------
    print("\n" + "=" * 78)
    print("SAVING ARTIFACTS")
    print("=" * 78)
    best.save_model(MODEL_PATH)
    p75.save_model(P75_MODEL_PATH)
    with open(PREPROCESSOR_PATH, "wb") as fh:
        pickle.dump(schema, fh)
    print(f"  model        -> {MODEL_PATH}")
    print(f"  P75 model    -> {P75_MODEL_PATH}")
    print(f"  preprocessor -> {PREPROCESSOR_PATH}")

    # --- 11. Demonstrate the reusable prediction function -------------------
    print("\n" + "=" * 78)
    print("DEMO: predict_days() on one raw test deal")
    print("=" * 78)
    sample_idx = X_test.index[0]
    sample_raw = df.loc[sample_idx].to_dict()  # raw row, original schema
    sample_raw.pop(TARGET, None)               # caller would not know the answer
    pred = predict_days(sample_raw)
    print(f"  predicted: {pred:.1f} days")
    print(f"  actual   : {y_test.loc[sample_idx]:.0f} days")

    print("\nDone.")


if __name__ == "__main__":
    main()
