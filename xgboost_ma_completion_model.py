"""
XGBoost model to predict `Business Days To Complete` for M&A deals.
==================================================================

Note: the target is measured in BUSINESS days (weekdays), not calendar days. To convert
a prediction to calendar time, multiply by ~1.4 (7/5) or divide by ~21.7 business
days/month.

Production-grade, reproducible pipeline that:
  * Loads `LARGE_DATASET (TOGGLES).xlsx`.
  * Applies a *strict* feature schema (numeric / binary / categorical / temporal /
    text-parsed) and aggressively drops every ticker / reference / helper column.
  * Incorporates the three new regulatory flags (SAMR, EC, CFIUS).
  * Performs a chronological train/test split (train <= 2023, test >= 2024) with a
    late-training validation slice used for early stopping.
  * Tunes an XGBoost regressor with RandomizedSearchCV + TimeSeriesSplit.
  * Reports MAE / MAPE / R2, feature importance (gain) and a SHAP summary plot.
  * Saves model artifacts and exposes a reusable `predict_days(...)` function.

Design notes
------------
* XGBoost handles missing numeric values natively -> we deliberately DO NOT impute.
* XGBoost is tree-based -> we deliberately DO NOT scale/standardize.
* Categoricals use XGBoost's native categorical support (`enable_categorical=True`).
* Early stopping cannot be threaded cleanly through per-fold RandomizedSearchCV, so we
  tune hyper-parameters with cross-validation first, then refit the winning
  configuration once with `early_stopping_rounds` against the explicit validation set.

Run:
    python xgboost_ma_completion_model.py
"""

# --------------------------------------------------------------------------------------
# Imports (all at top)
# --------------------------------------------------------------------------------------
import pickle
import warnings

import matplotlib

matplotlib.use("Agg")  # headless backend so the script runs end-to-end without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
FILE_PATH = "LARGE_DATASET (TOGGLES).xlsx"
SHEET_NAME = "Sheet1"
TARGET = "Business Days To Complete"
RANDOM_STATE = 42

MODEL_PATH = "xgboost_ma_model.json"
P75_MODEL_PATH = "xgboost_p75_model.json"
PREPROCESSOR_PATH = "preprocessor.pkl"
IMPORTANCE_PLOT_PATH = "feature_importance_gain.png"
SHAP_PLOT_PATH = "shap_summary.png"

# ---- Explicit feature schema (Section 2 of the specification) -------------------------
NUMERIC_FEATURES = [
    "Log TV",
    "Log Revenue",
    "Log Equity Value",
    "Announced Premium",
    "TV/EBITDA",
    "Acquirer Termination Fee",
    "Target Termination Fee",
    "Target Trailg 12 Mth Operating Margin",
]

# Binary flags currently stored as "Yes"/"No" text.
YESNO_BINARY_FEATURES = [
    "Additional Stake Purchase",
    "Competing Bid",
    "Cross Border",
    "Going Private",
    "PE Buyout",
    "Tender Offer",
]

# New regulator flags -- already stored as 0/1 integers, but we coerce defensively in
# case a future export delivers them as "Yes"/"No" text.
REGULATOR_BINARY_FEATURES = ["SAMR", "EC", "CFIUS"]

BINARY_FEATURES = YESNO_BINARY_FEATURES + REGULATOR_BINARY_FEATURES

CATEGORICAL_FEATURES = [
    "Payment Type",
    "Nature of Bid",
    "Target Industry Group",
    "Target Country/Region",
    "Acquirer Country/Region",
]

TEMPORAL_FEATURES = ["Announce_Year", "Announce_Month"]

# Text-parsed flags derived from `Deal Attributes` -> (new column name, search string).
DEAL_ATTRIBUTE_FLAGS = {
    "is_Reverse_Merger": "Reverse Merger",
    "is_Management_Buyout": "Management Buyout",
    "is_Squeeze_Out": "Squeeze out",
    "is_Secondary_Transaction": "Secondary Transaction",
    "is_Bankruptcy_Liquidation": "Bankruptcy/Liquidation",
}
TEXT_FLAG_FEATURES = list(DEAL_ATTRIBUTE_FLAGS.keys())

# Columns that must NEVER become features (Section 2.F). Ticker/REF/Reference columns are
# additionally caught by a name-based rule below.
EXPLICIT_DROP_COLUMNS = [
    "Target Ticker",
    "Acquirer Ticker",
    "Announce Date",
    "Deal Attributes",
    "Target Industry Sector",
    "Target Industry Subgroup",
    "Announced Total Value (mil.)",
    "Target Sales/Revenue/Turnover",
    "Announced Equity Value (mil.)",
]

# The full, ordered feature list the model expects.
FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + BINARY_FEATURES
    + CATEGORICAL_FEATURES
    + TEMPORAL_FEATURES
    + TEXT_FLAG_FEATURES
)


# --------------------------------------------------------------------------------------
# Helper predicates
# --------------------------------------------------------------------------------------
def _is_banned_name(col: str) -> bool:
    """True if a column name looks like a ticker / reference / helper column."""
    lowered = str(col).lower()
    return any(tok in lowered for tok in ("ticker", "ref", "reference")) or str(
        col
    ).startswith("Unnamed")


def _coerce_binary(series: pd.Series) -> pd.Series:
    """Map Yes/No/True/1 style values to a nullable Int8 0/1 series."""
    mapping = {
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
    }
    if series.dtype.kind in "biufc":  # already numeric -> just normalise to 0/1
        return (series.fillna(0) != 0).astype("int8")
    normalised = series.astype("string").str.strip().str.lower().map(mapping)
    return normalised.fillna(0).astype("int8")


# --------------------------------------------------------------------------------------
# Core preprocessing (shared by training and inference)
# --------------------------------------------------------------------------------------
def build_features(df_raw: pd.DataFrame, categories: dict | None = None) -> pd.DataFrame:
    """Transform a raw dataframe (original Excel columns) into the model feature matrix.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw data using the original Excel column names.
    categories : dict | None
        Optional mapping {column: pd.CategoricalDtype} learned at training time so that
        inference rows encode categories identically to training. When None (training),
        category dtypes are inferred from the data.

    Returns
    -------
    pd.DataFrame with exactly `FEATURE_COLUMNS`, correct dtypes, NaNs preserved.
    """
    df = df_raw.copy()

    # --- 1. Temporal features from Announce Date, then the raw date is dropped later. ---
    df["Announce Date"] = pd.to_datetime(df.get("Announce Date"), errors="coerce")
    df["Announce_Year"] = df["Announce Date"].dt.year.astype("Int64")
    df["Announce_Month"] = df["Announce Date"].dt.month.astype("Int64")

    # --- 2. Text-parsed binary flags from Deal Attributes. ---
    attrs = df.get("Deal Attributes", pd.Series(index=df.index, dtype="object"))
    attrs = attrs.astype("string").fillna("")
    for new_col, needle in DEAL_ATTRIBUTE_FLAGS.items():
        df[new_col] = attrs.str.contains(needle, case=False, regex=False).astype("int8")

    # --- 3. Numeric features -> float, NaNs left intact (no imputation). ---
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # --- 4. Binary flags -> 0/1 int8. ---
    for col in BINARY_FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = _coerce_binary(df[col])

    # --- 5. Categorical features -> pandas category dtype (native XGBoost handling). ---
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = pd.NA
        if categories is not None and col in categories:
            df[col] = df[col].astype("string").astype(categories[col])
        else:
            df[col] = df[col].astype("string").astype("category")

    # --- 6. Temporal -> plain integers (nullable Int handled by XGBoost). ---
    for col in TEMPORAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # --- 7. Select the exact, ordered feature schema. ---
    X = df[FEATURE_COLUMNS].copy()
    return X


# --------------------------------------------------------------------------------------
# Data loading + assembly
# --------------------------------------------------------------------------------------
def load_dataset(file_path: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load the Excel file, build X / y and the chronological year key.

    Returns (X, y, announce_year) all aligned on the same index.
    """
    print(f"Loading dataset from: {file_path}")
    df = pd.read_excel(file_path, sheet_name=SHEET_NAME)
    print(f"  Raw shape: {df.shape}")

    # Report and drop every banned column so the intent is auditable.
    banned = [c for c in df.columns if _is_banned_name(c) or c in EXPLICIT_DROP_COLUMNS]
    print(f"  Dropping {len(banned)} ticker/reference/helper/raw columns.")

    # Target isolated up front; guaranteed never to reach X.
    y = df[TARGET].astype("float64")
    announce_year = pd.to_datetime(df["Announce Date"], errors="coerce").dt.year

    X = build_features(df)

    # ---------------------------- Leakage / hygiene assertions ----------------------------
    assert TARGET not in X.columns, "Target leakage: 'Business Days To Complete' present in X!"
    for col in X.columns:
        assert not _is_banned_name(col), f"Banned column leaked into X: {col!r}"
    assert list(X.columns) == FEATURE_COLUMNS, "Feature columns diverged from schema."
    print(f"  Feature matrix shape: {X.shape} ({X.shape[1]} features, no leakage).")

    return X, y, announce_year


# --------------------------------------------------------------------------------------
# Chronological splitting
# --------------------------------------------------------------------------------------
def chronological_split(X, y, announce_year, announce_date):
    """Train on year <= 2023, test on year >= 2024; last 20% of train (by date) -> val."""
    train_mask = announce_year <= 2023
    test_mask = announce_year >= 2024

    X_train_full = X[train_mask].copy()
    y_train_full = y[train_mask].copy()
    X_test = X[test_mask].copy()
    y_test = y[test_mask].copy()

    print(f"\nChronological split:")
    print(f"  Train (Announce_Year <= 2023): {len(X_train_full)} rows")
    print(f"  Test  (Announce_Year >= 2024): {len(X_test)} rows")

    # Validation = latest 20% of the training period, ordered by actual announce date.
    train_dates = announce_date[train_mask]
    order = train_dates.sort_values().index
    n_val = max(1, int(round(0.20 * len(order))))
    val_idx = order[-n_val:]
    train_idx = order[:-n_val]

    X_train, y_train = X_train_full.loc[train_idx], y_train_full.loc[train_idx]
    X_val, y_val = X_train_full.loc[val_idx], y_train_full.loc[val_idx]
    print(f"  -> fit slice: {len(X_train)} rows | early-stopping val slice: {len(X_val)} rows")

    return X_train, y_train, X_val, y_val, X_test, y_test, X_train_full, y_train_full


# --------------------------------------------------------------------------------------
# Hyper-parameter search
# --------------------------------------------------------------------------------------
def tune_hyperparameters(X_train_full, y_train_full, objective):
    """RandomizedSearchCV over a TimeSeriesSplit to avoid look-ahead leakage."""
    param_dist = {
        "n_estimators": [100, 300, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
    }

    base = xgb.XGBRegressor(
        tree_method="hist",
        enable_categorical=True,
        objective=objective,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    tscv = TimeSeriesSplit(n_splits=4)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    print(f"\nTuning hyper-parameters (objective={objective}) via RandomizedSearchCV + "
          f"TimeSeriesSplit ...")
    search.fit(X_train_full, y_train_full)
    print(f"  Best CV MAE: {-search.best_score_:.2f} days")
    print(f"  Best params: {search.best_params_}")
    return search.best_params_


# --------------------------------------------------------------------------------------
# Final fit with early stopping
# --------------------------------------------------------------------------------------
def fit_final_model(best_params, X_train, y_train, X_val, y_val, objective, **extra):
    """Refit the winning configuration once, with early stopping on the val slice."""
    model = xgb.XGBRegressor(
        tree_method="hist",
        enable_categorical=True,
        objective=objective,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=20,
        **best_params,
        **extra,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best_it = getattr(model, "best_iteration", None)
    if best_it is not None:
        print(f"  Early stopping selected iteration: {best_it}")
    return model


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def evaluate(model, X_test, y_test, label="model"):
    """Report MAE, MAPE and R2 (RMSE intentionally de-emphasised due to long tail)."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100.0
    r2 = r2_score(y_test, preds)
    print(f"\nTest-set performance ({label}):")
    print(f"  MAE : {mae:8.2f} days")
    print(f"  MAPE: {mape:8.2f} %")
    print(f"  R2  : {r2:8.4f}")
    return preds, {"MAE": mae, "MAPE": mape, "R2": r2}


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def plot_feature_importance(model, path=IMPORTANCE_PLOT_PATH, top_n=15):
    """Print + plot the top-N features by gain."""
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp = (
        pd.Series(gain)
        .sort_values(ascending=False)
        .head(top_n)
    )
    print(f"\nTop {top_n} features by gain:")
    for name, val in imp.items():
        print(f"  {name:<45} {val:12.2f}")

    plt.figure(figsize=(9, 7))
    imp.iloc[::-1].plot(kind="barh", color="#2c7fb8")
    plt.xlabel("Gain")
    plt.title(f"Top {top_n} Feature Importances (gain)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved importance plot -> {path}")


def plot_shap_summary(model, X_sample, path=SHAP_PLOT_PATH, top_n=10):
    """Beeswarm SHAP summary for the top features."""
    print("\nComputing SHAP values ...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        plt.figure()
        shap.summary_plot(
            shap_values, X_sample, max_display=top_n, show=False, plot_type="dot"
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved SHAP summary plot -> {path}")
    except Exception as exc:  # SHAP can be brittle with native categoricals across versions
        print(f"  [warning] SHAP summary skipped: {exc}")


# --------------------------------------------------------------------------------------
# Reusable inference
# --------------------------------------------------------------------------------------
def predict_days(new_deal_dict: dict) -> float:
    """Predict `Business Days To Complete` for a single raw deal dictionary.

    The dictionary should use the original Excel column names (e.g. 'Announce Date',
    'Payment Type', 'SAMR', 'Deal Attributes', ...). Missing keys are tolerated and
    treated as NaN / absent flags. Preprocessing mirrors training exactly.
    """
    # Lazy-load artifacts so the function is self-contained for downstream reuse.
    with open(PREPROCESSOR_PATH, "rb") as fh:
        preproc = pickle.load(fh)

    model = xgb.XGBRegressor(enable_categorical=True)
    model.load_model(MODEL_PATH)

    df_raw = pd.DataFrame([new_deal_dict])
    X_new = build_features(df_raw, categories=preproc["categories"])
    X_new = X_new[preproc["feature_columns"]]
    return float(model.predict(X_new)[0])


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    # 1. Load + build features (with leakage assertions inside).
    X, y, announce_year = load_dataset(FILE_PATH)
    announce_date = pd.to_datetime(
        pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)["Announce Date"], errors="coerce"
    )

    # 2. Chronological split.
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train_full,
        y_train_full,
    ) = chronological_split(X, y, announce_year, announce_date)

    # 3. Tune + fit the primary MAE model.
    best_params = tune_hyperparameters(X_train_full, y_train_full, "reg:absoluteerror")
    print("\nFitting final MAE model with early stopping ...")
    model = fit_final_model(
        best_params, X_train, y_train, X_val, y_val, "reg:absoluteerror"
    )

    # 4. Evaluate primary model.
    evaluate(model, X_test, y_test, label="MAE (reg:absoluteerror)")

    # 5. Optional P75 (worst-case) quantile model for planning purposes.
    print("\n" + "=" * 70)
    print("Building optional P75 quantile model (reg:quantileerror, alpha=0.75) ...")
    p75_model = fit_final_model(
        best_params,
        X_train,
        y_train,
        X_val,
        y_val,
        "reg:quantileerror",
        quantile_alpha=0.75,
    )
    evaluate(p75_model, X_test, y_test, label="P75 (reg:quantileerror, alpha=0.75)")

    # 6. Diagnostics: importance + SHAP.
    plot_feature_importance(model)
    shap_sample = X_test if len(X_test) <= 400 else X_test.sample(400, random_state=RANDOM_STATE)
    plot_shap_summary(model, shap_sample)

    # 7. Persist artifacts.
    model.save_model(MODEL_PATH)
    p75_model.save_model(P75_MODEL_PATH)
    print(f"\nSaved models -> {MODEL_PATH}, {P75_MODEL_PATH}")

    preprocessor = {
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "binary_features": BINARY_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "temporal_features": TEMPORAL_FEATURES,
        "text_flag_features": TEXT_FLAG_FEATURES,
        "deal_attribute_flags": DEAL_ATTRIBUTE_FLAGS,
        # Save fitted category definitions so inference encodes identically to training.
        "categories": {c: X[c].dtype for c in CATEGORICAL_FEATURES},
        "best_params": best_params,
    }
    with open(PREPROCESSOR_PATH, "wb") as fh:
        pickle.dump(preprocessor, fh)
    print(f"Saved preprocessor metadata -> {PREPROCESSOR_PATH}")

    # 8. Smoke-test the reusable prediction function on the first test deal.
    print("\nSmoke-testing predict_days() on a sample raw deal ...")
    raw = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
    test_row = raw[pd.to_datetime(raw["Announce Date"]).dt.year >= 2024].iloc[0].to_dict()
    predicted = predict_days(test_row)
    print(f"  Predicted Business Days To Complete: {predicted:.1f} (actual: {test_row[TARGET]})")

    print("\nDone. Artifacts written; pipeline exited cleanly.")


if __name__ == "__main__":
    main()
