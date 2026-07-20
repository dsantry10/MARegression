"""
Two-Stage XGBoost model -- STOCK-CONSOLIDATED payment variant.
==============================================================

Identical architecture to `ma_completion_twostage_ensemble.py` (mixture-of-experts +
SAMR/EC regulatory floor), but with ONE deliberate difference in preprocessing:

    Any payment type that contains "Stock" is collapsed to "Stock".
        Cash and Stock  -> Stock
        Cash or Stock   -> Stock
        Stock           -> Stock
        Cash / Cash and Debt / Undisclosed -> unchanged (no stock component)

This treats every stock-containing consideration as a single "Stock" category. It is kept
as a SEPARATE model with its own artifacts so it is never conflated with the standard
model:

    standard model   ->  ma_twostage_ensemble.pkl        / twostage_preprocessor.pkl
    stock-consolidated -> ma_twostage_stockpay_ensemble.pkl / twostage_stockpay_preprocessor.pkl

Everything else (feature schema, threshold selection, evaluation) is reused unchanged from
the base module. Target is in BUSINESS days.

Run:
    python ma_completion_twostage_stockpay.py
"""
import pickle
import warnings

import numpy as np
import pandas as pd

from xgboost_ma_completion_model import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FILE_PATH,
    SHEET_NAME,
    TARGET,
    build_features,
)
import ma_completion_twostage_ensemble as base

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = base.RANDOM_STATE

# ---- Separate artifact paths (never overwrite the standard model) --------------------
ENSEMBLE_PATH = "ma_twostage_stockpay_ensemble.pkl"
PREPROCESSOR_PATH = "twostage_stockpay_preprocessor.pkl"
IMPORTANCE_PLOT_PATH = "twostage_stockpay_importance.png"
SHAP_PLOT_PATH = "twostage_stockpay_shap.png"


# --------------------------------------------------------------------------------------
# The one preprocessing difference: collapse any stock-containing payment to "Stock".
# --------------------------------------------------------------------------------------
def collapse_stock_payment(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with any 'Stock'-containing Payment Type mapped to 'Stock'."""
    out = df.copy()
    if "Payment Type" in out.columns:
        pt = out["Payment Type"].astype("string")
        is_stock = pt.str.contains("Stock", case=False, na=False)
        out.loc[is_stock, "Payment Type"] = "Stock"
    return out


def build_features_stockpay(df: pd.DataFrame, categories=None) -> pd.DataFrame:
    """build_features, but with stock-containing payment types collapsed to 'Stock'."""
    return build_features(collapse_stock_payment(df), categories=categories)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_xy():
    raw = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
    X = build_features_stockpay(raw)
    y = raw[TARGET].astype(float).values
    assert TARGET not in X.columns
    assert list(X.columns) == FEATURE_COLUMNS
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    announce_year = pd.to_datetime(raw["Announce Date"], errors="coerce").dt.year
    return raw, X, pd.Series(y), announce_year


# --------------------------------------------------------------------------------------
# Inference (variant-specific: collapse payment + use variant categories)
# --------------------------------------------------------------------------------------
def predict_days(new_deal_dict: dict) -> float:
    with open(PREPROCESSOR_PATH, "rb") as fh:
        prep = pickle.load(fh)
    with open(ENSEMBLE_PATH, "rb") as fh:
        ensemble = pickle.load(fh)
    X = build_features_stockpay(pd.DataFrame([new_deal_dict]), categories=prep["categories"])
    X = X[prep["feature_columns"]]
    for c in prep["categorical_features"]:
        X[c] = X[c].astype(prep["categories"][c])
    return float(ensemble.predict(X)[0])


# --------------------------------------------------------------------------------------
# Main -- reuses the base module's CV/eval/plot machinery on the collapsed features.
# --------------------------------------------------------------------------------------
def main():
    raw, X, y, announce_year = load_xy()  # y is a pandas Series (base CV helpers use .iloc)
    print("STOCK-CONSOLIDATED two-stage model")
    print(f"Payment Type after collapse: "
          f"{collapse_stock_payment(raw)['Payment Type'].value_counts().to_dict()}\n")

    # 1. Tune the regime threshold against repeated CV (same procedure as standard model).
    threshold = base.select_threshold_by_cv(X, y)

    # 2. Report both metrics.
    print("\n" + "=" * 72)
    print("REPEATED 3x5-FOLD CV R2 (tuning metric)")
    two_mean, two_std = base.repeated_cv_r2(lambda: base.TwoStageEnsemble(threshold=threshold), X, y)
    print(f"  Stock-consolidated two-stage ensemble : {two_mean:.3f} +/- {two_std:.3f}")

    print("\n" + "=" * 72)
    print("HONEST OUT-OF-TIME (train <=2023, test >=2024)")
    ensemble, _ = base.report_out_of_time(
        base.TwoStageEnsemble(threshold=threshold), X, y, announce_year
    )

    # 3. Refit on all data + diagnostics.
    print("\nFitting final stock-consolidated ensemble on full data ...")
    ensemble = base.TwoStageEnsemble(threshold=threshold).fit(X, y)
    base.plot_importance(ensemble, path=IMPORTANCE_PLOT_PATH)
    te = announce_year >= 2024
    shap_sample = X[te]
    if len(shap_sample) > 400:
        shap_sample = shap_sample.sample(400, random_state=RANDOM_STATE)
    base.plot_shap(ensemble, shap_sample, path=SHAP_PLOT_PATH)

    # 4. Persist SEPARATE artifacts.
    with open(ENSEMBLE_PATH, "wb") as fh:
        pickle.dump(ensemble, fh)
    prep = {
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "categories": {c: X[c].dtype for c in CATEGORICAL_FEATURES},
        "threshold": threshold,
        "weights": base.EXPERT_WEIGHTS,
        "payment_treatment": "stock_consolidated",  # provenance tag
    }
    with open(PREPROCESSOR_PATH, "wb") as fh:
        pickle.dump(prep, fh)
    print(f"\nSaved stock-consolidated model -> {ENSEMBLE_PATH}")
    print(f"Saved preprocessor -> {PREPROCESSOR_PATH}")

    # 5. Smoke-test.
    sample = raw[pd.to_datetime(raw["Announce Date"]).dt.year >= 2024].iloc[0].to_dict()
    print(f"\nSmoke-test predict_days: {predict_days(sample):.1f} "
          f"(actual {sample[TARGET]} business days)")
    print("\nDone. Stock-consolidated two-stage model trained and saved (separate artifacts).")


if __name__ == "__main__":
    main()
