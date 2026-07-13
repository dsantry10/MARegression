"""
Stacked ensemble for M&A `Business Days To Complete`.
=====================================================

Combines the project's three base regression models -- Linear Regression, Random Forest,
and XGBoost -- into a stacked ensemble and tests whether stacking beats the best single
model on 5-fold cross-validated R2.

(The task brief mentions "4 models"; the project defines three base learners -- Linear
Regression, Random Forest, XGBoost -- so those three are used as the level-0 learners.
The meta-learner is the 4th model in the stack.)

Design decisions
----------------
* Preprocessing is shared and applied ONCE, inside CV, so results are comparable and
  leak-free: median-impute + standardize numerics; most-frequent-impute + one-hot
  (rare levels grouped) categoricals. Reuses the vetted, leakage-free `build_features`
  feature schema from xgboost_ma_completion_model.py.
* All learners fit on a log1p target (the raw target is right-skewed; squared-error on the
  raw scale chases the long tail and posts negative R2). Metrics are reported back on the
  original business-day scale via TransformedTargetRegressor.
* Out-of-fold stacking (StackingRegressor, cv=5) so base predictions feeding the meta-model
  carry no leakage.
* Meta-learners compared: Ridge and a small XGBoost, each with passthrough False/True
  (passthrough feeds the original features to the meta-model alongside base predictions).

Run:
    python ma_stacked_ensemble.py
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

from xgboost_ma_completion_model import (
    CATEGORICAL_FEATURES,
    FILE_PATH,
    SHEET_NAME,
    TARGET,
    build_features,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
MODEL_PATH = "ma_stacked_ensemble.pkl"


# --------------------------------------------------------------------------------------
# Data + shared preprocessing
# --------------------------------------------------------------------------------------
def load_xy():
    raw = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
    X = build_features(raw)
    y = raw[TARGET].astype(float).values
    assert TARGET not in X.columns, "Target leakage!"
    return X, y


def make_shared_preprocessor(X):
    """One preprocessing used by every learner (comparable, leak-free)."""
    num = [c for c in X.columns if str(X[c].dtype).startswith(("float", "int", "Int"))]
    cat = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                               sparse_output=False))]), cat),
    ])


def log_target(regressor):
    """Fit on log1p(y), predict back on the original business-day scale."""
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


# ---- Base learners (existing hyperparameters where defined) --------------------------
def base_linear():
    return LinearRegression()


def base_rf():
    # Existing project used 500 trees; add light regularization for a small/noisy target.
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.6,
                                 random_state=RANDOM_STATE, n_jobs=-1)


def base_xgb():
    # Regularized config consistent with the project's XGBoost work.
    return xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                            min_child_weight=6, random_state=RANDOM_STATE, n_jobs=-1)


def wrap(pre, estimator):
    """pre -> estimator, fitted on log target."""
    return log_target(Pipeline([("pre", pre), ("m", estimator)]))


# --------------------------------------------------------------------------------------
# Stacking
# --------------------------------------------------------------------------------------
def make_stack(pre, meta, passthrough):
    """StackingRegressor with out-of-fold (cv=5) base predictions, on a log target."""
    estimators = [("linear", base_linear()), ("rf", base_rf()), ("xgb", base_xgb())]
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=meta,
        cv=5,
        passthrough=passthrough,
        n_jobs=1,
    )
    # Preprocess once, then stack; whole thing fit on log target inside CV.
    return log_target(Pipeline([("pre", pre), ("stack", stack)]))


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def cv_metrics(model, X, y, cv):
    r2 = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=1)
    rmse = -cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
    mae = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
    return r2.mean(), r2.std(), rmse.mean(), mae.mean()


def base_prediction_correlation(pre, X, y, cv):
    """Correlation of base models' out-of-fold predictions (stacking-diversity check)."""
    preds = {}
    for name, est in [("linear", base_linear()), ("rf", base_rf()), ("xgb", base_xgb())]:
        preds[name] = cross_val_predict(wrap(pre, est), X, y, cv=cv, n_jobs=1)
    corr = pd.DataFrame(preds).corr()
    return corr


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    X, y = load_xy()
    print(f"Dataset: {len(X)} deals, {X.shape[1]} features. Target='{TARGET}' "
          f"(median={np.median(y):.0f}, skew={pd.Series(y).skew():.2f} business days)\n")
    pre = make_shared_preprocessor(X)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []

    # 1. Base models.
    for name, est in [("Linear Regression", base_linear()),
                      ("Random Forest", base_rf()),
                      ("XGBoost", base_xgb())]:
        r2m, r2s, rmse, mae = cv_metrics(wrap(pre, est), X, y, cv)
        rows.append((name, r2m, r2s, rmse, mae))
        print(f"  base   {name:22} R2={r2m:+.3f}+/-{r2s:.3f}  RMSE={rmse:5.1f}  MAE={mae:5.1f}")

    # 2. Stacking variants.
    variants = [
        ("Stack: Ridge",            Ridge(alpha=1.0),                       False),
        ("Stack: Ridge + passthru", Ridge(alpha=1.0),                       True),
        ("Stack: XGB-meta",         xgb.XGBRegressor(n_estimators=200, max_depth=2,
                                        learning_rate=0.05, reg_lambda=5.0,
                                        random_state=RANDOM_STATE), False),
        ("Stack: XGB-meta + passthru", xgb.XGBRegressor(n_estimators=200, max_depth=2,
                                        learning_rate=0.05, reg_lambda=5.0,
                                        random_state=RANDOM_STATE), True),
    ]
    stack_models = {}
    for name, meta, passthrough in variants:
        model = make_stack(pre, meta, passthrough)
        stack_models[name] = model
        r2m, r2s, rmse, mae = cv_metrics(model, X, y, cv)
        rows.append((name, r2m, r2s, rmse, mae))
        print(f"  stack  {name:22} R2={r2m:+.3f}+/-{r2s:.3f}  RMSE={rmse:5.1f}  MAE={mae:5.1f}")

    # 3. Comparison table.
    table = pd.DataFrame(rows, columns=["model", "cv_r2", "cv_r2_std", "rmse", "mae"])
    print("\n" + "=" * 68)
    print("COMPARISON (5-fold CV, shuffled, random_state=%d)" % RANDOM_STATE)
    print("=" * 68)
    print(table.to_string(index=False,
          formatters={"cv_r2": "{:+.3f}".format, "cv_r2_std": "{:.3f}".format,
                      "rmse": "{:.1f}".format, "mae": "{:.1f}".format}))

    best_base = table.iloc[:3].sort_values("cv_r2").iloc[-1]
    best_overall = table.sort_values("cv_r2").iloc[-1]

    # 4. Base-prediction correlation diagnostic.
    corr = base_prediction_correlation(pre, X, y, cv)
    print("\nBase-model out-of-fold prediction correlation:")
    print(corr.to_string(float_format="{:.3f}".format))
    offdiag = corr.where(~np.eye(len(corr), dtype=bool)).stack()
    if (offdiag > 0.95).any():
        print("  [!] Base models are highly correlated (>0.95) -> limited stacking gains; "
              "consider more diverse base learners.")
    else:
        print(f"  Base models are usefully diverse (max off-diagonal corr={offdiag.max():.3f}).")

    # 5. Winner + honest verdict.
    print("\n" + "=" * 68)
    improvement = best_overall["cv_r2"] - best_base["cv_r2"]
    print(f"Best single base model : {best_base['model']} (R2={best_base['cv_r2']:+.3f})")
    print(f"Best overall           : {best_overall['model']} (R2={best_overall['cv_r2']:+.3f})")
    if best_overall["model"] == best_base["model"]:
        print("\nVERDICT: Stacking did NOT beat the best single model. Reported honestly.")
        print("  Next steps: tune base hyperparameters, add more diverse base learners "
              "(e.g. ExtraTrees, kNN), or engineer features. Small/noisy target caps gains.")
    else:
        print(f"\nVERDICT: '{best_overall['model']}' wins, improving CV R2 by "
              f"{improvement:+.3f} over the best single model ({best_base['model']}).")
        if improvement < 0.01:
            print("  CAVEAT: the improvement is within CV noise -- treat as marginal.")

    # 6. Persist the best-performing STACKING configuration (fit on all data). Even when a
    #    single base model wins overall, we save the strongest stack (the deliverable here).
    stack_table = table[table["model"].isin(stack_models)]
    best_stack_name = stack_table.sort_values("cv_r2").iloc[-1]["model"]
    winner = stack_models[best_stack_name]
    winner.fit(X, y)
    joblib.dump(winner, MODEL_PATH)
    print(f"\nSaved best stacking configuration ('{best_stack_name}', "
          f"R2={stack_table['cv_r2'].max():+.3f}) -> {MODEL_PATH}")
    print("\nCAVEATS: n=%d with a heavy-tailed target imposes a real R2 ceiling; CV std is "
          "sizeable, so small differences are noise." % len(X))


if __name__ == "__main__":
    main()
