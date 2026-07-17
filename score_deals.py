"""
Unified deal scorer -- run BOTH M&A completion-time models on the same deal(s).
==============================================================================

For every deal this produces two independent estimates of `Business Days To Complete`:

  1. full_history  -- the two-stage blended XGBoost ensemble trained on the full
                      2017-2026 dataset (~1,170 deals). Captures the regulatory long-tail
                      (SAMR/EC/CFIUS routing + sponsor exception). Best for large, complex,
                      or cross-border/regulatory deals.
  2. recent_2025+  -- the RandomForest+ExtraTrees bagged ensemble trained only on deals
                      announced 2025 onward (n=155). Reflects the current deal regime; more
                      robust on small, present-day, non-regulatory deals.

Two numbers, two lenses. Where they agree you have high confidence; where they diverge the
gap itself is informative (typically the full-history model runs longer on regulatory deals).

USAGE
  # score every deal in an .xlsx / .csv (rows = deals, original Excel column names):
  python score_deals.py path/to/deals.xlsx
  python score_deals.py path/to/deals.xlsx --out results.csv

  # or import and score a DataFrame / list of dicts programmatically:
  from score_deals import score_deals
  score_deals(df)            # -> DataFrame with both predictions + calendar conversions

Both models share ONE leakage-free preprocessing definition (build_features), so the two
estimates are strictly comparable. Predictions are in BUSINESS days; calendar-day and month
conversions are added for convenience (x1.4 and /21.7 respectively).
"""

import argparse
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

# Shared feature builder + the class needed to unpickle the two-stage ensemble.
from xgboost_ma_completion_model import build_features
import ma_completion_twostage_ensemble as tse  # noqa: F401  (registers TwoStageEnsemble for pickle)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BUSINESS_DAYS_PER_CALENDAR = 7 / 5      # ~1.4 calendar days per business day
BUSINESS_DAYS_PER_MONTH = 21.7          # ~21.7 business days per calendar month

# Artifact paths (see MODELS.md).
FULL_MODEL_PATH = "ma_twostage_ensemble.pkl"
FULL_PREP_PATH = "twostage_preprocessor.pkl"
RECENT_MODEL_PATH = "ma_2025plus_model.pkl"
RECENT_PREP_PATH = "ma_2025plus_preprocessor.pkl"


def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_models():
    """Load both fitted models + their metadata once. Returns a dict."""
    return {
        "full_model": _load(FULL_MODEL_PATH),
        "full_prep": _load(FULL_PREP_PATH),
        "recent_model": _load(RECENT_MODEL_PATH),
        "recent_prep": _load(RECENT_PREP_PATH),
    }


def _predict_full_history(df, bundle):
    """Two-stage blended ensemble (full-history) predictions for a raw deal DataFrame."""
    prep = bundle["full_prep"]
    X = build_features(df, categories=prep["categories"])[prep["feature_columns"]]
    for c in prep["categorical_features"]:
        X[c] = X[c].astype(prep["categories"][c])
    return np.asarray(bundle["full_model"].predict(X), dtype=float)


def _predict_recent(df, bundle):
    """RandomForest+ExtraTrees (2025+) predictions for a raw deal DataFrame."""
    meta = bundle["recent_prep"]
    X = build_features(df)[meta["feature_columns"]]  # aligns to the 2025+ trained schema
    return np.asarray(bundle["recent_model"].predict(X), dtype=float)


def score_deals(deals, bundle=None):
    """Score one or more deals with BOTH models.

    Parameters
    ----------
    deals : pandas.DataFrame | dict | list[dict]
        Raw deal(s) using the original Excel column names.
    bundle : dict | None
        Pre-loaded models from load_models(); loaded on demand if omitted.

    Returns
    -------
    pandas.DataFrame with one row per deal:
        full_history_bus_days, recent_2025plus_bus_days,
        <each>_calendar_days, <each>_months, and their spread.
    """
    if isinstance(deals, dict):
        deals = [deals]
    df = pd.DataFrame(deals) if not isinstance(deals, pd.DataFrame) else deals.copy()
    if bundle is None:
        bundle = load_models()

    full = _predict_full_history(df, bundle)
    recent = _predict_recent(df, bundle)

    out = pd.DataFrame({
        "full_history_bus_days": np.round(full, 1),
        "recent_2025plus_bus_days": np.round(recent, 1),
    })
    out["spread_bus_days"] = np.round(full - recent, 1)
    out["full_history_calendar_days"] = np.round(full * BUSINESS_DAYS_PER_CALENDAR, 0)
    out["recent_2025plus_calendar_days"] = np.round(recent * BUSINESS_DAYS_PER_CALENDAR, 0)
    out["full_history_months"] = np.round(full / BUSINESS_DAYS_PER_MONTH, 1)
    out["recent_2025plus_months"] = np.round(recent / BUSINESS_DAYS_PER_MONTH, 1)

    # Attach a couple of identifier columns if present, for readability.
    for id_col in ("Target Ticker", "Acquirer Ticker", "Announce Date"):
        if id_col in df.columns:
            out.insert(0, id_col, df[id_col].values)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score M&A deal(s) with both completion-time models.")
    ap.add_argument("input", help="Path to .xlsx or .csv with one or more deals (rows).")
    ap.add_argument("--sheet", default=None,
                    help="Excel sheet name (default: first sheet in the workbook).")
    ap.add_argument("--out", default=None, help="Optional path to write results as CSV.")
    args = ap.parse_args(argv)

    if args.input.lower().endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        # Default to the first sheet (index 0) so exports with arbitrary sheet names work.
        df = pd.read_excel(args.input, sheet_name=args.sheet if args.sheet is not None else 0)

    results = score_deals(df)
    pd.set_option("display.max_columns", None, "display.width", 200)
    print(f"\nScored {len(results)} deal(s) with both models (Business Days To Complete):\n")
    print(results.to_string(index=False))
    if args.out:
        results.to_csv(args.out, index=False)
        print(f"\nSaved -> {args.out}")
    return results


if __name__ == "__main__":
    main(sys.argv[1:])
