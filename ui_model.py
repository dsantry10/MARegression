"""
UI model layer: streamlined-inputs -> deal dict -> two-stage XGBoost prediction.

Variant-aware: the same code serves two SEPARATE, independently trained models via a
registry (no conflation — each has its own artifacts):

  * "standard"  -> ma_twostage_ensemble.pkl          (payment types as-is)
  * "stockpay"  -> ma_twostage_stockpay_ensemble.pkl  (any stock-containing payment
                    collapsed to "Stock")

Thin wrapper over the existing model so the UI never re-implements anything: derives
Log TV/Revenue/Equity from dollar inputs, normalizes category labels (fixes the
"Real Estate REIT" silent-mislabel), collapses payment for the stockpay variant, and
returns the two-stage experts (short/long), P(long), the weighted blend, floor status,
and the 2025+ recency cross-check.
"""
from __future__ import annotations

import functools
import pickle

import numpy as np
import pandas as pd

from xgboost_ma_completion_model import build_features
import ma_completion_twostage_ensemble as tse  # registers TwoStageEnsemble for unpickling

BUSINESS_DAYS_PER_CAL = 7 / 5
BUSINESS_DAYS_PER_MONTH = 21.7

# The 2025+ recency cross-check model is shared (standard payment) across variants.
RECENT_MODEL_PATH = "ma_2025plus_model.pkl"
RECENT_PREP_PATH = "ma_2025plus_preprocessor.pkl"
DATASET_PATH = "LARGE_DATASET (TOGGLES).xlsx"

# Model registry -- each variant is a fully separate trained artifact set.
MODELS = {
    "standard": {
        "label": "Standard — payment types as reported",
        "model": "ma_twostage_ensemble.pkl",
        "prep": "twostage_preprocessor.pkl",
        "collapse_stock_payment": False,
    },
    "stockpay": {
        "label": "Stock-consolidated — any stock consideration treated as Stock",
        "model": "ma_twostage_stockpay_ensemble.pkl",
        "prep": "twostage_stockpay_preprocessor.pkl",
        "collapse_stock_payment": True,
    },
}
DEFAULT_VARIANT = "stockpay"

INDUSTRY_ALIASES = {
    "real estate reit": "Real Estate",
    "reit": "Real Estate",
    "real estate investment trust": "Real Estate",
}


def _collapse_payment(df: pd.DataFrame) -> pd.DataFrame:
    """Map any stock-containing Payment Type to 'Stock' (stockpay variant)."""
    out = df.copy()
    if "Payment Type" in out.columns:
        pt = out["Payment Type"].astype("string")
        out.loc[pt.str.contains("Stock", case=False, na=False), "Payment Type"] = "Stock"
    return out


@functools.lru_cache(maxsize=4)
def _load(variant: str):
    cfg = MODELS[variant]
    with open(cfg["prep"], "rb") as fh:
        prep = pickle.load(fh)
    with open(cfg["model"], "rb") as fh:
        model = pickle.load(fh)
    return model, prep, cfg


@functools.lru_cache(maxsize=1)
def _load_recent():
    with open(RECENT_PREP_PATH, "rb") as fh:
        recent_prep = pickle.load(fh)
    with open(RECENT_MODEL_PATH, "rb") as fh:
        recent_model = pickle.load(fh)
    return recent_model, recent_prep


def _prep_features(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Build the feature matrix for a variant (with payment collapse if required)."""
    _, prep, cfg = _load(variant)
    src = _collapse_payment(df) if cfg["collapse_stock_payment"] else df
    X = build_features(src, categories=prep["categories"])[prep["feature_columns"]]
    for c in prep["categorical_features"]:
        X[c] = X[c].astype(prep["categories"][c])
    return X


def known_categories(field: str, variant: str = DEFAULT_VARIANT) -> list[str]:
    """Category values the model recognizes for a field (reflects the variant's vocab,
    e.g. the collapsed Payment Type list for the stockpay model)."""
    _, prep, _ = _load(variant)
    return sorted(prep["categories"][field].categories.tolist())


def _log10_or_nan(value) -> float:
    try:
        v = float(value)
        return float(np.log10(v)) if v and v > 0 else np.nan
    except (TypeError, ValueError):
        return np.nan


def normalize_industry(value: str, variant: str = DEFAULT_VARIANT):
    if value is None:
        return value, None
    known = set(known_categories("Target Industry Group", variant))
    if value in known:
        return value, None
    alias = INDUSTRY_ALIASES.get(str(value).strip().lower())
    if alias and alias in known:
        return alias, f"Industry '{value}' mapped to '{alias}' (training-vocabulary label)."
    return value, (f"Industry '{value}' is not in the model's vocabulary; it will be "
                   f"treated as missing and weaken the estimate.")


def build_deal_dict(inp: dict, variant: str = DEFAULT_VARIANT) -> tuple[dict, list[str]]:
    """Map streamlined UI inputs to a raw deal dict; return (deal, warnings)."""
    warnings: list[str] = []
    industry, warn = normalize_industry(inp.get("industry_group"), variant)
    if warn:
        warnings.append(warn)

    total_value = inp.get("total_value")
    equity_value = inp.get("equity_value")
    revenue = inp.get("revenue")

    tc, ac = inp.get("target_country"), inp.get("acquirer_country")
    cross_border = inp.get("cross_border")
    if cross_border is None and tc and ac:
        cross_border = "Yes" if tc != ac else "No"

    def yn(v):
        return "Yes" if v else "No"

    deal = {
        "Target Ticker": inp.get("target", ""),
        "Acquirer Ticker": inp.get("acquirer", ""),
        "Announce Date": inp.get("announce_date"),
        "Announced Total Value (mil.)": total_value,
        "Payment Type": inp.get("payment_type"),
        "TV/EBITDA": inp.get("tv_ebitda"),
        "Target Industry Group": industry,
        "Acquirer Termination Fee": inp.get("acq_term_fee"),
        "Target Termination Fee": inp.get("tgt_term_fee"),
        "Announced Premium": inp.get("premium"),
        "Announced Equity Value (mil.)": equity_value,
        "Target Country/Region": tc,
        "Acquirer Country/Region": ac,
        "Nature of Bid": inp.get("nature_of_bid"),
        "Target Sales/Revenue/Turnover": revenue,
        "Target Trailg 12 Mth Operating Margin": inp.get("operating_margin"),
        "Deal Attributes": inp.get("deal_attributes", "Company Takeover"),
        "Additional Stake Purchase": yn(inp.get("additional_stake")),
        "Competing Bid": yn(inp.get("competing_bid")),
        "Cross Border": cross_border or "No",
        "Going Private": yn(inp.get("going_private")),
        "PE Buyout": yn(inp.get("pe_buyout")),
        "Tender Offer": yn(inp.get("tender_offer")),
        "Log TV": _log10_or_nan(total_value),
        "Log Revenue": _log10_or_nan(revenue),
        "Log Equity Value": _log10_or_nan(equity_value),
        "SAMR": int(bool(inp.get("samr"))),
        "EC": int(bool(inp.get("ec"))),
        "CFIUS": int(bool(inp.get("cfius"))),
    }
    if not total_value:
        warnings.append("Total value is blank — Log TV (a top feature) will be missing.")
    return deal, warnings


def predict(deal: dict, variant: str = DEFAULT_VARIANT) -> dict:
    """Run the selected two-stage model + shared 2025+ recency cross-check."""
    model, _, _ = _load(variant)
    df = pd.DataFrame([deal])
    X = _prep_features(df, variant)

    p_long = float(model._p_long(X)[0])
    short = float(model._predict_expert(model.expert_short, X)[0])
    long = float(model._predict_expert(model.expert_long, X)[0])
    weighted = float(model.predict(X)[0])

    floor_applied = (
        int(deal.get("SAMR", 0)) == 1
        and int(deal.get("EC", 0)) == 1
        and deal.get("PE Buyout") != "Yes"
    )

    # 2025+ recency cross-check (standard payment features).
    recent_model, recent_prep = _load_recent()
    Xr = build_features(df)[recent_prep["feature_columns"]]
    recent = float(recent_model.predict(Xr)[0])

    return {
        "variant": variant,
        "variant_label": MODELS[variant]["label"],
        "p_long": p_long,
        "short": short,
        "long": long,
        "weighted": weighted,
        "floor_applied": floor_applied,
        "recent_2025plus": recent,
        "weighted_calendar": weighted * BUSINESS_DAYS_PER_CAL,
        "weighted_months": weighted / BUSINESS_DAYS_PER_MONTH,
    }


def estimated_close_date(announce_date, business_days: float):
    try:
        start = pd.Timestamp(announce_date)
        return start + pd.tseries.offsets.BusinessDay(int(round(business_days)))
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _dataset():
    return pd.read_excel(DATASET_PATH, sheet_name="Sheet1")


def comparators(deal: dict, variant: str = DEFAULT_VARIANT) -> dict:
    """Median/mean close time of nearest historical deals, tightening the filter.

    For the stockpay variant the dataset's payment types are collapsed the same way so the
    comparator subset matches the model's view of the deal.
    """
    d = _dataset()
    if MODELS[variant]["collapse_stock_payment"]:
        d = _collapse_payment(d)
    target = "Business Days To Complete"
    y = d[target]
    ig = deal.get("Target Industry Group")

    cuts = []
    base = pd.Series(True, index=d.index)
    if ig in set(d["Target Industry Group"]):
        base = d["Target Industry Group"] == ig
        cuts.append((f"Industry = {ig}", base))
    pay = _collapse_payment(pd.DataFrame([deal]))["Payment Type"].iloc[0] \
        if MODELS[variant]["collapse_stock_payment"] else deal.get("Payment Type")
    if pay in set(d["Payment Type"].dropna()):
        m = base & (d["Payment Type"] == pay)
        if m.sum() >= 5:
            cuts.append((f"+ {pay}", m))
            base = m
    for col, key, label in [("Going Private", "Going Private", "Going Private"),
                            ("PE Buyout", "PE Buyout", "PE Buyout"),
                            ("Tender Offer", "Tender Offer", "Tender Offer")]:
        if deal.get(key) == "Yes":
            m = base & (d[col] == "Yes")
            if m.sum() >= 5:
                cuts.append((f"+ {label}", m))
                base = m
    rows = []
    for label, mask in cuts:
        rows.append({"filter": label, "n": int(mask.sum()),
                     "median": round(float(y[mask].median()), 1),
                     "mean": round(float(y[mask].mean()), 1)})
    return {"rows": rows, "overall_median": round(float(y.median()), 1)}
