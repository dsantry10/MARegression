"""
UI model layer: streamlined-inputs -> deal dict -> two-stage XGBoost prediction.

Thin wrapper over the existing model so the UI never re-implements anything:
  * derives Log TV / Log Revenue / Log Equity from the dollar inputs,
  * normalizes common category labels (fixes silent-mislabel skew, e.g. the
    "Real Estate REIT" export label) and reports unknown categories,
  * returns the two-stage experts (short / long), P(long), the weighted blend,
    whether the SAMR+EC regulatory floor fired, and the 2025+ cross-check.
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

FULL_MODEL_PATH = "ma_twostage_ensemble.pkl"
FULL_PREP_PATH = "twostage_preprocessor.pkl"
RECENT_MODEL_PATH = "ma_2025plus_model.pkl"
RECENT_PREP_PATH = "ma_2025plus_preprocessor.pkl"
DATASET_PATH = "LARGE_DATASET (TOGGLES).xlsx"

# Common export labels -> the category the model was trained on.
INDUSTRY_ALIASES = {
    "real estate reit": "Real Estate",
    "reit": "Real Estate",
    "real estate investment trust": "Real Estate",
}


@functools.lru_cache(maxsize=1)
def _load():
    with open(FULL_PREP_PATH, "rb") as fh:
        full_prep = pickle.load(fh)
    with open(FULL_MODEL_PATH, "rb") as fh:
        full_model = pickle.load(fh)
    with open(RECENT_PREP_PATH, "rb") as fh:
        recent_prep = pickle.load(fh)
    with open(RECENT_MODEL_PATH, "rb") as fh:
        recent_model = pickle.load(fh)
    return full_model, full_prep, recent_model, recent_prep


def known_categories(field: str) -> list[str]:
    """Sorted list of category values the model recognizes for a categorical field."""
    _, full_prep, _, _ = _load()
    return sorted(full_prep["categories"][field].categories.tolist())


def _log10_or_nan(value) -> float:
    try:
        v = float(value)
        return float(np.log10(v)) if v and v > 0 else np.nan
    except (TypeError, ValueError):
        return np.nan


def normalize_industry(value: str):
    """Return (normalized_value, warning_or_None) for the target industry group."""
    if value is None:
        return value, None
    known = set(known_categories("Target Industry Group"))
    if value in known:
        return value, None
    alias = INDUSTRY_ALIASES.get(str(value).strip().lower())
    if alias and alias in known:
        return alias, f"Industry '{value}' mapped to '{alias}' (training-vocabulary label)."
    return value, (f"Industry '{value}' is not in the model's vocabulary; it will be "
                   f"treated as missing and weaken the estimate.")


def build_deal_dict(inp: dict) -> tuple[dict, list[str]]:
    """Map streamlined UI inputs to a raw deal dict; return (deal, warnings)."""
    warnings: list[str] = []
    industry, warn = normalize_industry(inp.get("industry_group"))
    if warn:
        warnings.append(warn)

    total_value = inp.get("total_value")
    equity_value = inp.get("equity_value")
    revenue = inp.get("revenue")

    # Auto-suggest cross-border from a country mismatch unless the user overrode it.
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


def predict(deal: dict) -> dict:
    """Run the two-stage model + 2025+ cross-check. Returns a results dict."""
    full_model, full_prep, recent_model, recent_prep = _load()
    df = pd.DataFrame([deal])

    X = build_features(df, categories=full_prep["categories"])[full_prep["feature_columns"]]
    for c in full_prep["categorical_features"]:
        X[c] = X[c].astype(full_prep["categories"][c])

    p_long = float(full_model._p_long(X)[0])
    short = float(full_model._predict_expert(full_model.expert_short, X)[0])
    long = float(full_model._predict_expert(full_model.expert_long, X)[0])
    weighted = float(full_model.predict(X)[0])

    # Did the SAMR+EC regulatory floor fire (with PE-sponsor exception)?
    floor_applied = (
        int(deal.get("SAMR", 0)) == 1
        and int(deal.get("EC", 0)) == 1
        and deal.get("PE Buyout") != "Yes"
    )

    # 2025+ cross-check.
    Xr = build_features(df)[recent_prep["feature_columns"]]
    recent = float(recent_model.predict(Xr)[0])

    return {
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
    """Add business_days to announce_date, returning a pandas Timestamp."""
    try:
        start = pd.Timestamp(announce_date)
        return start + pd.tseries.offsets.BusinessDay(int(round(business_days)))
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _dataset():
    return pd.read_excel(DATASET_PATH, sheet_name="Sheet1")


def comparators(deal: dict) -> dict:
    """Median/mean close time of nearest historical deals, tightening the filter."""
    d = _dataset()
    target = "Business Days To Complete"
    y = d[target]
    ig = deal.get("Target Industry Group")

    cuts = []
    base = pd.Series(True, index=d.index)
    if ig in set(d["Target Industry Group"]):
        base = d["Target Industry Group"] == ig
        cuts.append((f"Industry = {ig}", base))
    if deal.get("Payment Type") in set(d["Payment Type"].dropna()):
        m = base & (d["Payment Type"] == deal["Payment Type"])
        if m.sum() >= 5:
            cuts.append((f"+ {deal['Payment Type']}", m))
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
