#!/usr/bin/env python3
"""
Precedent transaction search / ranking engine for the `Precedent Sheet (2010-)`.

Given a newly announced deal, this scores every historical precedent in the
sheet on how good a comparable it is and returns a ranked shortlist with a
plain-English reason for each match.

Design (transparent weighted scoring, not a black box):
  Priority 1  Industry  -- Sector -> Group -> Subgroup, hierarchical.
              The ONLY hard veto: a "completely unrelated" industry
              (different, non-adjacent sector) is dropped entirely.
  Priority 2  Payment / consideration -- cash | stock | mix | tender.
  Priority 3  Strategic vs Sponsor (financial buyer) acquisition.
  Priority 3a/3b  Size -- 3a = size tier (mid/large/mega), 3b = continuous
              log-distance on deal value (coarse tier + fine gradient).
  Always      Same buyer -- if the acquirer has prior precedents in the sheet
              they are ALWAYS listed and pinned to the top (bypasses the veto).
  Modifier    Cross-border (ex-Canada) -- US<->Canada counts as domestic.
  Sparingly   EV/EBITDA proximity, recency, completed-vs-terminated.

Nothing except the industry veto is a hard filter; everything else is a soft,
tunable weight. All weights and mappings live in the CONFIG block below.

Usage
-----
  # Backtest / template off an existing deal in the sheet:
  python precedent_search.py --like "Whole Foods Market Inc" --top 15

  # Score a brand-new announced deal from scratch:
  python precedent_search.py \
      --sector "Health Care" --group "Health Care" --subgroup "Biotech & Pharma" \
      --payment cash --buyer-type strategic --size 8500 \
      --acquirer "Pfizer Inc" --target-country "United States" \
      --acquirer-country "United States" --top 20 --excel out.xlsx

Run `python precedent_search.py --help` for all options.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  CONFIG  -- everything tunable lives here                                    #
# --------------------------------------------------------------------------- #
DEFAULT_SHEET = "Precedent_Sheet_2010.xlsx"

WEIGHTS = {
    # Priority 1: industry (single best-level score, max = subgroup value)
    "industry_subgroup": 40.0,   # exact subgroup match
    "industry_group": 28.0,      # same group, different subgroup
    "industry_sector": 18.0,     # same sector, different group
    "industry_adjacent": 8.0,    # adjacent sector (see ADJACENT_SECTORS)
    # (a non-adjacent, different sector is VETOED -> row dropped)
    # Priority 2: payment / consideration
    "payment": 20.0,
    # Priority 3: strategic vs sponsor
    "buyer_type": 14.0,
    # Priority 3a / 3b: size
    "size_tier": 8.0,            # 3a  coarse tier proximity
    "size_cont": 8.0,            # 3b  continuous log-distance
    # Modifier: cross-border (ex-Canada)
    "xborder": 6.0,
    # "the rest", used sparingly
    "ev_ebitda": 2.0,
    "recency": 2.0,
    "status_completed": 1.0,
}

# Same-buyer deals are ALWAYS listed and floated to the very top.
SAME_BUYER_PIN = 10_000.0

# Sectors that are close enough NOT to trigger the "completely unrelated" veto.
# Same-sector never vetoes; these pairs are the allowed cross-sector neighbours.
ADJACENT_SECTORS = {
    "Technology": {"Communications"},
    "Communications": {"Technology"},
    "Consumer Discretionary": {"Consumer Staples"},
    "Consumer Staples": {"Consumer Discretionary"},
    "Energy": {"Utilities", "Materials"},
    "Utilities": {"Energy"},
    "Materials": {"Industrials", "Energy"},
    "Industrials": {"Materials"},
    "Financials": {"Real Estate"},
    "Real Estate": {"Financials"},
    "Health Care": set(),
}

# Size tiers (Announced Total Value, USD mil). The sheet is a >=$500M universe.
SIZE_TIERS = [
    ("Small",  0,        500),
    ("Mid",    500,      2_000),
    ("Large",  2_000,    10_000),
    ("Mega",   10_000,   float("inf")),
]

# Deal-Attribute tokens that mark a financial-sponsor (not strategic) buyer.
SPONSOR_TOKENS = {
    "PE Buyout", "Private Equity", "Management Buyout", "Venture Capital",
    "Leveraged Buyout", "Financing Round",
}
# Tokens marking a tender-offer structure.
TENDER_TOKENS = {"Tender Offer", "Dutch Auction/Self Tender"}

# Countries treated as one "domestic" bloc for cross-border (Canada does NOT
# count as cross-border per the brief).
DOMESTIC_BLOC = {"united states", "canada"}

# Generic / placeholder acquirer names that are NOT real repeat buyers and must
# be excluded from the "same buyer" always-list rule.
GENERIC_ACQUIRERS = {
    "shareholders", "management", "creditors", "investors", "private investor",
    "potential buyer", "undisclosed", "n a", "na", "employees", "consortium",
    "existing shareholders", "unknown buyer", "private company",
    "private group", "individual investor", "group of investors", "insiders",
}

RELEVANCE_BANDS = [(0.80, "Strong"), (0.60, "Good"), (0.40, "Moderate")]


# --------------------------------------------------------------------------- #
#  Normalisation helpers                                                       #
# --------------------------------------------------------------------------- #
def _clean_industry(val: str) -> str:
    """Collapse messy multi-value industry strings (e.g. 'Materials, Materials',
    'Energy, Energy, Energy') to a single primary label."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    parts = [p.strip() for p in str(val).split(",") if p.strip()]
    if not parts:
        return ""
    # If every token is identical, use it; otherwise take the first (primary).
    return parts[0]


def norm_name(name) -> str:
    """Normalise a company name for equality matching (strip suffixes/punct)."""
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    n = str(name).lower()
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    suffixes = [
        " incorporated", " inc", " corporation", " corp", " company", " co",
        " ltd", " limited", " llc", " lp", " lllp", " plc", " group",
        " holdings", " holding", " sa", " ag", " nv", " se", " spa", " oyj",
        " ab", " asa", " bv", " gmbh", " kk", " pte", " the",
    ]
    # Apply repeatedly so trailing stacks ("... Group Holdings Inc") collapse.
    changed = True
    while changed:
        changed = False
        for s in suffixes:
            if n.endswith(s):
                n = n[: -len(s)]
                changed = True
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _to_num(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def size_tier(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    for name, lo, hi in SIZE_TIERS:
        if lo <= value < hi:
            return name
    return ""


def _tier_index(tier: str) -> int:
    for i, (name, _lo, _hi) in enumerate(SIZE_TIERS):
        if name == tier:
            return i
    return -1


def consideration_category(payment_type: str) -> str:
    """Map raw Payment Type -> {cash, stock, mix, other}."""
    if payment_type is None or (isinstance(payment_type, float) and math.isnan(payment_type)):
        return "other"
    p = str(payment_type).strip().lower()
    if not p or p in ("undisclosed", "nan"):
        return "other"
    has_cash, has_stock = "cash" in p, "stock" in p
    if has_cash and has_stock:
        return "mix"        # 'Cash and Stock' / 'Cash or Stock'
    if has_stock:
        return "stock"
    if has_cash:
        return "cash"       # 'Cash', 'Cash and Debt'
    return "other"          # 'Debt', etc.


def _attr_tokens(attr: str) -> set[str]:
    return {t.strip() for t in str(attr or "").split(",") if t.strip()}


# --------------------------------------------------------------------------- #
#  Corpus feature engineering                                                  #
# --------------------------------------------------------------------------- #
def load_corpus(path: str = DEFAULT_SHEET) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    df = df.rename(columns=lambda c: str(c).strip())

    df["_sector"] = df["Target Industry Sector"].map(_clean_industry)
    df["_group"] = df["Target Industry Group"].map(_clean_industry)
    df["_subgroup"] = df["Target Industry Subgroup"].map(_clean_industry)

    df["_tv"] = _to_num(df["Announced Total Value (mil.)"])
    df["_ev"] = _to_num(df.get("Announced Equity Value (mil.)"))
    df["_tv_ebitda"] = _to_num(df["TV/EBITDA"])
    df["_tier"] = df["_tv"].map(size_tier)

    df["_payment_cat"] = df["Payment Type"].map(consideration_category)
    toks = df["Deal Attributes"].map(_attr_tokens)
    df["_is_tender"] = toks.map(lambda s: bool(s & TENDER_TOKENS))
    df["_is_sponsor"] = toks.map(lambda s: bool(s & SPONSOR_TOKENS))
    df["_buyer_type"] = np.where(df["_is_sponsor"], "sponsor", "strategic")

    tgt = df["Target Country/Region"].astype(str).str.strip().str.lower()
    acq = df["Acquirer Country/Region"].astype(str).str.strip().str.lower()
    df["_acq_country_known"] = acq.isin(
        [c for c in acq.unique() if c not in ("nan", "", "n.a.")]
    ) & ~acq.isin(["nan", "", "n.a."])
    # cross-border (ex-Canada): domestic iff both sides inside the bloc.
    domestic = tgt.isin(DOMESTIC_BLOC) & acq.isin(DOMESTIC_BLOC)
    df["_xborder"] = (~domestic).astype(object)
    # unknown acquirer country -> treat as neutral (handled in scoring)
    df.loc[~df["_acq_country_known"], "_xborder"] = np.nan

    df["_acq_norm"] = df["Acquirer Name"].map(norm_name)
    df["_acq_compact"] = df["_acq_norm"].str.replace(" ", "", regex=False)
    df["_acq_generic"] = df["_acq_norm"].isin(GENERIC_ACQUIRERS)

    df["_announce"] = pd.to_datetime(df["Announce Date"], errors="coerce")
    df["_completed"] = df["Deal Status"].astype(str).str.strip().str.lower().eq(
        "completed"
    )
    return df


# --------------------------------------------------------------------------- #
#  Query                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class DealQuery:
    sector: str = ""
    group: str = ""
    subgroup: str = ""
    payment: str = ""            # cash | stock | mix | tender
    buyer_type: str = ""         # strategic | sponsor
    size: float | None = None    # Announced Total Value, USD mil
    acquirer: str = ""
    target_country: str = "United States"
    acquirer_country: str = ""
    ev_ebitda: float | None = None
    announce_date: datetime | None = None
    xborder: bool | None = None  # override; else derived from countries

    # normalised, filled in __post_init__
    _payment_cat: str = field(default="", init=False)
    _tender_pref: bool = field(default=False, init=False)
    _acq_norm: str = field(default="", init=False)
    _acq_compact: str = field(default="", init=False)
    _xborder: bool | None = field(default=None, init=False)

    def __post_init__(self):
        # Treat NaN numerics as "unspecified".
        if self.size is not None and (isinstance(self.size, float) and math.isnan(self.size)):
            self.size = None
        if self.ev_ebitda is not None and (
            isinstance(self.ev_ebitda, float) and math.isnan(self.ev_ebitda)
        ):
            self.ev_ebitda = None
        pay = (self.payment or "").strip().lower()
        if pay == "tender":
            self._payment_cat, self._tender_pref = "cash", True
        else:
            self._payment_cat = pay if pay in ("cash", "stock", "mix") else ""
        self._acq_norm = norm_name(self.acquirer)
        self._acq_compact = self._acq_norm.replace(" ", "")
        if self.xborder is not None:
            self._xborder = bool(self.xborder)
        elif self.acquirer_country:
            tgt = self.target_country.strip().lower()
            acq = self.acquirer_country.strip().lower()
            self._xborder = not (tgt in DOMESTIC_BLOC and acq in DOMESTIC_BLOC)
        else:
            self._xborder = None

    @classmethod
    def from_row(cls, row: pd.Series) -> "DealQuery":
        """Build a query template from an existing corpus row (backtest mode)."""
        pay_cat = row["_payment_cat"]
        pay = pay_cat
        if row["_is_tender"] and pay_cat == "cash":
            pay = "tender"
        return cls(
            sector=row["_sector"], group=row["_group"], subgroup=row["_subgroup"],
            payment=pay, buyer_type=row["_buyer_type"], size=row["_tv"],
            acquirer=row["Acquirer Name"],
            target_country=str(row["Target Country/Region"]),
            acquirer_country=str(row["Acquirer Country/Region"]),
            ev_ebitda=row["_tv_ebitda"] if pd.notna(row["_tv_ebitda"]) else None,
            announce_date=row["_announce"] if pd.notna(row["_announce"]) else None,
        )


# --------------------------------------------------------------------------- #
#  Industry match level + veto                                                 #
# --------------------------------------------------------------------------- #
def industry_level(q: DealQuery, row: pd.Series) -> str:
    """Return one of: subgroup, group, sector, adjacent, unrelated."""
    if not q.sector:
        return "unknown"
    if q.subgroup and row["_subgroup"] and q.subgroup == row["_subgroup"]:
        return "subgroup"
    if q.group and row["_group"] and q.group == row["_group"]:
        return "group"
    if q.sector == row["_sector"]:
        return "sector"
    if row["_sector"] in ADJACENT_SECTORS.get(q.sector, set()):
        return "adjacent"
    return "unrelated"


# --------------------------------------------------------------------------- #
#  Scoring                                                                     #
# --------------------------------------------------------------------------- #
def _payment_score(q: DealQuery, row: pd.Series) -> tuple[float, str]:
    if not q._payment_cat:
        return 0.0, ""
    w = WEIGHTS["payment"]
    cat, is_tender = row["_payment_cat"], bool(row["_is_tender"])
    # Tender-specific query: reward tender structure (usually cash).
    if q._tender_pref:
        if is_tender and cat == "cash":
            return w, "Cash-tender match"
        if is_tender:
            return 0.85 * w, "Tender match"
        if cat == "cash":
            return 0.55 * w, "Cash (no tender)"
        return 0.1 * w, ""
    # Category query (cash / stock / mix)
    if cat == q._payment_cat:
        base, label = w, {"cash": "Cash", "stock": "Stock",
                          "mix": "Cash & stock"}[cat] + " match"
    elif "mix" in (cat, q._payment_cat) and cat in ("cash", "stock", "mix"):
        base, label = 0.5 * w, "Partial payment match"
    else:
        base, label = 0.0, ""
    if is_tender:  # small agreement bonus if new deal is also a tender
        base = min(w, base + 0.05 * w)
    return base, label


def _size_scores(q: DealQuery, row: pd.Series) -> tuple[float, float, str]:
    if q.size is None or q.size <= 0 or pd.isna(row["_tv"]) or row["_tv"] <= 0:
        return 0.0, 0.0, ""
    # 3a: tier proximity
    qi, ri = _tier_index(size_tier(q.size)), _tier_index(row["_tier"])
    tier_w = WEIGHTS["size_tier"]
    if qi < 0 or ri < 0:
        s_tier = 0.0
    else:
        gap = abs(qi - ri)
        s_tier = {0: 1.0, 1: 0.5, 2: 0.15}.get(gap, 0.0) * tier_w
    # 3b: continuous log-distance
    dlog = abs(math.log10(q.size) - math.log10(row["_tv"]))
    s_cont = math.exp(-dlog / 0.5) * WEIGHTS["size_cont"]
    label = ""
    if qi >= 0 and qi == ri:
        label = f"Same size tier ({row['_tier']})"
    elif dlog <= 0.3:
        label = "Similar size"
    return s_tier, s_cont, label


def score_row(q: DealQuery, row: pd.Series) -> dict:
    reasons: list[str] = []
    score = 0.0
    max_score = 0.0

    # --- same buyer (always list + pin) -----------------------------------
    # Exact normalised match, plus a space-insensitive compact fallback so
    # "ExxonMobil" matches "Exxon Mobil Corp". Generic placeholders excluded.
    same_buyer = bool(
        q._acq_norm
        and not row["_acq_generic"]
        and (
            q._acq_norm == row["_acq_norm"]
            or (len(q._acq_compact) >= 4 and q._acq_compact == row["_acq_compact"])
        )
    )

    # --- industry (priority 1) --------------------------------------------
    lvl = industry_level(q, row)
    ind_w = {
        "subgroup": WEIGHTS["industry_subgroup"],
        "group": WEIGHTS["industry_group"],
        "sector": WEIGHTS["industry_sector"],
        "adjacent": WEIGHTS["industry_adjacent"],
        "unrelated": 0.0,
        "unknown": 0.0,
    }[lvl]
    if lvl != "unknown":
        max_score += WEIGHTS["industry_subgroup"]
        score += ind_w
    ind_label = {
        "subgroup": f"Same subgroup ({row['_subgroup']})",
        "group": f"Same group ({row['_group']})",
        "sector": f"Same sector ({row['_sector']})",
        "adjacent": f"Adjacent sector ({row['_sector']})",
        "unrelated": "", "unknown": "",
    }[lvl]
    if ind_label:
        reasons.append(ind_label)

    # --- payment (priority 2) ---------------------------------------------
    if q._payment_cat:
        max_score += WEIGHTS["payment"]
        s, lab = _payment_score(q, row)
        score += s
        if lab:
            reasons.append(lab)

    # --- strategic / sponsor (priority 3) ---------------------------------
    if q.buyer_type:
        max_score += WEIGHTS["buyer_type"]
        if row["_buyer_type"] == q.buyer_type.strip().lower():
            score += WEIGHTS["buyer_type"]
            reasons.append(
                "Sponsor buyer" if q.buyer_type == "sponsor" else "Strategic buyer"
            )

    # --- size 3a / 3b -----------------------------------------------------
    if q.size is not None:
        max_score += WEIGHTS["size_tier"] + WEIGHTS["size_cont"]
        st, sc, lab = _size_scores(q, row)
        score += st + sc
        if lab:
            reasons.append(lab)

    # --- cross-border ex-Canada (modifier) --------------------------------
    if q._xborder is not None:
        max_score += WEIGHTS["xborder"]
        rx = row["_xborder"]
        if pd.notna(rx):
            if bool(rx) == q._xborder:
                score += WEIGHTS["xborder"]
                reasons.append(
                    "Cross-border match" if q._xborder else "Domestic (ex-CA) match"
                )
        else:
            score += 0.5 * WEIGHTS["xborder"]  # unknown country -> neutral half

    # --- the rest, sparingly ----------------------------------------------
    if q.ev_ebitda is not None and pd.notna(row["_tv_ebitda"]):
        max_score += WEIGHTS["ev_ebitda"]
        d = abs(q.ev_ebitda - row["_tv_ebitda"])
        score += math.exp(-d / 8.0) * WEIGHTS["ev_ebitda"]

    ref_date = q.announce_date or datetime.now()
    if pd.notna(row["_announce"]):
        max_score += WEIGHTS["recency"]
        yrs = abs((ref_date - row["_announce"]).days) / 365.25
        score += math.exp(-yrs / 6.0) * WEIGHTS["recency"]

    max_score += WEIGHTS["status_completed"]
    if row["_completed"]:
        score += WEIGHTS["status_completed"]

    match_pct = (score / max_score) if max_score > 0 else 0.0
    relevance = "Weak"
    for thr, lbl in RELEVANCE_BANDS:
        if match_pct >= thr:
            relevance = lbl
            break

    if same_buyer:
        reasons.insert(0, "** SAME BUYER (prior precedent) **")

    return {
        "score": score,
        "match_pct": match_pct,
        "relevance": relevance,
        "industry_level": lvl,
        "same_buyer": same_buyer,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- #
#  Search                                                                      #
# --------------------------------------------------------------------------- #
OUTPUT_COLS = [
    "Rank", "Same Buyer", "Relevance", "Match %", "Score",
    "Announce Date", "Target Name", "Acquirer Name",
    "Announced Total Value (mil.)", "Payment Type", "TV/EBITDA",
    "Deal Status", "Target Industry Sector", "Target Industry Group",
    "Target Industry Subgroup", "Deal Attributes",
    "Target Country/Region", "Acquirer Country/Region", "Why it matched",
]


def _decorate(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy().reset_index(drop=True)
    frame.insert(0, "Rank", range(1, len(frame) + 1))
    frame["Same Buyer"] = np.where(frame["same_buyer"], "YES", "")
    frame["Relevance"] = frame["relevance"]
    frame["Match %"] = (frame["match_pct"] * 100).round(1)
    frame["Score"] = frame["score"].round(1)
    frame["Why it matched"] = frame["reasons"].map(lambda r: "; ".join(r))
    return frame


def search(
    query: DealQuery,
    df: pd.DataFrame,
    top: int = 20,
    max_buyer: int | None = None,
    include_unrelated: bool = False,
    exclude_self: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (buyer_deals, comps).

    buyer_deals -- every prior precedent by the same acquirer (always listed,
                   optionally capped at `max_buyer`), ranked by score.
    comps       -- the top `top` independent market precedents (different buyer),
                   ranked by score, after the industry veto.
    """
    results = df.apply(lambda r: score_row(query, r), axis=1, result_type="expand")
    out = df.copy()
    for col in ("score", "match_pct", "relevance", "industry_level",
                "same_buyer", "reasons"):
        out[col] = results[col]

    # Drop the exact deal we templated from (same acquirer + subgroup + size).
    if exclude_self and query._acq_norm and query.subgroup and query.size:
        self_mask = (
            (out["_acq_norm"] == query._acq_norm)
            & (out["_subgroup"] == query.subgroup)
            & (out["_tv"].round(1) == round(query.size, 1))
        )
        out = out[~self_mask]

    buyer_deals = out[out["same_buyer"]].sort_values("score", ascending=False)
    if max_buyer is not None:
        buyer_deals = buyer_deals.head(max_buyer)

    comps = out[~out["same_buyer"]]
    # Hard veto for comps only: completely unrelated industry is dropped.
    if not include_unrelated:
        comps = comps[comps["industry_level"] != "unrelated"]
    comps = comps.sort_values("score", ascending=False).head(top)

    return _decorate(buyer_deals), _decorate(comps)


# --------------------------------------------------------------------------- #
#  Presentation                                                                #
# --------------------------------------------------------------------------- #
def _fmt_query(q: DealQuery) -> str:
    bits = [
        f"Industry : {q.sector} > {q.group} > {q.subgroup}".rstrip(" >"),
        f"Payment  : {q.payment or '-'}",
        f"Buyer    : {q.buyer_type or '-'}",
        f"Size     : {('$%.0fM' % q.size) if q.size else '-'}"
        f"  (tier {size_tier(q.size) or '-'})",
        f"Acquirer : {q.acquirer or '-'}",
        f"X-border : {q._xborder if q._xborder is not None else '-'}"
        " (US<->CA = domestic)",
    ]
    return "\n".join("  " + b for b in bits)


def _print_rows(frame: pd.DataFrame) -> None:
    for _, r in frame.iterrows():
        flag = "★" if r["Same Buyer"] == "YES" else " "
        print(f"{flag}{r['Rank']:>3}. {r['Match %']:>5.1f}%  {r['Relevance']:<8} "
              f"{str(r['Announce Date'])[:10]}  "
              f"{str(r['Target Name'])[:34]:<34} <- {str(r['Acquirer Name'])[:26]:<26} "
              f"${r['Announced Total Value (mil.)']:>8,.0f}M")
        print(f"       {r['Why it matched']}")


def print_results(q: DealQuery, buyer: pd.DataFrame, comps: pd.DataFrame) -> None:
    print("=" * 100)
    print("NEW DEAL QUERY")
    print(_fmt_query(q))
    print("=" * 100)
    if not buyer.empty:
        print(f"PRIOR DEALS BY THIS BUYER  ({len(buyer)}) -- always listed")
        print("-" * 100)
        _print_rows(buyer)
        print("-" * 100)
    print(f"CLOSEST MARKET PRECEDENTS  (top {len(comps)}, independent buyers)")
    print("-" * 100)
    if comps.empty:
        print("No precedents survived the industry veto. "
              "Re-run with --include-unrelated to widen the net.")
    else:
        _print_rows(comps)
    print("-" * 100)


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rank M&A precedents for a newly announced deal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sheet", default=DEFAULT_SHEET, help="Precedent workbook.")
    p.add_argument("--like", help="Use an existing Target Name in the sheet as "
                                  "the query template (backtest mode).")
    p.add_argument("--sector")
    p.add_argument("--group")
    p.add_argument("--subgroup")
    p.add_argument("--payment", choices=["cash", "stock", "mix", "tender"])
    p.add_argument("--buyer-type", choices=["strategic", "sponsor"])
    p.add_argument("--size", type=float, help="Announced Total Value, USD mil.")
    p.add_argument("--acquirer", help="Acquirer name (for same-buyer rule).")
    p.add_argument("--target-country", default="United States")
    p.add_argument("--acquirer-country", default="")
    p.add_argument("--ev-ebitda", type=float)
    p.add_argument("--xborder", choices=["yes", "no"],
                   help="Force cross-border flag (else derived from countries).")
    p.add_argument("--top", type=int, default=20,
                   help="Number of independent market comps to return.")
    p.add_argument("--max-buyer", type=int, default=None,
                   help="Cap on prior same-buyer deals listed (default: all).")
    p.add_argument("--include-unrelated", action="store_true",
                   help="Disable the completely-unrelated-industry veto.")
    p.add_argument("--excel", help="Write ranked results to this .xlsx.")
    return p


def query_from_args(args, df: pd.DataFrame) -> DealQuery:
    if args.like:
        mask = df["Target Name"].astype(str).str.strip().str.lower() == \
            args.like.strip().lower()
        if not mask.any():
            mask = df["Target Name"].astype(str).str.contains(
                re.escape(args.like), case=False, na=False
            )
        if not mask.any():
            sys.exit(f"--like: no target matching '{args.like}' in the sheet.")
        row = df[mask].iloc[0]
        print(f"[template] {row['Target Name']} <- {row['Acquirer Name']} "
              f"({str(row['_announce'])[:10]})")
        return DealQuery.from_row(row)

    xb = None if args.xborder is None else (args.xborder == "yes")
    return DealQuery(
        sector=args.sector or "", group=args.group or "",
        subgroup=args.subgroup or "", payment=args.payment or "",
        buyer_type=args.buyer_type or "", size=args.size,
        acquirer=args.acquirer or "",
        target_country=args.target_country,
        acquirer_country=args.acquirer_country,
        ev_ebitda=args.ev_ebitda, xborder=xb,
    )


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    df = load_corpus(args.sheet)
    q = query_from_args(args, df)
    if not q.sector:
        sys.exit("A --sector (or --like) is required to run a search.")
    buyer, comps = search(
        q, df, top=args.top, max_buyer=args.max_buyer,
        include_unrelated=args.include_unrelated,
    )
    print_results(q, buyer, comps)
    if args.excel:
        with pd.ExcelWriter(args.excel) as xw:
            comps[OUTPUT_COLS].to_excel(xw, sheet_name="Market Comps", index=False)
            if not buyer.empty:
                buyer[OUTPUT_COLS].to_excel(
                    xw, sheet_name="Prior Buyer Deals", index=False
                )
        print(f"\nWrote results -> {args.excel} "
              f"({len(comps)} comps, {len(buyer)} prior-buyer deals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
