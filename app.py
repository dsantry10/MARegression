"""
Deal-timing UI -- two-stage XGBoost model for `Business Days To Complete`.

Run:
    pip install -r requirements-ui.txt
    streamlit run app.py
"""
import datetime as dt

import pandas as pd
import streamlit as st

import ui_model as um

st.set_page_config(page_title="M&A Deal-Timing Model", page_icon="\U0001F4C8", layout="wide")

st.title("\U0001F4C8 M&A Deal-Timing Model")
st.caption("Two-stage XGBoost ensemble (short/long regime experts) — predicts "
           "**Business Days To Complete** for an announced deal. Trained on 1,170 deals, 2017–2026.")

# --------------------------------------------------------------------------- #
# Input form (left) -- streamlined; blanks are treated as unknown (NaN).
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Deal inputs")

    with st.expander("Basics", expanded=True):
        announce_date = st.date_input("Announce date", value=dt.date.today())
        c1, c2 = st.columns(2)
        target = c1.text_input("Target", "")
        acquirer = c2.text_input("Acquirer", "")
        total_value = st.number_input("Total value ($M)", min_value=0.0, value=1500.0, step=50.0)
        equity_value = st.number_input("Equity value ($M)", min_value=0.0, value=1500.0, step=50.0)
        revenue = st.number_input("Target LTM revenue ($M) — 0 = unknown",
                                  min_value=0.0, value=0.0, step=10.0)

    with st.expander("Terms", expanded=True):
        payment_type = st.selectbox("Payment type", um.known_categories("Payment Type"))
        premium = st.number_input("Announced premium (%)", value=20.0, step=1.0)
        tv_ebitda = st.number_input("TV / EBITDA (0 = unknown)", min_value=0.0, value=0.0, step=1.0)
        operating_margin = st.number_input("Target op. margin (%) — leave 0 if unknown",
                                           value=0.0, step=1.0)

    with st.expander("Classification", expanded=True):
        industry_group = st.selectbox("Target industry group",
                                      um.known_categories("Target Industry Group"))
        countries = um.known_categories("Target Country/Region")
        acq_countries = um.known_categories("Acquirer Country/Region")
        tci = countries.index("United States") if "United States" in countries else 0
        aci = acq_countries.index("United States") if "United States" in acq_countries else 0
        target_country = st.selectbox("Target country", countries, index=tci)
        acquirer_country = st.selectbox("Acquirer country", acq_countries, index=aci)
        nature_of_bid = st.selectbox("Nature of bid", um.known_categories("Nature of Bid"))

    with st.expander("Toggles", expanded=True):
        c1, c2 = st.columns(2)
        going_private = c1.checkbox("Going private")
        pe_buyout = c1.checkbox("PE buyout")
        tender_offer = c1.checkbox("Tender offer")
        additional_stake = c2.checkbox("Additional stake purchase")
        competing_bid = c2.checkbox("Competing bid")
        xborder_auto = "Yes" if target_country != acquirer_country else "No"
        st.caption(f"Cross-border auto-set to **{xborder_auto}** from countries.")

    with st.expander("Regulatory", expanded=True):
        c1, c2, c3 = st.columns(3)
        samr = c1.checkbox("SAMR")
        ec = c2.checkbox("EC")
        cfius = c3.checkbox("CFIUS")

    run = st.button("Predict close time", type="primary", use_container_width=True)


def _clean(v):
    return None if v in (0, 0.0) else v


inputs = dict(
    target=target, acquirer=acquirer, announce_date=str(announce_date),
    total_value=_clean(total_value), equity_value=_clean(equity_value), revenue=_clean(revenue),
    payment_type=payment_type, premium=premium, tv_ebitda=_clean(tv_ebitda),
    operating_margin=_clean(operating_margin), industry_group=industry_group,
    target_country=target_country, acquirer_country=acquirer_country, nature_of_bid=nature_of_bid,
    going_private=going_private, pe_buyout=pe_buyout, tender_offer=tender_offer,
    additional_stake=additional_stake, competing_bid=competing_bid,
    samr=samr, ec=ec, cfius=cfius, cross_border=None,
)

# --------------------------------------------------------------------------- #
# Output (right) -- critical takeaways.
# --------------------------------------------------------------------------- #
if not run:
    st.info("Enter the deal on the left and press **Predict close time**. "
            "Only a few fields are required; leave numeric fields at 0 if unknown "
            "(the model handles missing values natively).")
    st.stop()

deal, warnings = um.build_deal_dict(inputs)
res = um.predict(deal)
close = um.estimated_close_date(inputs["announce_date"], res["weighted"])

for w in warnings:
    st.warning(w)

# --- Headline ---
st.subheader("Weighted close expectation")
h1, h2, h3, h4 = st.columns(4)
h1.metric("Business days", f"{res['weighted']:.0f}")
h2.metric("Calendar days", f"{res['weighted_calendar']:.0f}")
h3.metric("Months", f"{res['weighted_months']:.1f}")
h4.metric("Est. close date", close.strftime("%b %d, %Y") if close is not None else "—")

# --- Experts + routing ---
st.subheader("Regime experts")
e1, e2, e3 = st.columns(3)
e1.metric("Short-regime expert", f"{res['short']:.0f} bd",
          help="What the model expects if this closes like a routine deal.")
e2.metric("Long-regime expert", f"{res['long']:.0f} bd",
          help="What the model expects if this becomes a slow / regulatory process.")
e3.metric("P(long regime)", f"{res['p_long']*100:.0f}%",
          help="Probability the deal takes the slow lane. The weighted headline "
               "blends the two experts by this probability.")

p = res["p_long"]
if p < 0.10:
    st.success(f"Routing: **{p*100:.0f}%** slow-lane — the model sees a clean, "
               f"routine deal; the headline tracks the short expert.")
elif p < 0.30:
    st.warning(f"Routing: **{p*100:.0f}%** slow-lane — some regulatory/complexity "
               f"risk is pulling the estimate up.")
else:
    st.error(f"Routing: **{p*100:.0f}%** slow-lane — the model expects a "
             f"drawn-out process; weight the long expert heavily.")
if res["floor_applied"]:
    st.info("SAMR + EC regulatory floor applied (P(long) floored to 20%). "
            "PE-sponsor deals are exempt from this floor.")

# --- Bar chart: short / weighted / long ---
chart = pd.DataFrame({"business days": [res["short"], res["weighted"], res["long"]]},
                     index=["Short expert", "Weighted", "Long expert"])
st.bar_chart(chart, horizontal=True)

# --- Confidence cross-check vs 2025+ model ---
st.subheader("Confidence cross-check")
diff = abs(res["weighted"] - res["recent_2025plus"])
c1, c2 = st.columns([1, 2])
c1.metric("2025+ model", f"{res['recent_2025plus']:.0f} bd",
          delta=f"{res['recent_2025plus'] - res['weighted']:+.0f} vs main")
if diff <= 15:
    c2.success(f"The full-history and 2025+ models agree within {diff:.0f} business days "
               f"— **high confidence**.")
else:
    c2.warning(f"The two models diverge by {diff:.0f} business days. The gap is often "
               f"informative (e.g. the full-history model runs longer on regulatory deals).")

# --- Comparators ---
st.subheader("Historical comparators")
comp = um.comparators(deal)
if comp["rows"]:
    df = pd.DataFrame(comp["rows"]).rename(
        columns={"filter": "Comparator set", "n": "N", "median": "Median (bd)", "mean": "Mean (bd)"})
    st.dataframe(df, hide_index=True, use_container_width=True)
    tightest = comp["rows"][-1]
    st.caption(f"Nearest set (**{tightest['filter']}**, n={tightest['n']}): "
               f"median **{tightest['median']:.0f}** bd vs model **{res['weighted']:.0f}** bd. "
               f"Overall dataset median = {comp['overall_median']:.0f} bd.")
else:
    st.caption("No close comparator subset found in the dataset.")

with st.expander("Model input echo (raw deal dict)"):
    st.json({k: v for k, v in deal.items() if v not in ("", None)})
