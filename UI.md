# Deal-Timing UI

A Streamlit app that runs the two-stage XGBoost model (short/long regime experts) on a
single deal via a streamlined form, and surfaces the critical takeaways.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens a local web app (default http://localhost:8501). It is a **server app**, not a
static page — it must run Python to execute the pickled model.

## Host it online (no local install) — Streamlit Community Cloud

1. Go to **share.streamlit.io** and sign in with your **GitHub** account.
2. Click **Create app → Deploy a public app from GitHub**.
3. Repository: `dsantry10/maregression` · Branch: `claude/model-ui` · Main file: `app.py`.
4. Open **Advanced settings** and set **Python version = 3.11** (so the pinned wheels
   resolve).
5. Click **Deploy**. First build takes a few minutes; afterward you get a permanent URL to
   bookmark.

`requirements.txt` is intentionally lean (streamlit, pandas, numpy, scikit-learn, xgboost,
openpyxl) so the deploy is fast and reliable. Retraining/plot extras (`shap`, `matplotlib`)
live in `requirements-dev.txt` and are **not** needed to run the app.

## Inputs (streamlined)

Grouped in the sidebar: **Basics** (announce date, names, total/equity value, revenue),
**Terms** (payment type, premium, TV/EBITDA, margin), **Classification** (industry group,
countries, nature of bid), **Toggles** (going-private, PE buyout, tender offer,
additional-stake, competing-bid), **Regulatory** (SAMR/EC/CFIUS).

Conveniences:
- `Log TV / Revenue / Equity` are derived automatically from the dollar inputs.
- Cross-border is auto-set from a target/acquirer country mismatch.
- Numeric fields left at 0 are treated as **unknown** (the model handles missing values
  natively — no imputation).
- Industry labels are normalized to the training vocabulary (e.g. `Real Estate REIT` →
  `Real Estate`) and any unknown category raises a visible warning.

## Model selector

The sidebar has a **Model** switch offering two *separate* trained models:

- **Standard** — payment types as reported (`ma_twostage_ensemble.pkl`).
- **Stock-consolidated** (default on this branch) — any stock-containing consideration
  (Cash and Stock, Cash or Stock, Stock) collapsed to a single `Stock` category
  (`ma_twostage_stockpay_ensemble.pkl`).

They are distinct artifacts and never conflated; the active model is labeled above the
result. The Payment-type dropdown reflects the chosen model's vocabulary (4 options for
stock-consolidated, 6 for standard).

## Outputs (critical takeaways)

- **Weighted close expectation** — business days, calendar days, months, and an estimated
  close date (business-day add from the announce date).
- **Regime experts** — the short-regime and long-regime expert estimates, plus P(long)
  with a plain-English routing interpretation and a regulatory-floor badge.
- **Confidence cross-check** — the 2025+ model's estimate beside the main one (agreement =
  higher confidence; divergence is flagged).
- **Historical comparators** — median/mean close time of the nearest deals by profile.

## Architecture

`app.py` (Streamlit view) -> `ui_model.py` (input mapping + prediction) -> the existing
`build_features` + pickled `ma_twostage_ensemble.pkl`. The UI never re-implements the
model; it calls the same artifacts the CLI and `score_deals.py` use, so numbers match
exactly.
