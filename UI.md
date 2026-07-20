# Deal-Timing UI

A Streamlit app that runs the two-stage XGBoost model (short/long regime experts) on a
single deal via a streamlined form, and surfaces the critical takeaways.

## Run

```bash
pip install -r requirements.txt -r requirements-ui.txt
streamlit run app.py
```

Opens a local web app (default http://localhost:8501). It is a **local/hosted server
app**, not a static page — it must run Python to execute the pickled model.

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
