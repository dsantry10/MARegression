# MARegression — M&A Deal Completion-Time Models

Predicts **`Business Days To Complete`** for announced M&A deals. Two production models,
one shared leakage-free preprocessing, one scorer that runs both.

**Start here → [`MODELS.md`](MODELS.md)** for the full guide (quick start, retraining,
model rationale, caveats).

```bash
pip install -r requirements.txt
python score_deals.py path/to/deals.xlsx        # scores every deal with BOTH models
```

## The one dataset that matters for modeling

**`LARGE_DATASET (TOGGLES).xlsx`** — 1,170 deals announced **2017–2026**, target column
`Business Days To Complete`, including the SAMR / EC / CFIUS regulatory toggles.
Every current model trains from this file:

| Model | Rows used | Algorithm |
|---|---|---|
| Full-history two-stage ensemble (short/long regime experts) | all 1,170 (2017–2026) | XGBoost-led blended mixture-of-experts |
| Recent-regime model | 155 (announced ≥2025) | RandomForest + ExtraTrees blend |

`LARGE_DATASET_2025plus.xlsx` is a generated convenience export of the ≥2025 rows — never
edit it by hand; regenerate it from the master file.

⚠️ **Data integrity note:** the SAMR/EC/CFIUS columns were originally Excel formulas. A
header edit via openpyxl once silently dropped their computed values (caught and fixed —
values are now static). If the master file is ever re-exported from a formula workbook,
verify the toggles are materialized values, e.g.
`pd.read_excel(...)[['SAMR','EC','CFIUS']].notna().all()`.

## Repository layout

```
score_deals.py                       # run BOTH models on any deal file
MODELS.md                            # full documentation
requirements.txt                     # pinned environment
LARGE_DATASET (TOGGLES).xlsx         # MASTER dataset (2017–2026)
xgboost_ma_completion_model.py       # shared feature schema + base XGBoost
ma_completion_twostage_ensemble.py   # full-history two-stage model
ma_2025plus_model.py                 # 2025+ model
*.pkl / *.json / *.png               # fitted artifacts + diagnostics
legacy/                              # prior 2025-present OLS/RF project (different
                                     # datasets) — reference only, not used by models
```

## Branches

- **Default branch** — canonical; everything above lives here.
- `claude/stacked-ensemble` — stacking experiment (documented negative result: stacking
  did not beat XGBoost alone).
- `claude/precedent-search-algo-*` — separate precedent-transaction search work.
- `claude/xgboost-ma-completion-model-tbz5lj` — early standalone XGBoost attempt, kept
  for reference only; superseded by the models above.
