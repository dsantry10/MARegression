# M&A Deal Completion-Time Models

Two independent models predict **`Business Days To Complete`** for an M&A deal. Run both on
any deal to get **two clean, comparable data points**.

| Model | Trained on | Algorithm | CV R² | Best for |
|---|---|---|---|---|
| **full_history** | full 2017–2026 dataset (~1,170 deals) | Two-stage blended XGBoost ensemble (mixture-of-experts + regulatory routing) | 0.276 CV / 0.26 out-of-time | Large, complex, cross-border, or regulatory (SAMR/EC/CFIUS) deals |
| **stock_consolidated** | same, payment types with any stock collapsed to "Stock" | Same two-stage architecture, separate artifacts | 0.272 CV / 0.272 out-of-time | Same, when you want all stock/mixed consideration treated as one "Stock" category |
| **recent_2025+** | deals announced 2025 onward (155 deals) | RandomForest + ExtraTrees bagged blend | 0.242 CV | Present-day, smaller, non-regulatory deals |

Both share **one leakage-free preprocessing definition** (`build_features` in
`xgboost_ma_completion_model.py`), so their predictions are strictly comparable. The target
is in **business days** — multiply by ~1.4 for calendar days, or divide by ~21.7 for months.

---

## Quick start

```bash
pip install -r requirements.txt

# Score one or more deals (rows = deals, original Excel column names) with BOTH models:
python score_deals.py path/to/deals.xlsx
python score_deals.py path/to/deals.xlsx --out results.csv
```

Programmatic use:

```python
from score_deals import score_deals
score_deals(deal_dict)     # dict, list[dict], or DataFrame -> results DataFrame
```

Output columns: `full_history_bus_days`, `recent_2025plus_bus_days`, `spread_bus_days`,
plus calendar-day and month conversions for each.

**Reading two data points:** where the models agree, confidence is high. Where they diverge,
the gap is informative — the full-history model runs longer on regulatory/complex deals
(it has learned the SAMR/EC/CFIUS long-tail), while the 2025+ model reflects the faster,
flag-free current regime.

---

## Repository map

```
score_deals.py                      # ← unified scorer: run BOTH models on a deal
requirements.txt                    # pinned environment

# Full-history model (2017–2026)
xgboost_ma_completion_model.py      # feature schema + build_features (shared) + base XGBoost
ma_completion_twostage_ensemble.py  # two-stage blended ensemble (the production full-history model)
ma_twostage_ensemble.pkl            # fitted ensemble
twostage_preprocessor.pkl           # category maps + feature columns

# Recent model (2025+)
ma_2025plus_model.py                # bake-off + RandomForest+ExtraTrees training
ma_2025plus_model.pkl               # fitted model
ma_2025plus_preprocessor.pkl        # feature columns + metadata
LARGE_DATASET_2025plus.xlsx         # convenience truncated dataset (155 rows)

LARGE_DATASET (TOGGLES).xlsx        # full source dataset (target = Business Days To Complete)

legacy/                             # prior 2025-present OLS/RF project — reference only
```

---

## Datasets & branch governance (anti-conflation rules)

**Datasets**

| File | Coverage | Used by | Rule |
|---|---|---|---|
| `LARGE_DATASET (TOGGLES).xlsx` | 1,170 deals, **2017–2026**, incl. SAMR/EC/CFIUS toggles | **All current models** | The single master. Edit only via value-preserving tools; verify toggles non-null after any edit. |
| `LARGE_DATASET_2025plus.xlsx` | 155 deals, ≥2025 | 2025+ model convenience | Generated from the master — regenerate, never hand-edit. |
| `legacy/MA_Statistics_2025-Present__Claude_Code_.xlsx`, `legacy/New_Training_Sheet_2.xlsx` | old project | `legacy/` scripts only | Never used by current models. |

**Branches**

| Branch | Role |
|---|---|
| default branch | **Canonical.** Correct master dataset + both models + scorer + docs. |
| `claude/stacked-ensemble` | Experiment branch (stacking; documented negative result). Kept in sync with the fixed dataset. |
| `claude/precedent-search-algo-*` | Separate precedent-search workstream. |
| `claude/xgboost-ma-completion-model-tbz5lj` | Early standalone attempt, reference only — do not build on it. |

New experiments: branch off the default branch, never off another experiment branch, so
every model always trains on the canonical master dataset.

---

## Retraining (repeatable)

Both training scripts read `LARGE_DATASET (TOGGLES).xlsx`, are deterministic (fixed seeds),
and regenerate their own artifacts + diagnostics:

```bash
python ma_completion_twostage_ensemble.py   # retrains full-history model
python ma_2025plus_model.py                 # retrains 2025+ model (re-runs its own eval)
```

To refresh the truncated dataset after new data arrives, re-run the truncation step (keeps
rows with `Announce Date` year ≥ 2025) — see the top of `ma_2025plus_model.py`.

---

## Why two different algorithms?

This was decided empirically, not by assumption.

- **Full history (1,170 rows):** enough data for gradient boosting; a two-stage
  mixture-of-experts with a `P(long)` router captures the heavy regulatory long-tail
  (skew ≈ 3.3, max 1,078 days). A `SAMR+EC` routing floor (with a PE-sponsor exception)
  corrects the router's under-reaction to regulatory flags.
- **2025+ only (155 rows):** a repeated-CV bake-off across 11 families showed **bagged trees
  beat boosting** at this sample size — RandomForest+ExtraTrees (0.242) vs XGBoost (0.168) —
  because bagging averages out variance where boosting chases noise. The post-2025 target is
  also far tamer (skew 1.24, max 236), and all regulatory flags are 0, so the regulatory
  machinery is dropped as zero-variance.

## Evaluation notes / caveats

- **Primary metric is repeated K-Fold CV R².** With only 155 rows the 2025+ model cannot be
  judged on a single holdout.
- **The 2026 slice is censored** — only deals that have already closed carry a target, biasing
  it toward fast deals. Its R² is not informative and is reported only as a caveat.
- Both models are honest but modest (R² ≈ 0.24–0.28); the dataset has a real signal ceiling
  driven by a small set of unpredictable long-running deals. Treat predictions as central
  estimates, not guarantees, and widen the range for multi-jurisdiction regulatory deals.
