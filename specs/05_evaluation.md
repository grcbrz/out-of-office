# Spec 05 — Evaluation & Explainability

**Status:** Draft
**Created:** 2026-04-23
**Depends on:** Spec 04 (Model Training)
**Consumed by:** Spec 06 (Serving)

---

## 1. Problem Statement

Evaluate the production model and all fold models across three granularities (per fold, aggregated, per ticker), compute explainability outputs (SHAP values + attention weights for transformers), and persist results to both MLflow and CSV. Evaluation runs after every training cycle and at inference time for per-prediction explanation.

---

## 2. Scope

**In scope:**
- Classification metrics: per fold, aggregated across folds, per ticker breakdown
- Financial metrics: Sharpe ratio, max drawdown, hit rate on directional calls
- SHAP values for the production candidate (LightGBM via `TreeExplainer`; `KernelExplainer` fallback for arbitrary estimators)
- Attention weight extraction stub kept in place (no-op until a transformer candidate ships — see Spec 04 §6.2)
- Per-prediction explanation logged alongside signal at inference time
- Evaluation report persisted to MLflow (metrics + artifacts) and CSV
- Metric thresholds as quality gates — training fails if production fold falls below threshold

**Out of scope:**
- Live P&L tracking (no execution)
- Backtesting engine (future spec)
- Dashboard visualisation (Spec 06)

---

## 3. Metrics

### 3.1 Classification Metrics

Computed on the **validation fold** for each fold, then aggregated.

| Metric | Scope | Notes |
|---|---|---|
| F1-macro | Per fold, aggregated, per ticker | Primary selection metric |
| F1 per class | Per fold, aggregated, per ticker | BUY / HOLD / SELL separately |
| Precision per class | Per fold, aggregated, per ticker | |
| Recall per class | Per fold, aggregated, per ticker | |
| MCC (Matthews Correlation Coefficient) | Per fold, aggregated | Robust to class imbalance |
| ROC-AUC (one-vs-rest, macro) | Per fold, aggregated | Computed from `predict_proba`, **never** from hard labels |
| Confusion matrix | Per fold, aggregated | Logged as artifact |
| Baseline F1-macro | Per fold | Naive last-direction baseline; sets the gate floor (§4) |

**Aggregation method:** Mean and std across folds (not a single pooled computation). Std across folds quantifies model stability — a high mean with high std is a warning sign.

### 3.2 Financial Metrics

Computed on validation fold predictions using `forward_return` (stored in feature file, never used as model input).

> **Threshold-aware.** Sharpe, max drawdown, hit rate, and signal distribution are computed on **threshold-applied** predictions (Spec 04 §7.4) — i.e. with the per-fold τ used to demote low-confidence BUY/SELL to HOLD. Classification metrics in §3.1 stay on the raw predictions. Without this split, model selection on F1 would prefer strategies that simply stop trading.

| Metric | Definition | Notes |
|---|---|---|
| Hit rate | % of BUY/SELL signals where direction was correct | Excludes HOLD |
| Simulated daily return | `simple_return × signal direction` (+1 BUY, −1 SELL, 0 HOLD) | `simple_return = expm1(log_forward_return)` — log returns are converted before compounding (log returns can be < −1, breaking `(1 + r).cumprod`) |
| Sharpe ratio | `mean(strategy_return) / std(strategy_return) × sqrt(252)` | Annualised, on simple-return strategy series |
| Max drawdown | Min of `(cum / cummax − 1)` where `cum = (1 + strategy_return).cumprod()` | Per fold; bounded in `(−1, 0]` |
| Signal distribution | % BUY / % HOLD / % SELL per fold | Detect degenerate models |

> **Note:** Financial metrics are diagnostic, not selection criteria in v1. F1-macro drives model selection (Spec 04). Financial metrics are logged for human review.

### 3.3 Per-Ticker Breakdown

For each ticker in the validation fold:
- F1-macro
- F1 per class
- Hit rate
- Signal distribution

Logged as a CSV artifact per fold. Enables identifying tickers where the global model systematically underperforms — input for the future per-ticker fine-tuning increment.

---

## 4. Quality Gates

Minimum thresholds on the **production fold** (final fold). Training pipeline raises `EvaluationQualityGateError` and does not persist the production artifact if any gate fails.

Two classes of gate, evaluated together:

### 4.1 Absolute floors (regression guards)

| Metric | Minimum threshold | Rationale |
|---|---|---|
| F1-macro | ≥ 0.40 | Sanity floor; well above 3-class random (0.33) |
| MCC | ≥ 0.10 | Non-trivial correlation |
| Hit rate (BUY + SELL) | ≥ 0.52 | Better than coin flip with margin |
| Signal distribution | No class > 70% of predictions | Detect degenerate always-HOLD models |

### 4.2 Baseline delta (the real bar)

The production model **must beat the naive last-direction baseline (Spec 04 §6) by at least `min_delta_over_baseline` F1-macro points** on the production fold. This is the gate that answers *did the model add anything over the trivial benchmark?* If `baseline_f1_macro` is missing the delta gate is skipped with a warning (legacy artifacts only).

| Setting | Default | Rationale |
|---|---|---|
| `min_delta_over_baseline` | 0.02 | 2 percentage points — enough to be noticeable, not so high that early models can never ship. Tighten as model matures. |

Thresholds are configurable in `configs/evaluation.yaml`.

---

## 5. Explainability

### 5.1 SHAP Values

**Library:** `shap`
**Method (v1):** `shap.TreeExplainer` for LightGBM (exact, no sampling).
**Fallback:** `shap.KernelExplainer` on a 100-sample background subset for any non-tree estimator (e.g. the baseline; future neural models). Log which explainer was used on each run.
**Computed on:** Validation fold (training-time) and per prediction (inference-time)

**Output per fold:**
- Mean absolute SHAP value per feature → feature importance ranking
- SHAP values array (TreeExplainer/KernelExplainer multi-class: shape `(n_classes, n_val_samples, n_features)`) — stored as a CSV artifact, flattened per class

**Output per prediction (inference-time):**
- SHAP values for that single prediction
- Top 5 features by absolute SHAP value (defaults to the predicted class)
- Logged alongside the BUY/HOLD/SELL signal in the prediction output

### 5.2 Attention Weights (deferred)

`AttentionExtractor` is preserved but inert in v1: there is no transformer candidate in production (see Spec 04 §6.2). It returns `None` for every model, and inference logs `attention_weights: null` accordingly. When a real transformer candidate ships, register its `model_name` in `_TRANSFORMER_MODELS` to re-enable extraction.

### 5.3 Feature Importance Report

Aggregated mean absolute SHAP across all validation folds → ranked feature list.
Persisted as `reports/evaluation/{date}/feature_importance.csv`:

| rank | feature | mean_abs_shap | std_abs_shap |
|---|---|---|---|
| 1 | close_zscore | 0.042 | 0.008 |
| 2 | macd_hist | 0.038 | 0.011 |
| ... | ... | ... | ... |

---

## 6. Persistence

### 6.1 MLflow

Logged under the same experiment run as training (Spec 04):

| Type | Content |
|---|---|
| Metrics | All classification + financial metrics, per fold and aggregated |
| Artifacts | Confusion matrix (CSV), per-ticker breakdown (CSV), SHAP matrix (CSV), attention weights (npy), feature importance (CSV) |
| Tags | `quality_gate_passed: true/false`, `production_fold: true/false` |

### 6.2 CSV Reports

Persisted to `reports/evaluation/{date}/`:

```
reports/evaluation/2026-04-23/
├── metrics_per_fold.csv          # all classification + financial metrics per fold
├── metrics_aggregated.csv        # mean + std across folds
├── metrics_per_ticker.csv        # per-ticker breakdown across all folds
├── confusion_matrix_{fold}.csv   # one per fold
├── feature_importance.csv        # aggregated SHAP ranking
└── quality_gate_result.json      # pass/fail + threshold values
```

CSV files are appended across runs (date-stamped rows), not overwritten. Full history retained.

---

## 7. Inference-Time Explanation

At prediction time (Spec 06), each signal is accompanied by:

```json
{
  "ticker": "AAPL",
  "date": "2026-04-24",
  "signal": "BUY",
  "confidence": 0.71,
  "forward_return_estimate": null,
  "explanation": {
    "top_features": [
      {"feature": "close_zscore", "shap_value": 0.18},
      {"feature": "macd_hist", "shap_value": 0.12},
      {"feature": "bullish_percent", "shap_value": 0.09},
      {"feature": "close_lag1", "shap_value": -0.07},
      {"feature": "volume_zscore", "shap_value": 0.06}
    ],
    "attention_weights": [0.02, 0.03, 0.15, ...],
    "explainer_used": "DeepExplainer"
  }
}
```

`confidence` = softmax probability of the predicted class.
`attention_weights` = null for N-HiTS.

---

## 8. Module Structure

```
src/evaluation/
├── __init__.py
├── classification.py       # compute_classification_metrics — F1, MCC, AUC, confusion matrix
├── financial.py            # compute_financial_metrics — Sharpe, drawdown, hit rate
├── aggregation.py          # aggregate_across_folds, per_ticker_breakdown
├── quality_gate.py         # QualityGate — threshold checks, EvaluationQualityGateError
├── explainability/
│   ├── __init__.py
│   ├── shap_explainer.py   # SHAPExplainer — DeepExplainer + KernelExplainer fallback
│   └── attention.py        # AttentionExtractor — extract + average attention weights
├── persistence.py          # log_to_mlflow, write_csv_reports
└── pipeline.py             # EvaluationPipeline — orchestrates full evaluation run
```

---

## 9. Configuration

```yaml
# configs/evaluation.yaml
quality_gates:
  # Absolute floors — guard against catastrophic regression.
  f1_macro_min: 0.40
  mcc_min: 0.10
  hit_rate_min: 0.52
  max_signal_concentration: 0.70   # no single class > 70% of predictions

  # Baseline delta — the real bar.
  min_delta_over_baseline: 0.02

shap:
  background_sample_size: 100      # for KernelExplainer fallback
  top_n_features: 5                # per-prediction explanation

reports:
  output_dir: reports/evaluation
  append_mode: true
```

---

## 10. Acceptance Criteria

- [ ] F1-macro, F1 per class, Precision, Recall, MCC, ROC-AUC computed per fold and aggregated (mean + std)
- [ ] ROC-AUC computed from `predict_proba`; falls back to `None` when probabilities are unavailable (never computed on hard labels)
- [ ] Confusion matrix computed and stored per fold
- [ ] Per-ticker breakdown (F1-macro, F1 per class, hit rate, signal distribution) logged per fold
- [ ] Sharpe and max drawdown computed on **simple** strategy returns (`expm1(log_forward_return) * direction`)
- [ ] Hit rate computed on directional signals only (HOLD excluded) using sign of forward log return
- [ ] `forward_return` never passed as model input — enforced before metric computation
- [ ] Naive last-direction baseline trained on every fold; baseline F1-macro logged alongside model F1
- [ ] `baseline_f1_macro` and `baseline_aggregated_f1_macro` persisted in production metadata
- [ ] Quality gates evaluated on production fold; `EvaluationQualityGateError` raised on absolute-floor or baseline-delta failure
- [ ] Production artifact not written if quality gate fails
- [ ] SHAP values computed for the production candidate; tree boosters use `TreeExplainer`, others fall back to `KernelExplainer`; explainer type logged
- [ ] `attention_weights` is `null` in every prediction record while no transformer candidate is in `_TRANSFORMER_MODELS`
- [ ] Feature importance CSV produced (aggregated mean abs SHAP across folds)
- [ ] All metrics logged to MLflow under correct parent/child run
- [ ] CSV reports written to `reports/evaluation/{date}/`; append mode, not overwrite
- [ ] `quality_gate_result.json` written with pass/fail and threshold values
- [ ] Inference-time explanation includes top-5 SHAP features + confidence + attention weights
- [ ] Explainer used (Deep vs Kernel) logged per run

---

## 11. Tests Required

| Test | Type | Notes |
|---|---|---|
| `test_f1_macro_computation` | Unit | Known predictions; assert correct F1-macro |
| `test_mcc_computation` | Unit | Spot-check against sklearn reference |
| `test_confusion_matrix_shape` | Unit | 3×3 matrix for 3 classes |
| `test_sharpe_ratio_computation` | Unit | Synthetic daily returns; assert annualised value |
| `test_max_drawdown_computation` | Unit | Known peak-trough series |
| `test_hit_rate_excludes_hold` | Unit | HOLD signals not counted in hit rate |
| `test_aggregation_mean_std` | Unit | 3 mock folds; assert mean and std correct |
| `test_per_ticker_breakdown` | Unit | 3 tickers in val fold; assert per-ticker metrics |
| `test_quality_gate_passes` | Unit | Metrics above threshold and over baseline; no error |
| `test_quality_gate_fails_f1` | Unit | F1 below absolute floor; `EvaluationQualityGateError` raised |
| `test_quality_gate_fails_baseline_delta` | Unit | F1 above absolute floor but ≤ baseline + δ; error raised |
| `test_quality_gate_skips_baseline_when_missing` | Unit | `baseline_f1_macro` absent; warning logged, no error |
| `test_quality_gate_fails_concentration` | Unit | 85% HOLD signals; error raised |
| `test_roc_auc_uses_proba` | Unit | Probabilities passed → finite AUC; `None` → roc_auc field is `None` |
| `test_financial_metrics_simple_returns` | Unit | Hand-computed Sharpe and MDD on a small log-return series via `expm1` |
| `test_shap_deep_explainer` | Unit | Mock model; assert SHAP output shape |
| `test_shap_kernel_fallback` | Unit | Force fallback; assert `KernelExplainer` used and logged |
| `test_attention_returns_null_for_lightgbm` | Unit | Production candidate has no attention → null |
| `test_attention_returns_null_for_baseline` | Unit | Baseline has no attention → null |
| `test_shap_uses_tree_explainer_for_lightgbm` | Unit | TreeExplainer selected; KernelExplainer not invoked |
| `test_shap_falls_back_to_kernel_for_unknown_model` | Unit | Non-tree estimator → KernelExplainer used and logged |
| `test_feature_importance_ranking` | Unit | Known SHAP matrix; assert correct rank order |
| `test_inference_explanation_structure` | Unit | Assert top-5 features, confidence, explainer logged |
| `test_csv_append_mode` | Unit | Two runs same date; rows appended, not overwritten |
| `test_mlflow_metrics_logged` | Integration | Mock run; assert all metric keys present in MLflow |
| `test_full_evaluation_pipeline` | Integration | Mock fold output; assert all reports written |
