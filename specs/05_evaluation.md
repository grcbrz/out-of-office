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
- SHAP values for all three model architectures
- Attention weight extraction for PatchTST and Autoformer
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
| ROC-AUC (one-vs-rest) | Per fold, aggregated | Per class |
| Confusion matrix | Per fold, aggregated | Logged as artifact |

**Aggregation method:** Mean and std across folds (not a single pooled computation). Std across folds quantifies model stability — a high mean with high std is a warning sign.

### 3.2 Financial Metrics

Computed on validation fold predictions using `forward_return` (stored in feature file, never used as model input).

| Metric | Definition | Notes |
|---|---|---|
| Hit rate | % of BUY/SELL signals where direction was correct | Excludes HOLD |
| Simulated daily return | `forward_return` × signal direction (+1 BUY, −1 SELL, 0 HOLD) | No transaction costs in v1 |
| Sharpe ratio | `mean(daily_return) / std(daily_return) × sqrt(252)` | Annualised |
| Max drawdown | Maximum peak-to-trough cumulative return loss | Per fold |
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

| Metric | Minimum threshold | Rationale |
|---|---|---|
| F1-macro | ≥ 0.35 | Above random (0.33 for 3 classes) |
| MCC | ≥ 0.05 | Non-trivial correlation |
| Hit rate (BUY + SELL) | ≥ 0.50 | Better than coin flip on directional calls |
| Signal distribution | No class > 80% of predictions | Detect degenerate always-HOLD models |

Thresholds are configurable in `configs/evaluation.yaml`. These are conservative v1 baselines — tighten as the model matures.

---

## 5. Explainability

### 5.1 SHAP Values

**Library:** `shap`
**Method:** `shap.DeepExplainer` for PyTorch models
**Computed on:** Validation fold (training-time) and per prediction (inference-time)

**Output per fold:**
- Mean absolute SHAP value per feature → feature importance ranking
- SHAP values matrix: `(n_val_samples, n_features)` — stored as CSV artifact

**Output per prediction (inference-time):**
- SHAP values for that single prediction
- Top 5 features by absolute SHAP value
- Logged alongside the BUY/HOLD/SELL signal in the prediction output

**Edge case:** If `DeepExplainer` is incompatible with a given architecture version, fall back to `shap.KernelExplainer` on a 100-sample background subset. Log which explainer was used.

### 5.2 Attention Weights (PatchTST, Autoformer only)

**Extracted from:** Final attention layer of the encoder
**Shape:** `(n_heads, seq_len, seq_len)` — averaged across heads for interpretability
**Stored as:** NumPy array artifact per fold (`.npy` file)

At inference time, per-prediction attention weights extracted and stored as a flat vector (mean across heads, last query position) alongside the signal.

N-HiTS has no attention mechanism — skip silently, log `attention_weights: null` in prediction output.

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
  f1_macro_min: 0.35
  mcc_min: 0.05
  hit_rate_min: 0.50
  max_signal_concentration: 0.80   # no single class > 80% of predictions

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
- [ ] Confusion matrix computed and stored per fold
- [ ] Per-ticker breakdown (F1-macro, F1 per class, hit rate, signal distribution) logged per fold
- [ ] Sharpe ratio, max drawdown, hit rate computed on validation fold using `forward_return`
- [ ] `forward_return` never passed as model input — enforced before metric computation
- [ ] Quality gates evaluated on production fold; `EvaluationQualityGateError` raised on failure
- [ ] Production artifact not written if quality gate fails
- [ ] SHAP values computed for all three architectures; fallback to `KernelExplainer` logged
- [ ] Attention weights extracted for PatchTST and Autoformer; null for N-HiTS
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
| `test_quality_gate_passes` | Unit | Metrics above threshold; no error |
| `test_quality_gate_fails_f1` | Unit | F1 below 0.35; `EvaluationQualityGateError` raised |
| `test_quality_gate_fails_concentration` | Unit | 85% HOLD signals; error raised |
| `test_shap_deep_explainer` | Unit | Mock model; assert SHAP output shape |
| `test_shap_kernel_fallback` | Unit | Force fallback; assert `KernelExplainer` used and logged |
| `test_attention_extraction_transformer` | Unit | Mock PatchTST; assert weight shape and averaging |
| `test_attention_null_for_nhits` | Unit | N-HiTS model; attention returns null |
| `test_feature_importance_ranking` | Unit | Known SHAP matrix; assert correct rank order |
| `test_inference_explanation_structure` | Unit | Assert top-5 features, confidence, explainer logged |
| `test_csv_append_mode` | Unit | Two runs same date; rows appended, not overwritten |
| `test_mlflow_metrics_logged` | Integration | Mock run; assert all metric keys present in MLflow |
| `test_full_evaluation_pipeline` | Integration | Mock fold output; assert all reports written |
