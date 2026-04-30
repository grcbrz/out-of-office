# Spec 04 — Model Training

**Status:** Draft
**Created:** 2026-04-23
**Depends on:** Spec 03 (Feature Engineering)
**Consumed by:** Spec 05 (Evaluation & Explainability)

---

## 1. Problem Statement

Train a global BUY/HOLD/SELL classifier across all 50 tickers using a sliding walk-forward harness. Three candidate architectures (N-HiTS, PatchTST, Autoformer) are trained in parallel per fold. The best-performing model per fold is retained as the production artifact. Training is fully reproducible from config, logged via MLflow, and produces no data leakage by construction.

---

## 2. Scope

**In scope:**
- Sliding walk-forward harness (252-day train window, 21-day step)
- Global model training across all tickers (ticker identity as a feature)
- Three candidate models: N-HiTS, PatchTST, Autoformer
- Parallel training per fold (one process per model)
- Model selection: best F1-macro on validation fold wins
- Hyperparameter configuration via YAML (no hardcoded values)
- MLflow run tracking: params, metrics, artifacts, git commit hash
- Latest fold artifact persistence only
- Reproducibility: fixed random seeds, pinned deps, config-driven

**Out of scope:**
- Per-ticker fine-tuning / transfer learning (future increment)
- Hyperparameter search (future spec — fixed configs in v1)
- Serving or inference (Spec 06)
- Drift monitoring and retraining triggers (Spec 07)

---

## 3. Input

**File:** `data/features/{ticker}/{date}.csv`
**Schema:** `FeatureRecord` (Spec 03)
All 50 tickers loaded and concatenated into a single global dataset, sorted by `(date, ticker)`.

**Feature columns used for training** (model input `X`):

- `log_return`, `log_return_lag1/2/3`
- `sma_10`, `sma_20`, `ema_10`, `ema_20`
- `macd`, `macd_signal`, `macd_hist`, `close_to_sma20`
- `obv`, `obv_lag1`, `vwap_ratio`
- `close_lag1/2/3`, `volume_lag1/2/3`
- `close_zscore`, `volume_zscore`
- `day_of_week`, `week_of_year`, `month`, `is_month_end`
- `bullish_percent`, `bearish_percent`, `company_news_score`, `buzz_weekly_average`
- `sentiment_available`
- `close_outlier_flag`, `volume_outlier_flag`
- `ticker_id` (integer-encoded ticker identity — see Section 5.1)

**Excluded from model input** (present in file, never passed to model):
- `forward_return` — target construction only, enforced by `LookaheadBiasGuard`
- `date`, `ticker` (raw string) — used for indexing only

**Target:** `target` column → encoded as `{BUY: 2, HOLD: 1, SELL: 0}`

---

## 4. Walk-Forward Harness

### 4.1 Structure

```
Total history: ~500 trading days per ticker (2 years)

Fold layout (per step):
├── Train window : 252 trading days  (fixed size, slides forward)
├── Validation   : 21 trading days   (immediately follows train)
└── Step size    : 21 trading days   (window advances by 1 month)
```

No overlap between train and validation within any fold. No data from validation or future folds leaks into training.

### 4.2 Fold Generation

```
fold_1:  train[0:252],   val[252:273]
fold_2:  train[21:273],  val[273:294]
fold_3:  train[42:294],  val[294:315]
...
final:   train[T-252:T], val[T:T+21]  ← production fold
```

Folds generated from the global dataset sorted by date. Ticker mixing within a date is allowed — the model is global. Cross-ticker contamination across time is not possible by construction.

### 4.3 Final Production Fold

The last fold uses the most recent 252 trading days as training data. Its validation set is the most recent 21 days. The model winning on this fold is the production artifact.

---

## 5. Data Preparation

### 5.1 Ticker Encoding

`ticker` (string) → `ticker_id` (integer) via a fixed mapping persisted to `configs/ticker_map.json`. Mapping is stable across runs — new tickers appended, existing tickers never re-encoded. This preserves embedding consistency across retraining.

### 5.2 Null Handling Before Training

- Sentiment nulls: impute with column median computed on the training fold only. Median stored in the run artifact for inference-time use.
- `vwap_ratio` nulls: same strategy.
- Any remaining nulls after imputation: raise `TrainingDataError` — do not train on NaN inputs.

Imputation parameters are computed on the train fold and applied to the validation fold. No leakage.

### 5.3 Class Weight Balancing

Target distribution is approximately 30/40/30 (BUY/HOLD/SELL). Apply class weights inversely proportional to frequency, computed per training fold. Passed to loss function — not via oversampling (which risks leakage in time series).

---

## 6. Model Architectures

> **History.** Earlier drafts of this spec listed three candidate transformers
> (N-HiTS, PatchTST, Autoformer) sourced from `neuralforecast`. The shipped
> implementations were sklearn placeholders under transformer-flavoured names —
> a mismatch between spec and code. v1 corrected that by shipping a single,
> truthfully-named gradient-boosted candidate. v1.1 adds RandomForest as a
> diversity candidate. Real neuralforecast architectures remain deferred
> until daily 3-class equity classification with ~12k training rows produces
> evidence that transformers outperform tree ensembles here, which is not
> the empirical default on this scale of data.

> **Mandatory naive baseline.** Every fold trains a `BaselineLastDirectionWrapper` alongside the candidates (Spec 05 §4.2). The baseline is **never** a production candidate — `select_winner` filters it out — but its F1-macro is logged per fold and aggregated, and the quality gate (Spec 05 §4.2) requires the production model to beat it by `min_delta_over_baseline` F1 points. Without this benchmark there is no answer to "did the model add anything over the trivial rule?".

> **Why two tree candidates?** LightGBM (boosting) and RandomForest (bagging) have uncorrelated failure modes: boosting overfits sequential noise on choppy folds; bagging averages noise away but lags steep regime changes. Per-fold selection picks the better of the two; aggregate selection picks the model with the highest mean F1 across folds.

### 6.1 LightGBM (production candidate, primary)

- Multi-class gradient-boosted trees (`lightgbm.LGBMClassifier`)
- Robust to mixed-scale features; cyclic calendar encodings handled natively without scaling
- Class imbalance handled via `sample_weight` derived from `DataPreparer.compute_class_weights` (per-fold, computed on training data only)
- SHAP via `shap.TreeExplainer` — exact, fast, no sampling background needed
- Config key: `models.lightgbm`

```yaml
# configs/models/lightgbm.yaml
n_estimators: 400
learning_rate: 0.05
num_leaves: 31
max_depth: -1            # num_leaves is the binding cap
min_child_samples: 20
feature_fraction: 0.9
bagging_fraction: 0.8
bagging_freq: 5
reg_alpha: 0.0
reg_lambda: 0.1
random_seed: 42
n_jobs: -1
```

Hardcoded inside the wrapper (not configurable):

| Setting | Value | Reason |
|---|---|---|
| `objective` | `multiclass` | We classify SELL/HOLD/BUY |
| `num_class` | `3` | Pipeline emits exactly 3 targets |
| `metric` | `multi_logloss` | Tracks the training objective |

### 6.2 RandomForest (production candidate, diversity)

- Bagged trees (`sklearn.ensemble.RandomForestClassifier`)
- **Class imbalance contract differs from LightGBM:** `class_weight={class_id: weight}` is passed at *construction*, not at fit time. RandomForest's bootstrap sampling reweights draws correctly only when class_weight is set on the constructor; passing weights as `sample_weight` at fit would interact with the bootstrap and double-up minority-class influence.
- SHAP via `shap.TreeExplainer` — same path as LightGBM
- Config key: `models.randomforest`

```yaml
# configs/models/randomforest.yaml
n_estimators: 400
max_depth: null            # let trees grow; rely on min_samples_leaf for regularisation
min_samples_leaf: 5
min_samples_split: 10
max_features: sqrt         # ~sqrt(30) ≈ 5 features per split
bootstrap: true
n_jobs: 1                  # determinism — matches LightGBM thread pinning
random_seed: 42
```

### 6.3 Adding additional candidates

The harness iterates `_CANDIDATE_NAMES` in `src/models/training_pipeline.py` and selects on F1-macro. To add a candidate (e.g. a real `neuralforecast` Autoformer):

1. Implement `BaseModelWrapper` subclass under `src/models/architectures/{name}.py`.
2. Append the model name to `_CANDIDATE_NAMES`.
3. Branch in `_instantiate_wrapper`.
4. Add `configs/models/{name}.yaml`.
5. Add the name to `_PREFERENCE_ORDER` in `src/models/selector.py` for tie-breaks.

Random seeds remain fixed globally for reproducibility:

```python
import random, numpy as np, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)
```

---

## 7. Training Harness

### 7.1 Per-Fold Flow

```
for each fold:
    1. Slice train/val windows from global dataset
    2. Compute imputation params on train fold
    3. Apply imputation to train + val
    4. Compute class weights on train fold
    5. Train every candidate + the baseline (sequentially in v1; parallelisation deferred until candidates ≥ 2)
    6. Evaluate each model on val fold → F1-macro (baseline included, but excluded from selection)
    7. Select winner (highest F1-macro among non-baseline candidates; tie-break uses `_PREFERENCE_ORDER`)
    8. Log every model's metrics to MLflow
    9. If final fold: persist winner artifact to models/production/, including baseline benchmark in metadata
```

### 7.2 Parallel Training

Each model trained in a separate process via `concurrent.futures.ProcessPoolExecutor`. Each process:
- Receives its own copy of train/val data
- Logs to the same MLflow parent run as a nested child run
- Returns `(model_name, f1_macro, artifact_path)`

### 7.3 Early Stopping

All neural-network candidates use validation loss as the early stopping criterion (patience = 20 steps). Prevents overfitting on small folds. LightGBM uses `n_estimators` capped via YAML (`configs/models/lightgbm.yaml`) and bagging instead.

### 7.4 Confidence-Threshold Calibration (per fold)

After each model is fit and predictions on the validation fold are produced, the harness calibrates a **confidence threshold τ** by grid-searching `DEFAULT_CANDIDATE_TAUS` (`0.34, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70`). For each candidate τ:

1. Apply the threshold: any prediction with `max(predict_proba) ≤ τ` is demoted to HOLD.
2. Compute strategy returns and Sharpe on the thresholded prediction series.

The τ that produces the highest validation Sharpe wins (ties broken by preferring the **lowest** τ — keeps the strategy from collapsing to all-HOLD when several thresholds are statistically equivalent). If every τ produces fewer than 5% non-HOLD signals the model is treated as having no usable confidence signal: the lowest candidate τ is returned and the quality gate (Spec 05 §4) will catch the resulting Sharpe / hit rate failure.

**Honesty caveat.** τ is calibrated on the same validation window the per-fold financial metrics are reported on, which makes the in-fold Sharpe slightly optimistic. With 21-day val windows splitting further would leave too few rows for stable calibration. The walk-forward harness mitigates this — across N folds the optimism averages out, and serving trades the next fold using the previous fold's τ.

τ is logged per fold and persisted at the **top level** of `metadata.json` (`confidence_threshold`) so `ArtifactLoader` (Spec 06) can read it without depending on the shape of `production_fold`. The baseline never has a τ — its proba is one-hot so thresholding is a no-op.

**Selection vs. trading.** Classification metrics (F1, MCC, ROC-AUC) are computed on **raw** predictions because thresholding collapses the confusion matrix toward HOLD and would bias model selection toward strategies that simply stop trading. Financial metrics (Sharpe, MDD, hit rate, signal distribution, signal counts) are computed on the **thresholded** predictions because that is what production trades.

---

## 8. Model Selection

**Metric:** F1-macro on validation fold (equal weight across BUY, HOLD, SELL classes)

**Tie-breaking:** order defined by `_PREFERENCE_ORDER` in `src/models/selector.py` (currently `["lightgbm", "randomforest"]` — boosting preferred over bagging on equal F1 because it tends to be more sample-efficient on tabular daily-equity data and trains faster).

**Winner logged** with tag `production=true` in MLflow. Every run is logged, including the baseline.

---

## 9. Artifact Persistence

**Production model:** `models/production/{model_name}/`

Contents:
```
models/production/
└── lightgbm/                     # winning candidate name
    ├── model.pkl                 # pickled wrapper state (booster + features + config)
    ├── config.yaml               # exact hyperparams used
    ├── imputation_params.json    # median values for null imputation
    ├── ticker_map.json           # ticker → ticker_id mapping
    ├── class_weights.json        # per-class weights used in training
    ├── monitoring_reference.json # signal counts + training window for drift gates
    └── metadata.json             # fold dates, val metrics, baseline benchmark, git hash, mlflow run id
```

Future neural-forecast candidates would persist a `model.pt` instead of (or alongside) `model.pkl` — the artifact loader handles both.

Previous production artifacts are overwritten. Git history + MLflow provide the audit trail.

---

## 10. MLflow Tracking

**Experiment name:** `stock-recommender`
**Tracking URI:** local (`mlruns/`)

Per fold, per model, log:
- **Params:** model name, hyperparams, fold start/end dates, train size, val size, random seed
- **Metrics:** F1-macro, F1 per class (BUY/HOLD/SELL), MCC, ROC-AUC
- **Tags:** `fold_index`, `is_production_fold`, `winner`, `git_commit`
- **Artifacts:** model weights, config, imputation params

---

## 11. Configuration

All training parameters in `configs/training.yaml`:

```yaml
walk_forward:
  train_window: 252       # trading days
  step_size: 21           # trading days
  min_folds: 3            # abort if fewer folds available

data:
  feature_dir: data/features
  ticker_map: configs/ticker_map.json

training:
  random_seed: 42
  early_stopping_patience: 20
  n_workers: 1            # v1 ships with a single candidate; raise when more land

mlflow:
  experiment_name: stock-recommender
  tracking_uri: mlruns/

artifacts:
  production_dir: models/production
```

---

## 12. Module Structure

```
src/models/
├── __init__.py
├── architectures/
│   ├── __init__.py
│   ├── base.py             # BaseModelWrapper — train/predict/predict_proba/save/load contract
│   ├── lightgbm.py         # LightGBMWrapper — primary production candidate (boosting)
│   ├── randomforest.py     # RandomForestWrapper — diversity production candidate (bagging)
│   └── baseline.py         # BaselineLastDirectionWrapper — naive benchmark (Spec 05 §4.2)
├── harness.py              # generate_folds — walk-forward fold construction
├── selector.py             # ModelResult, select_winner — F1-macro comparison, tie-breaking
├── preparation.py          # DataPreparer — encoding, imputation, class weights
├── persistence.py          # save_artifact, load_artifact
└── training_pipeline.py    # TrainingPipeline — entry point
```

---

## 13. Acceptance Criteria

- [ ] Walk-forward folds generated with 252-day train window and 21-day step; no temporal leakage between folds
- [ ] All three models trained per fold; training is parallel (separate processes)
- [ ] `forward_return` absent from model input; enforced by `LookaheadBiasGuard` before training
- [ ] Imputation params computed on train fold only; applied to val fold without leakage
- [ ] Class weights computed per training fold
- [ ] Model selection uses F1-macro on val fold; tie-breaking order respected
- [ ] All three models + metrics logged to MLflow per fold (parent + nested child runs)
- [ ] Production artifact written only from final fold winner
- [ ] `models/production/` contains `model.pkl`, `config.yaml`, `imputation_params.json`, `ticker_map.json`, `class_weights.json`, `monitoring_reference.json`, `metadata.json`
- [ ] `metadata.json` contains the baseline benchmark (`baseline.name`, `baseline.mean_f1_macro`, `baseline.production_fold_f1_macro`)
- [ ] `metadata.json` contains a top-level `confidence_threshold` field (the per-fold τ persisted on the production fold's winning candidate)
- [ ] Per-fold metrics dict carries `confidence_threshold`; financial metrics are computed on thresholded predictions; classification metrics on raw predictions
- [ ] Random seeds fixed and logged; rerunning same config produces identical results
- [ ] `ticker_map.json` stable across runs; new tickers appended, existing never re-encoded
- [ ] Training aborts with `TrainingDataError` if any NaN remains after imputation
- [ ] Hyperparameters sourced from `configs/models/{name}.yaml`; pipeline-invariants (`objective`, `num_class`, `metric`) hardcoded and documented in §6.1

---

## 14. Tests Required

| Test | Type | Notes |
|---|---|---|
| `test_fold_generation_no_leakage` | Unit | Val dates never appear in train window |
| `test_fold_count` | Unit | Correct number of folds for given dataset length |
| `test_imputation_train_only` | Unit | Median computed on train; val uses train median |
| `test_class_weights_per_fold` | Unit | Weights inverse to class frequency in train fold |
| `test_ticker_encoding_stable` | Unit | Re-encoding same tickers produces same IDs |
| `test_ticker_encoding_append` | Unit | New ticker gets new ID; existing IDs unchanged |
| `test_model_selection_f1_macro` | Unit | Higher F1-macro wins among candidates |
| `test_model_selection_tiebreak_prefers_lightgbm` | Unit | Equal F1 → preference order respected (lightgbm > randomforest) |
| `test_select_winner_excludes_baseline` | Unit | Baseline never returned as production winner |
| `test_lightgbm_wrapper_train_predict_roundtrip` | Unit | Fit on synthetic data; predict + predict_proba return correct shapes |
| `test_lightgbm_wrapper_class_weights_applied` | Unit | sample_weight passed to `model.fit` reflects per-class weights |
| `test_randomforest_wrapper_train_predict_roundtrip` | Unit | Fit on synthetic data; predict + predict_proba return correct shapes |
| `test_randomforest_class_weight_passed_to_constructor` | Unit | `class_weight` reaches `RandomForestClassifier(...)` at __init__, not at fit |
| `test_randomforest_string_class_weight_keys_coerced_to_int` | Unit | JSON-stringified keys (`{"0": ...}`) are coerced before reaching sklearn |
| `test_apply_confidence_threshold` | Unit | Predictions with `max(proba) ≤ τ` demoted to HOLD; others unchanged |
| `test_calibrate_threshold_picks_sharpe_max` | Unit | Synthetic proba + log-return series; calibrator returns the τ with the best Sharpe |
| `test_calibrate_threshold_lowest_tau_on_tie` | Unit | When two τ produce equal Sharpe the lower one is returned (more trades) |
| `test_calibrate_threshold_degenerate_fallback` | Unit | If every τ leaves <5% directional signals the lowest candidate is returned |
| `test_fold_metrics_apply_threshold_to_financial_only` | Unit | Per-fold metrics: F1 on raw preds, hit_rate on thresholded preds |
| `test_lookahead_guard_before_training` | Unit | `forward_return` in features → `LookaheadBiasError` |
| `test_nan_after_imputation_raises` | Unit | Residual NaN → `TrainingDataError` |
| `test_random_seed_reproducibility` | Unit | Two runs same config → identical val metrics |
| `test_artifact_contents_includes_baseline` | Unit | `metadata.json` carries baseline benchmark fields |
| `test_mlflow_run_logged` | Integration | Mock training; assert MLflow run created with params + metrics |
| `test_full_harness_mock` | Integration | 2 folds, mock candidate + real baseline; assert fold loop, selection, artifact write |

---

## 15. Future Increment — Transfer Learning

In the next planned increment:
- Freeze lower layers of the winning global model (pattern recognition)
- Fine-tune top layers per ticker using recent ticker-specific history
- Per-ticker artifacts stored separately under `models/production/per_ticker/{ticker}/`
- This spec remains unchanged — transfer learning is additive
