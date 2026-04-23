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

All three models receive the same input tensor per fold. Implementations sourced from `neuralforecast` (N-HiTS, Autoformer) and a custom PatchTST wrapper or `neuralforecast` if available.

### 6.1 N-HiTS

- Multi-rate signal sampling; native hierarchical decomposition handles trend and seasonality
- Input: univariate per ticker with exogenous features
- Config key: `models.nhits`

```yaml
# configs/models/nhits.yaml
input_size: 30        # lookback window (trading days)
h: 1                  # forecast horizon (t+1)
n_freq_downsample: [2, 1, 1]
mlp_units: [[512, 512], [512, 512], [512, 512]]
dropout_prob_theta: 0.1
max_steps: 500
learning_rate: 1e-3
random_seed: 42
```

### 6.2 PatchTST

- Patch-based transformer; strong on local temporal patterns in financial series
- Input: multivariate (all features as channels)
- Config key: `models.patchtst`

```yaml
# configs/models/patchtst.yaml
input_size: 42        # lookback window; must be divisible by patch_len
patch_len: 6
stride: 3
h: 1
d_model: 64
n_heads: 4
d_ff: 128
dropout: 0.1
max_steps: 500
learning_rate: 1e-3
random_seed: 42
```

### 6.3 Autoformer

- Included for completeness; expected to underperform N-HiTS and PatchTST on t+1
- Auto-correlation decomposition; higher compute cost on M2
- Config key: `models.autoformer`

```yaml
# configs/models/autoformer.yaml
input_size: 42
h: 1
hidden_size: 64
n_head: 4
dropout: 0.1
max_steps: 300        # reduced — M2 compute budget
learning_rate: 1e-3
random_seed: 42
```

All random seeds fixed globally:

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
    5. Train N-HiTS, PatchTST, Autoformer in parallel (multiprocessing)
    6. Evaluate each model on val fold → F1-macro
    7. Select winner (highest F1-macro; tie → prefer N-HiTS > PatchTST > Autoformer)
    8. Log all three models + metrics to MLflow
    9. If final fold: persist winner artifact to models/production/
```

### 7.2 Parallel Training

Each model trained in a separate process via `concurrent.futures.ProcessPoolExecutor`. Each process:
- Receives its own copy of train/val data
- Logs to the same MLflow parent run as a nested child run
- Returns `(model_name, f1_macro, artifact_path)`

### 7.3 Early Stopping

All three models use validation loss as the early stopping criterion (patience = 20 steps). Prevents overfitting on small folds.

---

## 8. Model Selection

**Metric:** F1-macro on validation fold (equal weight across BUY, HOLD, SELL classes)

**Tie-breaking:** N-HiTS > PatchTST > Autoformer (prefer simpler architecture)

**Winner logged** with tag `production=true` in MLflow. All three runs logged regardless.

---

## 9. Artifact Persistence

**Production model:** `models/production/{model_name}/`

Contents:
```
models/production/
└── nhits/                        # or patchtst / autoformer
    ├── model.pt                  # PyTorch weights
    ├── config.yaml               # exact hyperparams used
    ├── imputation_params.json    # median values for null imputation
    ├── ticker_map.json           # ticker → ticker_id mapping
    ├── class_weights.json        # per-class weights used in training
    └── metadata.json             # fold dates, val metrics, git hash, mlflow run id
```

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
  n_workers: 3            # one per model; tune down if M2 memory pressure

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
│   ├── nhits.py            # NHiTSWrapper — train, predict, save, load
│   ├── patchtst.py         # PatchTSTWrapper
│   └── autoformer.py       # AutoformerWrapper
├── harness.py              # WalkForwardHarness — fold generation + orchestration
├── selector.py             # ModelSelector — F1-macro comparison, tie-breaking
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
- [ ] `models/production/` contains model weights, config, imputation params, ticker map, metadata
- [ ] Random seeds fixed and logged; rerunning same config produces identical results
- [ ] Early stopping active for all three models (patience=20)
- [ ] `ticker_map.json` stable across runs; new tickers appended, existing never re-encoded
- [ ] Training aborts with `TrainingDataError` if any NaN remains after imputation
- [ ] All hyperparams sourced from YAML config; no hardcoded values in code

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
| `test_model_selection_f1_macro` | Unit | Higher F1-macro wins |
| `test_model_selection_tiebreak` | Unit | Equal F1 → N-HiTS preferred |
| `test_lookahead_guard_before_training` | Unit | `forward_return` in features → `LookaheadBiasError` |
| `test_nan_after_imputation_raises` | Unit | Residual NaN → `TrainingDataError` |
| `test_random_seed_reproducibility` | Unit | Two runs same config → identical val metrics |
| `test_artifact_contents` | Unit | All required files present in production artifact |
| `test_mlflow_run_logged` | Integration | Mock training; assert MLflow run created with params + metrics |
| `test_parallel_training_returns_all_three` | Integration | All three model results returned per fold |
| `test_full_harness_mock` | Integration | 2 folds, mock models; assert fold loop, selection, artifact write |

---

## 15. Future Increment — Transfer Learning

In the next planned increment:
- Freeze lower layers of the winning global model (pattern recognition)
- Fine-tune top layers per ticker using recent ticker-specific history
- Per-ticker artifacts stored separately under `models/production/per_ticker/{ticker}/`
- This spec remains unchanged — transfer learning is additive
