# Spec 07 — Monitoring & Retraining

**Status:** Draft
**Created:** 2026-04-23
**Depends on:** Spec 06 (Serving)
**Consumed by:** None (terminal spec)

---

## 1. Problem Statement

Monitor the production model for feature drift, prediction drift, and performance degradation on a nightly basis. Raise structured alerts to file when thresholds are breached. Trigger retraining automatically when drift or degradation crosses a defined threshold, replacing the production artifact if the new model passes quality gates. This closes the feedback loop of the nightly batch pipeline.

---

## 2. Scope

**In scope:**
- Feature drift detection per feature (KS test + PSI as programmatic gates)
- Prediction drift detection (signal distribution shift)
- Performance degradation detection (directional hit rate on realised returns)
- Evidently AI reports for human-readable drift summaries
- Structured alert file on threshold breach
- Automatic retraining trigger when drift or degradation gates fail
- Monitoring metadata persistence (CSV + JSON)
- Integration into nightly batch sequence

**Out of scope:**
- External alerting (email, Slack, push notifications)
- Dashboard visualisation (future increment)
- A/B testing between model versions
- Online learning or partial model updates

---

## 3. Reference Window

All drift detection compares a **current window** against a **reference window**:

| Window | Definition |
|---|---|
| Reference | Training data of the current production model (last 252 trading days at training time) |
| Current | Most recent 21 trading days of feature data (one retraining cycle) |

Reference window statistics (mean, std, distribution) are computed at training time and stored in the production artifact (`monitoring_reference.json`). They are not recomputed nightly — only the current window is computed fresh.

---

## 4. Feature Drift Detection

### 4.1 Programmatic Gates (Statistical Tests)

Run nightly for each feature in the model input set.

**KS Test (Kolmogorov-Smirnov):**
- Compares current window distribution against reference distribution
- `p-value < 0.05` → drift detected for that feature

**PSI (Population Stability Index):**
- Measures magnitude of distribution shift
- `PSI < 0.10` → no drift
- `0.10 ≤ PSI < 0.20` → moderate drift (warning)
- `PSI ≥ 0.20` → significant drift (alert + retraining trigger candidate)

**Gate logic:**

```
feature_drift_triggered = any feature where:
    ks_pvalue < 0.05 AND psi >= 0.20
```

Both conditions must be met to avoid false positives from the KS test alone on large samples.

**Applied to:** All numeric features. Boolean and categorical features (e.g. `day_of_week`, `sentiment_available`) monitored via frequency distribution shift (chi-squared test, threshold `p < 0.05`).

### 4.2 Evidently Report

Generated weekly (every 7 nightly runs) — not nightly, to avoid report volume overhead.

- `evidently.DataDriftPreset` across all features
- Persisted to `reports/monitoring/{date}/data_drift.html`
- Also saved as JSON for programmatic access: `reports/monitoring/{date}/data_drift.json`

---

## 5. Prediction Drift Detection

Monitor the distribution of BUY/HOLD/SELL signals over the current 21-day window vs. the reference distribution logged at evaluation time (Spec 05 `metrics_aggregated.csv`).

**Method:** Chi-squared test on signal counts.
**Threshold:** `p-value < 0.05` → prediction drift detected.

Additionally flag if any single class exceeds 80% of predictions in the current window (degenerate model signal — same gate as Spec 05 quality gate, applied continuously).

---

## 6. Performance Degradation Detection

Measures whether the model's directional signals are still profitable on realised returns.

**Inputs:**
- `data/predictions/{date}.csv` — signals from the past 21 trading days
- `data/raw/ohlcv/{ticker}/{date}.csv` — actual next-day close prices

**Metric:** Rolling hit rate over 21 trading days (BUY/SELL signals only, excluding HOLD).

```
hit_rate = correct_direction_signals / total_buy_sell_signals
```

**Threshold:** `hit_rate < 0.45` for two consecutive 21-day windows → degradation alert + retraining trigger.

Two consecutive windows required to avoid triggering on a single bad month.

---

## 7. Retraining Trigger Logic

Retraining is triggered (in addition to the scheduled 21-day cadence from Spec 04) when any of the following conditions are met:

| Condition | Gate |
|---|---|
| Feature drift | Any feature: `ks_pvalue < 0.05` AND `psi >= 0.20` |
| Prediction drift | Chi-squared `p < 0.05` on signal distribution |
| Performance degradation | Hit rate < 0.45 for 2 consecutive windows |

**Trigger behaviour:**
1. Log alert to `data/monitoring/alerts/{date}.json`
2. Set `retraining_required: true` in `data/monitoring/status.json`
3. Nightly batch checks `status.json` at startup — if `retraining_required: true`, runs full training + evaluation regardless of the 21-day cadence
4. After successful retraining and quality gate pass, reset `retraining_required: false`
5. If quality gate fails after drift-triggered retraining, alert is logged and `retraining_required` remains `true` for the next night

---

## 8. Alert File Structure

Written to `data/monitoring/alerts/{date}.json` only when at least one gate is breached:

```json
{
  "alert_date": "2026-04-24",
  "retraining_triggered": true,
  "feature_drift": {
    "triggered": true,
    "features": [
      {
        "feature": "close_zscore",
        "ks_pvalue": 0.003,
        "psi": 0.24,
        "severity": "significant"
      }
    ]
  },
  "prediction_drift": {
    "triggered": false,
    "chi2_pvalue": 0.18,
    "current_distribution": {"BUY": 0.29, "HOLD": 0.41, "SELL": 0.30}
  },
  "performance_degradation": {
    "triggered": false,
    "hit_rate_current_window": 0.52,
    "consecutive_windows_below_threshold": 0
  }
}
```

---

## 9. Monitoring Status File

`data/monitoring/status.json` — updated nightly, read by the batch orchestrator:

```json
{
  "last_monitored": "2026-04-24",
  "retraining_required": false,
  "consecutive_degradation_windows": 0,
  "last_retraining_trigger": null,
  "last_retraining_trigger_reason": null
}
```

---

## 10. Monitoring Metadata Persistence

Nightly metrics appended to `reports/monitoring/monitoring_history.csv`:

| Column | Notes |
|---|---|
| `date` | |
| `n_features_drifted` | count of features breaching both KS + PSI gates |
| `max_psi` | highest PSI across all features |
| `prediction_drift_pvalue` | chi-squared p-value |
| `hit_rate_21d` | rolling 21-day hit rate |
| `retraining_triggered` | bool |
| `trigger_reason` | feature_drift / prediction_drift / performance_degradation / null |

---

## 11. Nightly Batch Integration

Updated nightly batch sequence:

```
scripts/run_nightly.py --start-date {date}
├── IngestionPipeline.run()
├── PreprocessingPipeline.run()
├── FeaturePipeline.run()
├── MonitoringPipeline.run()           ← NEW — runs before training decision
│   ├── feature drift detection
│   ├── prediction drift detection
│   ├── performance degradation check
│   ├── write alert file if triggered
│   └── update status.json
├── TrainingPipeline.run()             ← runs if: retraining day OR retraining_required=true
├── EvaluationPipeline.run()           ← runs if training ran
└── PredictionClient.run()             ← always runs
```

---

## 12. Module Structure

```
src/monitoring/
├── __init__.py
├── drift/
│   ├── __init__.py
│   ├── feature_drift.py      # FeatureDriftDetector — KS test + PSI per feature
│   ├── prediction_drift.py   # PredictionDriftDetector — chi-squared on signal distribution
│   └── evidently_report.py   # EvidentlyReporter — HTML + JSON report (weekly)
├── degradation.py            # DegradationDetector — rolling hit rate vs realised returns
├── trigger.py                # RetrainingTrigger — gate evaluation + status.json update
├── alerts.py                 # AlertWriter — structured alert JSON
├── persistence.py            # append_monitoring_csv, update_status
└── pipeline.py               # MonitoringPipeline — orchestrates full monitoring run
```

---

## 13. Configuration

```yaml
# configs/monitoring.yaml
reference_window_days: 252      # matches training window
current_window_days: 21         # matches retraining step

feature_drift:
  ks_pvalue_threshold: 0.05
  psi_warning_threshold: 0.10
  psi_alert_threshold: 0.20

prediction_drift:
  chi2_pvalue_threshold: 0.05
  max_signal_concentration: 0.80

performance_degradation:
  hit_rate_threshold: 0.45
  consecutive_windows_required: 2

evidently:
  report_frequency_days: 7
  output_dir: reports/monitoring

alerts:
  output_dir: data/monitoring/alerts

monitoring_history:
  output_path: reports/monitoring/monitoring_history.csv
```

---

## 14. Acceptance Criteria

- [ ] KS test and PSI computed nightly per numeric feature against reference window stats from artifact
- [ ] Feature drift gate: both `ks_pvalue < 0.05` AND `psi >= 0.20` required to trigger
- [ ] Chi-squared test computed nightly on signal distribution; triggers on `p < 0.05`
- [ ] Degenerate signal flag triggers if any class > 80% of predictions in current window
- [ ] Hit rate computed from prediction CSV matched against realised OHLCV next-day returns
- [ ] Performance degradation requires 2 consecutive windows below threshold before triggering
- [ ] Alert JSON written only when at least one gate is breached
- [ ] `status.json` updated nightly; `retraining_required` set correctly
- [ ] Nightly batch reads `status.json` and triggers training if `retraining_required: true`
- [ ] After successful retraining + quality gate pass, `retraining_required` reset to `false`
- [ ] If post-drift retraining fails quality gates, `retraining_required` remains `true`
- [ ] Evidently HTML + JSON report generated every 7 nightly runs
- [ ] Monitoring metrics appended to `monitoring_history.csv` on every run
- [ ] All thresholds sourced from `configs/monitoring.yaml`; no hardcoded values

---

## 15. Tests Required

| Test | Type | Notes |
|---|---|---|
| `test_ks_test_detects_drift` | Unit | Synthetic distributions with known shift; assert p < 0.05 |
| `test_ks_test_no_drift` | Unit | Same distribution; assert p > 0.05 |
| `test_psi_no_drift` | Unit | PSI < 0.10 for stable distribution |
| `test_psi_moderate_drift` | Unit | 0.10 ≤ PSI < 0.20; warning only |
| `test_psi_significant_drift` | Unit | PSI ≥ 0.20; alert triggered |
| `test_feature_drift_gate_requires_both` | Unit | KS fails but PSI < 0.20 → no trigger |
| `test_prediction_drift_chi2` | Unit | Known shift in signal counts; assert trigger |
| `test_degenerate_signal_flag` | Unit | 85% HOLD → concentration flag |
| `test_hit_rate_computation` | Unit | Known predictions + returns; assert correct hit rate |
| `test_degradation_single_window` | Unit | 1 window below threshold → no trigger |
| `test_degradation_two_windows` | Unit | 2 consecutive windows below threshold → trigger |
| `test_alert_written_on_breach` | Unit | Any gate breach → alert file written |
| `test_no_alert_on_clean_run` | Unit | All gates pass → no alert file |
| `test_status_json_updated` | Unit | After trigger; `retraining_required: true` |
| `test_status_reset_after_retraining` | Unit | Successful retrain; `retraining_required: false` |
| `test_status_persists_on_failed_retrain` | Unit | Failed quality gate; flag remains `true` |
| `test_evidently_report_frequency` | Unit | Report generated on day 7, not days 1–6 |
| `test_monitoring_history_appended` | Unit | Two runs; two rows in CSV |
| `test_full_monitoring_pipeline` | Integration | Mock feature + prediction data; assert all outputs |
| `test_nightly_batch_triggers_training_on_flag` | Integration | `retraining_required: true` in status; assert training called |
