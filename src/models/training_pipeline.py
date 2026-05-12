from __future__ import annotations

import logging
import random
from pathlib import Path
import os
import gc

import numpy as np
import pandas as pd
import yaml

from src.evaluation.aggregation import aggregate_across_folds
from src.evaluation.classification import compute_classification_metrics
from src.evaluation.financial import compute_financial_metrics
from src.features.audit import lookahead_bias_guard
from src.models.harness import Fold, generate_folds
from src.models.persistence import save_artifact
from src.models.preparation import DataPreparer
from src.models.selector import ModelResult, select_winner
from src.models.threshold import (
    DEFAULT_CANDIDATE_TAUS,
    apply_confidence_threshold,
    calibrate_threshold,
)
from src.monitoring.persistence import save_monitoring_reference

# CRITICAL: Memory handling
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

logger = logging.getLogger(__name__)

_PRODUCTION_DIR = Path("models/production")
_FOLD_ARTIFACT_DIR = Path("models/folds")
_MODEL_CONFIG_DIR = Path("configs/models")
_MIN_FOLDS = 3

_INT_TO_LABEL = {0: "SELL", 1: "HOLD", 2: "BUY"}

# Production candidates trained per fold. Currently:
#   * lightgbm     — gradient-boosted trees
#   * randomforest — bagged trees (uncorrelated failure modes vs boosting)
# Add real neuralforecast architectures here as they land — the harness
# already supports N candidates and selects on F1-macro.
_CANDIDATE_NAMES: tuple[str, ...] = ("lightgbm", "randomforest")
_BASELINE_NAME = "baseline_last_direction"


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


class TrainingPipeline:
    """Walk-forward training across the registered candidates plus a naive baseline.

    v1 ships with a single candidate (``lightgbm``) and the baseline. Add
    candidates by appending names to ``_CANDIDATE_NAMES`` and branching in
    ``_instantiate_wrapper``.
    """

    def __init__(
        self,
        train_window: int = 120,
        step_size: int = 21,
        random_seed: int = 42,
        weight_half_life_days: int = 0,
        production_dir: Path = _PRODUCTION_DIR,
        fold_artifact_dir: Path = _FOLD_ARTIFACT_DIR,
    ) -> None:
        self._train_window = train_window
        self._step_size = step_size
        self._random_seed = random_seed
        self._weight_half_life_days = weight_half_life_days
        self._production_dir = production_dir
        self._fold_artifact_dir = fold_artifact_dir

    def run(self, global_df: pd.DataFrame) -> None:
        """Run full walk-forward training harness on the global feature dataset."""
        # Limit threads to prevent segfault
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        _set_global_seeds(self._random_seed)
        lookahead_bias_guard(global_df)

        preparer = DataPreparer()
        global_df = preparer.encode_tickers(global_df)

        folds = generate_folds(global_df, self._train_window, self._step_size)
        if len(folds) < _MIN_FOLDS:
            raise ValueError(f"only {len(folds)} folds available, minimum is {_MIN_FOLDS}")

        fold_metrics: list[dict] = []
        # Per-model F1 history across folds for aggregate selection.
        model_f1_history: dict[str, list[float]] = {
            name: [] for name in (*_CANDIDATE_NAMES, _BASELINE_NAME)
        }
        # Final-fold results — used to pick the artifact of the aggregate winner.
        final_fold_results: dict[str, ModelResult] = {}
        final_fold: Fold | None = None
        final_preparer: DataPreparer | None = None

        for fold in folds:
            preparer_copy = DataPreparer()
            preparer_copy._ticker_map = dict(preparer._ticker_map)
            preparer_copy.fit_imputation(fold.train)

            train_df = preparer_copy.apply_imputation(fold.train)
            val_df = preparer_copy.apply_imputation(fold.val)

            class_weights = preparer_copy.compute_class_weights(fold.train["target"])

            results = self._train_all_models(
                train_df, val_df, class_weights, fold.index, self._fold_artifact_dir,
                self._weight_half_life_days,
            )
            fold_winner = select_winner(results)
            baseline_result = next((r for r in results if r.is_baseline), None)
            baseline_f1 = baseline_result.f1_macro if baseline_result else None
            logger.info(
                "fold %d winner: %s (f1=%.3f) | baseline f1=%s",
                fold.index, fold_winner.model_name, fold_winner.f1_macro,
                f"{baseline_f1:.3f}" if baseline_f1 is not None else "n/a",
            )

            for r in results:
                model_f1_history[r.model_name].append(r.f1_macro)

            metrics = self._compute_fold_metrics(
                fold.index, fold_winner, val_df, baseline_f1=baseline_f1,
            )
            fold_metrics.append(metrics)

            if fold.is_final:
                final_fold_results = {r.model_name: r for r in results}
                final_fold = fold
                final_preparer = preparer_copy

             # Clean up after each fold to prevent memory buildup
            del train_df, val_df, results
            gc.collect()

        # Select production model by mean F1 across folds, ignoring the baseline.
        mean_f1 = {
            name: (sum(scores) / len(scores) if scores else 0.0)
            for name, scores in model_f1_history.items()
        }
        for name in (*_CANDIDATE_NAMES, _BASELINE_NAME):
            logger.info("mean F1 across folds — %s: %.3f", name, mean_f1[name])

        aggregate_winner = select_winner([
            ModelResult(model_name=name, f1_macro=f1)
            for name, f1 in mean_f1.items()
            if name in _CANDIDATE_NAMES
        ])
        final_winner = final_fold_results.get(aggregate_winner.model_name)
        final_baseline = final_fold_results.get(_BASELINE_NAME)
        final_metrics = (
            self._compute_fold_metrics(
                final_fold.index, final_winner, final_fold.val,
                baseline_f1=final_baseline.f1_macro if final_baseline else None,
            )
            if final_winner and final_fold is not None else None
        )
        logger.info(
            "aggregate winner: %s (mean f1=%.3f, baseline mean f1=%.3f)",
            aggregate_winner.model_name, aggregate_winner.f1_macro,
            mean_f1[_BASELINE_NAME],
        )

        if final_winner and final_preparer and final_metrics and final_fold:
            aggregated = aggregate_across_folds(fold_metrics)
            save_artifact(
                model_name=final_winner.model_name,
                model_dir=Path(final_winner.artifact_path or "."),
                imputation_params=final_preparer.get_imputation_params(),
                ticker_map=final_preparer._ticker_map,
                class_weights={
                    str(k): v for k, v in
                    final_preparer.compute_class_weights(final_fold.train["target"]).items()
                },
                metadata={
                    "model_name": final_winner.model_name,
                    "random_seed": self._random_seed,
                    "fold_metrics": fold_metrics,
                    "aggregated": aggregated,
                    "production_fold": final_metrics,
                    # Production τ — what serving must apply on every prediction.
                    # Pulled to the top of metadata so ArtifactLoader doesn't
                    # depend on the shape of production_fold.
                    "confidence_threshold": (
                        float(final_winner.confidence_threshold)
                        if final_winner.confidence_threshold is not None else None
                    ),
                    "baseline": {
                        "name": _BASELINE_NAME,
                        "mean_f1_macro": float(mean_f1[_BASELINE_NAME]),
                        "production_fold_f1_macro": (
                            float(final_baseline.f1_macro) if final_baseline else None
                        ),
                    },
                },
                production_dir=self._production_dir,
            )
            train_dates = pd.to_datetime(final_fold.train["date"])
            train_size = len(final_fold.train)
            save_monitoring_reference(
                artifact_dir=self._production_dir / final_winner.model_name,
                training_start=train_dates.min().date(),
                training_end=train_dates.max().date(),
                signal_counts={
                    "BUY": final_metrics.get("buy_count", 0),
                    "HOLD": final_metrics.get("hold_count", 0),
                    "SELL": final_metrics.get("sell_count", 0),
                },
                tickers=sorted(final_preparer._ticker_map.keys()),
            )

            try:
                import mlflow
                mlflow.log_param("train_size", train_size)
                mlflow.log_param("train_window_days", self._train_window)
                mlflow.log_param("step_size_days", self._step_size)
                mlflow.log_param("n_folds", len(folds))
                mlflow.log_metric("baseline_mean_f1_macro", float(mean_f1[_BASELINE_NAME]))
                if final_baseline is not None:
                    mlflow.log_metric(
                        "baseline_production_fold_f1_macro", float(final_baseline.f1_macro)
                    )
                if final_winner.confidence_threshold is not None:
                    mlflow.log_metric(
                        "production_confidence_threshold",
                        float(final_winner.confidence_threshold),
                    )
            except ImportError:
                pass

    # ------------------------------------------------------------------

    def _train_all_models(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        class_weights: dict,
        fold_index: int,
        artifact_dir: Path,
        weight_half_life_days: int = 0,
    ) -> list[ModelResult]:
        """Train baseline + all candidate architectures, return their ModelResults."""
        results: list[ModelResult] = []
        for name in (*_CANDIDATE_NAMES, _BASELINE_NAME):
            try:
                result = _train_one_model(
                    name, train_df, val_df, class_weights, fold_index, artifact_dir,
                    weight_half_life_days=weight_half_life_days,
                )
                logger.info("fold %d %s f1=%.3f", fold_index, name, result.f1_macro)
                results.append(result)
            except Exception as exc:
                logger.error("fold %d %s failed: %s", fold_index, name, exc)
                results.append(ModelResult(
                    model_name=name, f1_macro=0.0,
                    is_baseline=(name == _BASELINE_NAME),
                ))
        return results

    def _compute_fold_metrics(
        self,
        fold_index: int,
        winner: ModelResult,
        val_df: pd.DataFrame,
        baseline_f1: float | None = None,
    ) -> dict:
        """Build per-fold metrics dict from the winner's predictions on val_df.

        Classification metrics (F1, MCC, ROC-AUC) are computed on the **raw**
        predictions because thresholding collapses the confusion matrix toward
        HOLD and would bias model selection toward strategies that simply stop
        trading. Financial metrics (Sharpe, MDD, hit rate, signal distribution,
        signal counts) are computed on the **threshold-applied** predictions
        because that is what production trades on.
        """
        val_mask = val_df["target"].notna()
        val_subset = val_df.loc[val_mask].reset_index(drop=True)
        if winner.predictions is None or len(val_subset) == 0:
            logger.warning("fold %d: cannot compute metrics (no labelled rows)", fold_index)
            return {
                "fold": fold_index,
                "model_name": winner.model_name,
                "f1_macro": winner.f1_macro,
                "baseline_f1_macro": baseline_f1,
                "confidence_threshold": winner.confidence_threshold,
            }

        # Classification metrics — raw predictions.
        y_pred_raw = pd.Series(winner.predictions).map(_INT_TO_LABEL)
        classification = compute_classification_metrics(
            val_subset["target"], y_pred_raw,
            y_proba=winner.predictions_proba,
            proba_classes=winner.proba_classes,
        )

        # Financial metrics — thresholded predictions when τ + proba available.
        thresholded_predictions = _maybe_apply_threshold(
            winner.predictions, winner.predictions_proba, winner.confidence_threshold,
        )
        y_pred_traded = pd.Series(thresholded_predictions).map(_INT_TO_LABEL)
        if "forward_return" in val_subset.columns:
            financial = compute_financial_metrics(y_pred_traded, val_subset["forward_return"])
        else:
            financial = {
                "hit_rate": None, "sharpe_ratio": None, "max_drawdown": None,
                "signal_distribution": {},
            }

        signal_dist = financial.get("signal_distribution", {})
        signal_counts = y_pred_traded.value_counts().to_dict()
        return {
            "fold": fold_index,
            "model_name": winner.model_name,
            "f1_macro": float(classification["f1_macro"]),
            "mcc": float(classification["mcc"]),
            "roc_auc": classification.get("roc_auc"),
            "hit_rate": financial["hit_rate"],
            "sharpe_ratio": financial["sharpe_ratio"],
            "max_drawdown": financial["max_drawdown"],
            "signal_distribution": signal_dist,
            "buy_share": float(signal_dist.get("BUY", 0.0)),
            "hold_share": float(signal_dist.get("HOLD", 0.0)),
            "sell_share": float(signal_dist.get("SELL", 0.0)),
            "buy_count": int(signal_counts.get("BUY", 0)),
            "hold_count": int(signal_counts.get("HOLD", 0)),
            "sell_count": int(signal_counts.get("SELL", 0)),
            "baseline_f1_macro": baseline_f1,
            "confidence_threshold": winner.confidence_threshold,
        }


# ----------------------------------------------------------------------
# Module-level helpers — kept module-level so they're easy to mock in tests.
# ----------------------------------------------------------------------

def _load_model_config(model_name: str) -> dict:
    path = _MODEL_CONFIG_DIR / f"{model_name}.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _instantiate_wrapper(model_name: str, config: dict):
    if model_name == "lightgbm":
        from src.models.architectures.lightgbm import LightGBMWrapper
        # Force single-threaded mode
        config = config.copy() if config else {}
        config['n_jobs'] = 1
        config['num_threads'] = 1
        config['verbosity'] = -1  # Reduce output
        return LightGBMWrapper(config)
    if model_name == "randomforest":
        from src.models.architectures.randomforest import RandomForestWrapper
        config = config.copy() if config else {}
        config['n_jobs'] = 1
        return RandomForestWrapper(config)
    if model_name == _BASELINE_NAME:
        from src.models.architectures.baseline import BaselineLastDirectionWrapper
        return BaselineLastDirectionWrapper(config)
    raise ValueError(f"unknown model: {model_name}")


def _extract_proba_classes(wrapper) -> tuple[int, ...] | None:
    """Return the int class IDs in the column order of predict_proba.

    sklearn estimators expose ``classes_``; the baseline wrapper does not need it
    (it always emits the canonical SELL/HOLD/BUY column order).
    """
    if getattr(wrapper, "_model", None) is not None and hasattr(wrapper._model, "classes_"):
        return tuple(int(c) for c in wrapper._model.classes_)
    return None


def _maybe_apply_threshold(
    predictions: np.ndarray,
    predictions_proba: np.ndarray | None,
    tau: float | None,
) -> np.ndarray:
    """Return thresholded predictions if τ and proba are present, else raw.

    Used to keep the metrics path tidy when τ wasn't calibrated (e.g. baseline,
    or no forward_return column in val).
    """
    if predictions_proba is None or tau is None:
        return np.asarray(predictions)
    return apply_confidence_threshold(predictions, predictions_proba, tau)


def _train_one_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    class_weights: dict,
    fold_index: int,
    artifact_dir: Path,
    weight_half_life_days: int = 0,
) -> ModelResult:
    """Train a single wrapper, evaluate on val_df, persist artifact, return ModelResult."""
    _set_global_seeds(42)

    config = _load_model_config(model_name)
    wrapper = _instantiate_wrapper(model_name, config)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    is_baseline = model_name == _BASELINE_NAME
    if weight_half_life_days > 0 and not is_baseline:
        from src.models.time_decay import compute_time_decay_weights
        sample_weight = compute_time_decay_weights(train_df["date"], weight_half_life_days)
    else:
        sample_weight = None

    wrapper.train(train_df, train_df["target"], class_weights, sample_weight=sample_weight)

    val_mask = val_df["target"].notna()
    if val_mask.sum() == 0:
        logger.warning("fold %d %s: no labelled validation rows", fold_index, model_name)
        return ModelResult(model_name=model_name, f1_macro=0.0, is_baseline=is_baseline)

    val_subset = val_df.loc[val_mask].reset_index(drop=True)
    predictions = wrapper.predict(val_subset)
    try:
        predictions_proba = wrapper.predict_proba(val_subset)
    except Exception as exc:
        logger.warning("fold %d %s: predict_proba failed (%s)", fold_index, model_name, exc)
        predictions_proba = None

    proba_classes = _extract_proba_classes(wrapper)

    y_pred_str = pd.Series(predictions).map(_INT_TO_LABEL)
    metrics = compute_classification_metrics(
        val_subset["target"], y_pred_str,
        y_proba=predictions_proba,
        proba_classes=proba_classes,
    )
    f1 = float(metrics["f1_macro"])

    fold_artifact = artifact_dir / f"fold{fold_index}_{model_name}"
    wrapper.save(fold_artifact)

    confidence_threshold = _calibrate_fold_threshold(
        model_name=model_name,
        fold_index=fold_index,
        val_subset=val_subset,
        predictions=predictions,
        predictions_proba=predictions_proba,
    )

    return ModelResult(
        model_name=model_name,
        f1_macro=f1,
        artifact_path=str(fold_artifact),
        predictions=predictions,
        predictions_proba=predictions_proba,
        proba_classes=proba_classes,
        is_baseline=is_baseline,
        confidence_threshold=confidence_threshold,
    )


def _calibrate_fold_threshold(
    model_name: str,
    fold_index: int,
    val_subset: pd.DataFrame,
    predictions: np.ndarray | None,
    predictions_proba: np.ndarray | None,
) -> float | None:
    """Pick the Sharpe-maximising τ on this fold's val set.

    Skipped (returns ``None``) when probabilities are absent, when
    ``forward_return`` is not in the val subset, or when the model is the
    baseline. The baseline emits one-hot probabilities so τ is meaningless
    for it; setting it to ``None`` keeps the harness honest.
    """
    if model_name == _BASELINE_NAME:
        return None
    if predictions is None or predictions_proba is None:
        return None
    if "forward_return" not in val_subset.columns:
        return None
    forward_returns = val_subset["forward_return"].astype(float)
    if forward_returns.isna().all():
        return None

    tau, _ = calibrate_threshold(
        predictions=predictions,
        predictions_proba=predictions_proba,
        forward_log_returns=forward_returns,
        candidate_taus=DEFAULT_CANDIDATE_TAUS,
    )
    logger.info("fold %d %s confidence τ=%.2f", fold_index, model_name, tau)
    return tau
