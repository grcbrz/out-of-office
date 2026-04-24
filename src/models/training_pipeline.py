from __future__ import annotations

import logging
import random
from pathlib import Path

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
from src.monitoring.persistence import save_monitoring_reference

logger = logging.getLogger(__name__)

_PRODUCTION_DIR = Path("models/production")
_FOLD_ARTIFACT_DIR = Path("models/folds")
_MODEL_CONFIG_DIR = Path("configs/models")
_MIN_FOLDS = 3

_INT_TO_LABEL = {0: "SELL", 1: "HOLD", 2: "BUY"}


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


class TrainingPipeline:
    """Orchestrates walk-forward training across N-HiTS, PatchTST, and Autoformer."""

    def __init__(
        self,
        train_window: int = 252,
        step_size: int = 21,
        random_seed: int = 42,
        production_dir: Path = _PRODUCTION_DIR,
        fold_artifact_dir: Path = _FOLD_ARTIFACT_DIR,
    ) -> None:
        self._train_window = train_window
        self._step_size = step_size
        self._random_seed = random_seed
        self._production_dir = production_dir
        self._fold_artifact_dir = fold_artifact_dir

    def run(self, global_df: pd.DataFrame) -> None:
        """Run full walk-forward training harness on the global feature dataset."""
        _set_global_seeds(self._random_seed)
        lookahead_bias_guard(global_df)

        preparer = DataPreparer()
        global_df = preparer.encode_tickers(global_df)

        folds = generate_folds(global_df, self._train_window, self._step_size)
        if len(folds) < _MIN_FOLDS:
            raise ValueError(f"only {len(folds)} folds available, minimum is {_MIN_FOLDS}")

        fold_metrics: list[dict] = []
        final_winner: ModelResult | None = None
        final_fold: Fold | None = None
        final_preparer: DataPreparer | None = None
        final_metrics: dict | None = None

        for fold in folds:
            preparer_copy = DataPreparer()
            preparer_copy._ticker_map = dict(preparer._ticker_map)
            preparer_copy.fit_imputation(fold.train)

            train_df = preparer_copy.apply_imputation(fold.train)
            val_df = preparer_copy.apply_imputation(fold.val)

            class_weights = preparer_copy.compute_class_weights(fold.train["target"])

            results = self._train_all_models(
                train_df, val_df, class_weights, fold.index, self._fold_artifact_dir,
            )
            winner = select_winner(results)
            logger.info("fold %d winner: %s (f1=%.3f)", fold.index, winner.model_name, winner.f1_macro)

            metrics = self._compute_fold_metrics(fold.index, winner, val_df)
            fold_metrics.append(metrics)

            if fold.is_final:
                final_winner = winner
                final_fold = fold
                final_preparer = preparer_copy
                final_metrics = metrics

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
            )

            # Log training metadata to MLflow
            try:
                import mlflow
                mlflow.log_param("train_size", train_size)
                mlflow.log_param("train_window_days", self._train_window)
                mlflow.log_param("step_size_days", self._step_size)
                mlflow.log_param("n_folds", len(folds))
            except ImportError:
                pass

    def _train_all_models(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        class_weights: dict,
        fold_index: int,
        artifact_dir: Path,
    ) -> list[ModelResult]:
        """Train three models sequentially, return results."""
        results: list[ModelResult] = []
        for name in ["nhits", "patchtst", "autoformer"]:
            try:
                result = _train_one_model(name, train_df, val_df, class_weights, fold_index, artifact_dir)
                logger.info("fold %d %s f1=%.3f", fold_index, name, result.f1_macro)
                results.append(result)
            except Exception as exc:
                logger.error("fold %d %s failed: %s", fold_index, name, exc)
                results.append(ModelResult(model_name=name, f1_macro=0.0))
        return results

    def _compute_fold_metrics(
        self, fold_index: int, winner: ModelResult, val_df: pd.DataFrame,
    ) -> dict:
        """Build per-fold metrics dict from the winner's predictions on val_df."""
        val_mask = val_df["target"].notna()
        val_subset = val_df.loc[val_mask].reset_index(drop=True)
        if winner.predictions is None or len(val_subset) == 0:
            logger.warning("fold %d: cannot compute metrics (no labelled rows)", fold_index)
            return {"fold": fold_index, "model_name": winner.model_name, "f1_macro": winner.f1_macro}

        y_pred = pd.Series(winner.predictions).map(_INT_TO_LABEL)
        classification = compute_classification_metrics(val_subset["target"], y_pred)

        if "forward_return" in val_subset.columns:
            financial = compute_financial_metrics(y_pred, val_subset["forward_return"])
        else:
            financial = {"hit_rate": None, "sharpe_ratio": None, "max_drawdown": None, "signal_distribution": {}}

        signal_dist = financial.get("signal_distribution", {})
        signal_counts = y_pred.value_counts().to_dict()
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
        }


def _load_model_config(model_name: str) -> dict:
    path = _MODEL_CONFIG_DIR / f"{model_name}.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _instantiate_wrapper(model_name: str, config: dict):
    if model_name == "nhits":
        from src.models.architectures.nhits import NHiTSWrapper
        return NHiTSWrapper(config)
    if model_name == "patchtst":
        from src.models.architectures.patchtst import PatchTSTWrapper
        return PatchTSTWrapper(config)
    if model_name == "autoformer":
        from src.models.architectures.autoformer import AutoformerWrapper
        return AutoformerWrapper(config)
    raise ValueError(f"unknown model: {model_name}")


def _train_one_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    class_weights: dict,
    fold_index: int,
    artifact_dir: Path,
) -> ModelResult:
    """Train a single wrapper, evaluate on val_df, persist artifact, return ModelResult."""
    _set_global_seeds(42)

    config = _load_model_config(model_name)
    wrapper = _instantiate_wrapper(model_name, config)
    wrapper.train(train_df, train_df["target"], class_weights)

    val_mask = val_df["target"].notna()
    if val_mask.sum() == 0:
        logger.warning("fold %d %s: no labelled validation rows", fold_index, model_name)
        return ModelResult(model_name=model_name, f1_macro=0.0)

    val_subset = val_df.loc[val_mask].reset_index(drop=True)
    predictions = wrapper.predict(val_subset)
    y_pred_str = pd.Series(predictions).map(_INT_TO_LABEL)
    metrics = compute_classification_metrics(val_subset["target"], y_pred_str)
    f1 = float(metrics["f1_macro"])

    fold_artifact = artifact_dir / f"fold{fold_index}_{model_name}"
    wrapper.save(fold_artifact)

    return ModelResult(
        model_name=model_name,
        f1_macro=f1,
        artifact_path=str(fold_artifact),
        predictions=predictions,
    )
