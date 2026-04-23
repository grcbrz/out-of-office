from __future__ import annotations

import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.audit import lookahead_bias_guard
from src.models.harness import Fold, generate_folds
from src.models.persistence import save_artifact
from src.models.preparation import DataPreparer
from src.models.selector import ModelResult, select_winner

logger = logging.getLogger(__name__)

_PRODUCTION_DIR = Path("models/production")
_MIN_FOLDS = 3


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
        n_workers: int = 3,
    ) -> None:
        self._train_window = train_window
        self._step_size = step_size
        self._random_seed = random_seed
        self._production_dir = production_dir
        self._n_workers = n_workers

    def run(self, global_df: pd.DataFrame) -> None:
        """Run full walk-forward training harness on the global feature dataset."""
        _set_global_seeds(self._random_seed)
        lookahead_bias_guard(global_df)

        preparer = DataPreparer()
        global_df = preparer.encode_tickers(global_df)

        folds = generate_folds(global_df, self._train_window, self._step_size)
        if len(folds) < _MIN_FOLDS:
            raise ValueError(f"only {len(folds)} folds available, minimum is {_MIN_FOLDS}")

        final_winner: ModelResult | None = None
        final_fold: Fold | None = None
        final_preparer: DataPreparer | None = None

        for fold in folds:
            preparer_copy = DataPreparer()
            preparer_copy._ticker_map = dict(preparer._ticker_map)
            preparer_copy.fit_imputation(fold.train)

            train_df = preparer_copy.apply_imputation(fold.train)
            val_df = preparer_copy.apply_imputation(fold.val)

            class_weights = preparer_copy.compute_class_weights(fold.train["target"])

            results = self._train_all_models(train_df, val_df, class_weights, fold.index)
            winner = select_winner(results)
            logger.info("fold %d winner: %s (f1=%.3f)", fold.index, winner.model_name, winner.f1_macro)

            if fold.is_final:
                final_winner = winner
                final_fold = fold
                final_preparer = preparer_copy

        if final_winner and final_preparer:
            save_artifact(
                model_name=final_winner.model_name,
                model_dir=Path(final_winner.artifact_path or "."),
                imputation_params=final_preparer.get_imputation_params(),
                ticker_map=final_preparer._ticker_map,
                class_weights={str(k): v for k, v in
                               final_preparer.compute_class_weights(final_fold.train["target"]).items()},
                metadata={
                    "model_name": final_winner.model_name,
                    "f1_macro": final_winner.f1_macro,
                    "random_seed": self._random_seed,
                },
                production_dir=self._production_dir,
            )

    def _train_all_models(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        class_weights: dict,
        fold_index: int,
    ) -> list[ModelResult]:
        """Train three models in parallel processes, return results."""
        model_names = ["nhits", "patchtst", "autoformer"]
        results: list[ModelResult] = []

        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            futures = {
                executor.submit(
                    _train_one_model, name, train_df, val_df, class_weights, fold_index
                ): name
                for name in model_names
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info("fold %d %s f1=%.3f", fold_index, name, result.f1_macro)
                except Exception as exc:
                    logger.error("fold %d %s failed: %s", fold_index, name, exc)
                    results.append(ModelResult(model_name=name, f1_macro=0.0))

        return results


def _train_one_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    class_weights: dict,
    fold_index: int,
) -> ModelResult:
    """Train a single model in a subprocess. Returns ModelResult with mock f1_macro."""
    _set_global_seeds(42)
    # In a real run this would train the model and evaluate on val_df.
    # For now returns a mock f1 so the harness/selector logic is fully testable
    # without requiring heavy GPU training in the test suite.
    mock_f1 = {"nhits": 0.40, "patchtst": 0.38, "autoformer": 0.35}.get(model_name, 0.0)
    return ModelResult(model_name=model_name, f1_macro=mock_f1, artifact_path=None)
