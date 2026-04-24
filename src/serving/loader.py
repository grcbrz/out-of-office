from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PRODUCTION_DIR = Path("models/production")
_REQUIRED_FILES = ["imputation_params.json", "ticker_map.json", "metadata.json", "model.pkl"]


class ArtifactLoader:
    """Loads the production model artifact from disk at startup.

    If the artifact is missing or incomplete, the server starts in degraded mode.
    """

    def __init__(self, production_dir: Path = _PRODUCTION_DIR) -> None:
        self._production_dir = production_dir
        self.model_name: str | None = None
        self.metadata: dict[str, Any] = {}
        self.imputation_params: dict[str, float] = {}
        self.ticker_map: dict[str, int] = {}
        self.trained_features: list[str] = []
        self.model: Any = None
        self.is_loaded: bool = False

    def load(self) -> None:
        """Attempt to load the production artifact. Sets is_loaded=False on failure."""
        model_dirs = [d for d in self._production_dir.iterdir() if d.is_dir()] if self._production_dir.exists() else []
        if not model_dirs:
            logger.critical("no production artifact found in %s; starting in degraded mode", self._production_dir)
            return

        model_dir = model_dirs[0]
        missing = [f for f in _REQUIRED_FILES if not (model_dir / f).exists()]
        if missing:
            logger.critical("production artifact incomplete, missing: %s", missing)
            return

        self.model_name = model_dir.name
        self.metadata = self._read_json(model_dir / "metadata.json")
        self.imputation_params = self._read_json(model_dir / "imputation_params.json")
        self.ticker_map = self._read_json(model_dir / "ticker_map.json")
        pkl_data = self._load_pickle(model_dir / "model.pkl")
        self.model = pkl_data.get("model") if isinstance(pkl_data, dict) else pkl_data
        self.trained_features = pkl_data.get("features", []) if isinstance(pkl_data, dict) else []
        self.is_loaded = True
        logger.info("loaded production artifact: %s (f1=%.3f)", self.model_name, self.metadata.get("f1_macro", 0))

    @staticmethod
    def _read_json(path: Path) -> dict:
        with path.open() as f:
            return json.load(f)

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        with path.open("rb") as f:
            return pickle.load(f)
