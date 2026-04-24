from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def save_artifact(
    model_name: str,
    model_dir: Path,
    imputation_params: dict,
    ticker_map: dict,
    class_weights: dict,
    metadata: dict,
    production_dir: Path,
) -> None:
    """Write the production artifact for the winning model to models/production/."""
    dest = production_dir / model_name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    _write_json(dest / "imputation_params.json", imputation_params)
    _write_json(dest / "ticker_map.json", ticker_map)
    _write_json(dest / "class_weights.json", {str(k): v for k, v in class_weights.items()})
    _write_json(dest / "metadata.json", metadata)

    # Copy model weights if present (model.pt for torch artifacts, model.pkl for sklearn)
    for fname in ("model.pt", "model.pkl"):
        src = model_dir / fname
        if src.exists():
            shutil.copy(src, dest / fname)

    config_src = model_dir / "config.yaml"
    if config_src.exists():
        shutil.copy(config_src, dest / "config.yaml")

    logger.info("production artifact written to %s", dest)


def load_artifact(model_name: str, production_dir: Path) -> dict:
    """Load metadata and params from the production artifact directory."""
    src = production_dir / model_name
    result = {}
    for fname in ["imputation_params.json", "ticker_map.json", "class_weights.json", "metadata.json"]:
        fpath = src / fname
        if fpath.exists():
            with fpath.open() as f:
                result[fname.replace(".json", "")] = json.load(f)
    return result


def _write_json(path: Path, data: dict) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)
