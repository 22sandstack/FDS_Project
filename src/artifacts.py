from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def stable_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def write_json_atomic(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(temporary, index=False)
    os.replace(temporary, path)


class ArtifactStore:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def refit_dir(self, model_id: str, signature: str, test_year: int) -> Path:
        return self.run_dir / "models" / model_id / signature / f"refit_{test_year}"

    def paths(
        self, model_id: str, signature: str, test_year: int, create: bool = True
    ) -> dict[str, Path]:
        directory = self.refit_dir(model_id, signature, test_year)
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return {
            "dir": directory,
            "metadata": directory / "metadata.json",
            "predictions": directory / "predictions.parquet",
            "model": directory / "model.bin",
            "latest": directory / "latest.pt",
            "best": directory / "best.pt",
            "completed": directory / "COMPLETED",
        }

    def is_complete(self, paths: dict[str, Path], signature: str) -> bool:
        if not paths["completed"].exists() or not paths["predictions"].exists():
            return False
        try:
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            return metadata.get("model_signature") == signature
        except Exception:
            return False

    @staticmethod
    def mark_complete(paths: dict[str, Path]) -> None:
        temporary = paths["completed"].with_suffix(".tmp")
        temporary.write_text("complete\n", encoding="utf-8")
        os.replace(temporary, paths["completed"])
