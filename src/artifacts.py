from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def write_json_atomic(
    data: dict,
    path: Path,
) -> None:
    """Atomically write a JSON file."""
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def write_parquet_atomic(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Atomically write a parquet file."""
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_parquet(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        path,
    )


def write_csv_atomic(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Atomically write a CSV file."""
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        path,
    )
