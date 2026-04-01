from __future__ import annotations

from pathlib import Path
import pandas as pd


def safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Prefer parquet; fallback to CSV if parquet engine is missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False)