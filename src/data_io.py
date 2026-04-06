from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def resolve_existing_path(p: Path, *, project_root: Path) -> Path:
    """
    Resolve a data path robustly if notebook/script working directory differs.
    """
    if p.exists():
        return p

    candidates = [
        project_root / p,
        project_root / "data" / "data_raw" / p.name,
        project_root.parent / "data" / "data_raw" / p.name,
        project_root / "data_raw" / p.name,
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Data file not found: {p} (also tried: {[str(x) for x in candidates]})"
    )


def load_binance_spot_1d_csv(path: Path, *, project_root: Path, logger=None) -> pd.DataFrame:
    """
    Load Binance spot OHLCV csv with columns:
      datetime_utc, open, high, low, close, volume, ...
    Output: DataFrame with columns [Open, High, Low, Close, Volume] float
    Index: tz-naive DatetimeIndex (for backtesting.py)
    """
    path = resolve_existing_path(path, project_root=project_root)
    df = pd.read_csv(path)

    if "datetime_utc" not in df.columns:
        raise ValueError(
            f"Expected 'datetime_utc' in {path}. Columns: {list(df.columns)[:20]}"
        )

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.sort_values("datetime_utc").drop_duplicates("datetime_utc", keep="last")
    df = df.set_index("datetime_utc")

    rename = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    df = df.rename(columns=rename)

    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            raise ValueError(
                f"Missing column {c} in {path}. Available: {list(df.columns)[:30]}"
            )

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    out = df[ohlcv_cols].astype(float).copy()

    # Keep additional numeric columns (for example on-chain features) after OHLCV.
    # This is backward compatible: for pure OHLCV CSVs, extras is empty.
    extra_cols = [c for c in df.columns if c not in ohlcv_cols]
    if extra_cols:
        extra = df[extra_cols].apply(pd.to_numeric, errors="coerce")
        out = pd.concat([out, extra], axis=1)

    # backtesting.py prefers timezone-naive index
    out.index = out.index.tz_convert(None)

    # Minimal OHLC sanity checks
    hi_should = out[["Open", "Close", "Low"]].max(axis=1)
    lo_should = out[["Open", "Close", "High"]].min(axis=1)

    bad_high = (out["High"] < hi_should)
    bad_low = (out["Low"] > lo_should)

    n_bad_h = int(bad_high.sum())
    n_bad_l = int(bad_low.sum())

    if (n_bad_h or n_bad_l) and logger is not None:
        logger.warning(
            "OHLC issues in %s: bad_high=%d, bad_low=%d. Fixing by clamping High/Low.",
            path.name, n_bad_h, n_bad_l
        )

    if n_bad_h:
        out.loc[bad_high, "High"] = hi_should.loc[bad_high]
    if n_bad_l:
        out.loc[bad_low, "Low"] = lo_should.loc[bad_low]

    if (out[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError(f"Non-positive prices detected in {path}. Please inspect the data source.")

    return out


def load_all_symbols(
    data_paths: Dict[str, Path],
    *,
    project_root: Path,
    logger=None,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sym, p in data_paths.items():
        df = load_binance_spot_1d_csv(p, project_root=project_root, logger=logger)
        out[sym] = df
        if logger is not None:
            logger.info(
                "%s loaded: n=%d | %s .. %s",
                sym, len(df), df.index.min().date(), df.index.max().date()
            )
    return out
