from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


RAW_NUMERIC_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_base",
    "taker_buy_quote",
]

RAW_INT_COLS = ["trades"]

KEEP_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ret_1bar",
    "logret_1bar",
    "range",
    "true_range",
    "atr_14",
    "vol_30",
]

REQUIRED_AFTER_FEATURES = ["ret_1bar", "atr_14", "vol_30"]


@dataclass(frozen=True)
class CleaningStats:
    rows_before: int
    rows_after_dropna_ohlc: int
    rows_after_feature_dropna: int
    dropped_rows_ohlc_na: int
    dropped_rows_feature_na: int
    datetime_nulls: int
    datetime_duplicates: int
    bad_high_count: int
    bad_low_count: int
    nonpositive_close_count: int


def infer_periods_per_year(interval: str) -> int:
    """
    Infer annualization factor from interval.
    """
    x = interval.strip().lower()
    if x.endswith("d"):
        n = int(x[:-1] or "1")
        return int(round(365 / n))
    if x.endswith("h"):
        n = int(x[:-1] or "1")
        return int(round((24 / n) * 365))
    if x.endswith("m"):
        n = int(x[:-1] or "1")
        return int(round((60 / n) * 24 * 365))
    # fallback to daily
    return 365


def load_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime_utc" not in df.columns:
        raise ValueError(f"Raw CSV must contain datetime_utc. Got columns: {list(df.columns)[:20]}")
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in RAW_NUMERIC_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in RAW_INT_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


def _validate_and_drop_core_nans(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    checks = {
        "datetime_nulls": int(out["datetime_utc"].isna().sum()),
        "datetime_duplicates": int(out["datetime_utc"].duplicated().sum()),
    }

    before = len(out)
    out = out.dropna(subset=["datetime_utc", "open", "high", "low", "close"]).copy()
    checks["dropped_rows_ohlc_na"] = int(before - len(out))

    checks["bad_high_count"] = int((out["high"] < out[["open", "close"]].max(axis=1)).sum())
    checks["bad_low_count"] = int((out["low"] > out[["open", "close"]].min(axis=1)).sum())
    checks["nonpositive_close_count"] = int((out["close"] <= 0).sum())
    return out, checks


def add_features(df: pd.DataFrame, *, periods_per_year: int) -> pd.DataFrame:
    """
    Reproduces data-btc.ipynb / data-eth.ipynb feature pipeline in generic form.
    """
    out = df.copy()
    out = out.set_index("datetime_utc").sort_index()

    out["ret_1bar"] = out["close"].pct_change()
    out["logret_1bar"] = np.log(out["close"]).diff()

    out["range"] = (out["high"] - out["low"]) / out["close"].shift(1)

    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            (out["high"] - out["low"]),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["true_range"] = tr
    out["atr_14"] = out["true_range"].rolling(14, min_periods=14).mean()
    out["vol_30"] = out["ret_1bar"].rolling(30, min_periods=30).std() * np.sqrt(periods_per_year)

    # compatibility aliases for old 1d notebooks
    out["ret_1d"] = out["ret_1bar"]
    out["logret_1d"] = out["logret_1bar"]

    return out


def build_clean_dataset(
    df_raw: pd.DataFrame,
    *,
    interval: str,
) -> tuple[pd.DataFrame, CleaningStats]:
    rows_before = len(df_raw)
    typed = _coerce_types(df_raw)
    typed, checks = _validate_and_drop_core_nans(typed)
    rows_after_dropna_ohlc = len(typed)

    ppy = infer_periods_per_year(interval)
    feat = add_features(typed, periods_per_year=ppy)
    feat = feat[KEEP_COLS + ["ret_1d", "logret_1d"]].copy()

    before_feature_drop = len(feat)
    feat_clean = feat.dropna(subset=REQUIRED_AFTER_FEATURES).copy()
    rows_after_feature_dropna = len(feat_clean)

    stats = CleaningStats(
        rows_before=rows_before,
        rows_after_dropna_ohlc=rows_after_dropna_ohlc,
        rows_after_feature_dropna=rows_after_feature_dropna,
        dropped_rows_ohlc_na=int(checks["dropped_rows_ohlc_na"]),
        dropped_rows_feature_na=int(before_feature_drop - rows_after_feature_dropna),
        datetime_nulls=int(checks["datetime_nulls"]),
        datetime_duplicates=int(checks["datetime_duplicates"]),
        bad_high_count=int(checks["bad_high_count"]),
        bad_low_count=int(checks["bad_low_count"]),
        nonpositive_close_count=int(checks["nonpositive_close_count"]),
    )
    return feat_clean, stats


def make_processed_output_path(
    *,
    active: str,
    interval: str,
    output_dir: Path = Path(__file__).resolve().parent,
) -> Path:
    return output_dir / f"{active.lower()}-{interval}.csv"


def process_raw_csv_to_features_csv(
    *,
    raw_csv: Path,
    active: str,
    interval: str,
    output_dir: Path = Path(__file__).resolve().parent,
) -> tuple[Path, CleaningStats]:
    df_raw = load_raw_csv(raw_csv)
    clean, stats = build_clean_dataset(df_raw, interval=interval)
    out_path = make_processed_output_path(active=active, interval=interval, output_dir=output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=True)
    return out_path, stats


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean Binance raw klines CSV and add metrics.")
    p.add_argument("--raw-csv", required=True, help="Path to ACTIVE-TIMEFRAME-RAW.csv")
    p.add_argument("--active", required=True, help="ACTIVE label for output filename, e.g. btc")
    p.add_argument("--interval", required=True, help="Timeframe label for output filename, e.g. 1d, 4h")
    p.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory for ACTIVE-TIMEFRAME.csv output",
    )
    return p


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    out_path, stats = process_raw_csv_to_features_csv(
        raw_csv=Path(args.raw_csv),
        active=args.active,
        interval=args.interval,
        output_dir=Path(args.output_dir),
    )
    print(f"Saved cleaned CSV: {out_path}")
    print(stats)


if __name__ == "__main__":
    main()
