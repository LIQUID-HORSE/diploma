"""Walk-forward splitter utilities.

Implements the walk-forward design described in the course specification:
- Rolling train window (fixed length)
- Fixed test window
- Non-overlapping test windows for easy stitching
- Warm-up buffer for indicators; metrics are computed only on the pure test window.

Folds are represented as half-open intervals [start, end) on a DatetimeIndex.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class WFFold:
    """One walk-forward fold definition.

    All intervals are half-open: [start, end). `*_end` is exclusive.

    `buffer_start` is the start of the warm-up period that directly precedes test_start
    and is included in the test backtest run, but excluded from OOS metrics.
    """

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    buffer_start: pd.Timestamp

    def __post_init__(self) -> None:
        if not (self.train_start < self.train_end <= self.test_start < self.test_end):
            raise ValueError(
                f"Invalid fold ordering: {self.train_start=}, {self.train_end=}, {self.test_start=}, {self.test_end=}"
            )
        if self.buffer_start > self.test_start:
            raise ValueError("buffer_start must be <= test_start")

    @property
    def test_with_buffer_start(self) -> pd.Timestamp:
        return self.buffer_start

    def slice_train(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[(df.index >= self.train_start) & (df.index < self.train_end)]

    def slice_test_with_buffer(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[(df.index >= self.buffer_start) & (df.index < self.test_end)]

    def slice_pure_test(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[(df.index >= self.test_start) & (df.index < self.test_end)]


def _align_next(index: pd.DatetimeIndex, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Return first index value >= ts (or None if beyond end)."""
    pos = index.searchsorted(ts, side="left")
    if pos >= len(index):
        return None
    return pd.Timestamp(index[pos])


def _buffer_start_by_bars(index: pd.DatetimeIndex, test_start: pd.Timestamp, warmup_bars: int) -> pd.Timestamp:
    pos = index.searchsorted(test_start, side="left")
    if pos <= 0:
        return pd.Timestamp(index[0])
    start_pos = max(0, pos - warmup_bars)
    return pd.Timestamp(index[start_pos])


def generate_folds(
    index: pd.DatetimeIndex,
    *,
    train_years: int = 3,
    test_months: int = 6,
    step_months: int = 6,
    warmup_bars: int = 252,
    require_full_warmup: bool = True,
) -> List[WFFold]:
    """Generate rolling walk-forward folds for a given datetime index.

    Notes
    -----
    - Uses calendar offsets for train/test/step.
    - Then aligns boundaries to the next available bar in the index.
    - If `require_full_warmup` is True, folds that don't have at least `warmup_bars`
      available before the aligned test_start are skipped.

    Returns
    -------
    list[WFFold]
    """
    if len(index) < 10:
        return []

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a pandas.DatetimeIndex")

    index = index.sort_values()
    start = pd.Timestamp(index[0])
    end = pd.Timestamp(index[-1])

    folds: List[WFFold] = []
    k = 0

    test_start_nominal = start + pd.DateOffset(years=train_years)
    while True:
        test_start = _align_next(index, pd.Timestamp(test_start_nominal))
        if test_start is None:
            break

        train_start = _align_next(index, test_start - pd.DateOffset(years=train_years))
        if train_start is None:
            break

        train_end = test_start  # exclusive

        test_end = _align_next(index, test_start + pd.DateOffset(months=test_months))
        if test_end is None:
            # If test_end beyond data, stop (no incomplete fold)
            break

        buffer_start = _buffer_start_by_bars(index, test_start, warmup_bars)
        if require_full_warmup:
            # Ensure there are warmup_bars bars before test_start
            pos = index.searchsorted(test_start, side="left")
            if pos < warmup_bars:
                # Not enough history for buffer -> skip this fold and move on
                test_start_nominal = test_start_nominal + pd.DateOffset(months=step_months)
                continue

        # Guard against pathological alignment where windows collapse
        if train_start >= train_end or test_start >= test_end:
            test_start_nominal = test_start_nominal + pd.DateOffset(months=step_months)
            continue

        folds.append(
            WFFold(
                fold_id=k,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                buffer_start=buffer_start,
            )
        )
        k += 1
        test_start_nominal = test_start_nominal + pd.DateOffset(months=step_months)

        if test_start_nominal >= end:
            break

    return folds
