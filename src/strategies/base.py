"""Strategy base helpers and indicators for backtesting.py.

All strategies in this project are long/flat (no short) at the MVP stage.

Key implementation details from the spec:
- trade_on_close=False: signals computed on close t, execution at open t+1
- warm-up buffer is included in test runs, but trading should start only at test_start
  (to prevent trading inside the warm-up segment).

We implement a simple `start_date` gate: strategies do nothing before start_date.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from backtesting import Strategy
except Exception as e:  # pragma: no cover
    Strategy = object  # type: ignore[misc,assignment]
    _IMPORT_ERROR = e
else:  # pragma: no cover
    _IMPORT_ERROR = None


def _as_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(np.asarray(x, dtype=float))


def sma(close, n: int) -> np.ndarray:
    s = _as_series(close)
    return s.rolling(int(n)).mean().to_numpy()


def ema(close, n: int) -> np.ndarray:
    s = _as_series(close)
    return s.ewm(span=int(n), adjust=False).mean().to_numpy()


def rolling_std(close, n: int) -> np.ndarray:
    s = _as_series(close)
    return s.rolling(int(n)).std(ddof=0).to_numpy()


def rsi_wilder(close, n: int) -> np.ndarray:
    """RSI with Wilder smoothing (alpha=1/n)."""
    s = _as_series(close)
    d = s.diff()
    up = d.clip(lower=0.0)
    down = (-d).clip(lower=0.0)
    alpha = 1.0 / float(int(n))
    roll_up = up.ewm(alpha=alpha, adjust=False).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).to_numpy()


def pct_change(close) -> np.ndarray:
    s = _as_series(close)
    return s.pct_change().fillna(0.0).to_numpy()


def ann_vol_from_close(close, lookback: int, periods_per_year: int = 365) -> np.ndarray:
    """Rolling annualized volatility estimate from close-to-close returns."""
    s = _as_series(close)
    r = s.pct_change()
    vol = r.rolling(int(lookback)).std(ddof=0) * np.sqrt(periods_per_year)
    # IMPORTANT: do NOT use backfill (bfill) here.
    # Backfilling would populate the initial NaNs with *future* realized volatility,
    # which is a classic look-ahead leak. We prefer forward-fill (ffill) and then
    # set remaining leading NaNs to 0.0. In the vol-target overlay, vol<=0 means
    # "no scaling" (size=1.0), which is a conservative default for the very first bars.
    vol = vol.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return vol.to_numpy()


def boll_mid(close, n: int) -> np.ndarray:
    return sma(close, n)


def boll_lower(close, n: int, k: float) -> np.ndarray:
    m = sma(close, n)
    sd = rolling_std(close, n)
    return (m - float(k) * sd)


def zscore_to_sma(close, n: int) -> np.ndarray:
    s = _as_series(close)
    m = s.rolling(int(n)).mean()
    sd = s.rolling(int(n)).std(ddof=0).replace(0.0, np.nan)
    z = (s - m) / sd
    return z.fillna(0.0).to_numpy()


def donchian_high_prev(close, n: int) -> np.ndarray:
    """Previous N-bar highest close (shifted by 1 to avoid look-ahead)."""
    s = _as_series(close)
    return s.shift(1).rolling(int(n)).max().to_numpy()


def donchian_low_prev(close, n: int) -> np.ndarray:
    """Previous N-bar lowest close (shifted by 1 to avoid look-ahead)."""
    s = _as_series(close)
    return s.shift(1).rolling(int(n)).min().to_numpy()


def tsmom(close, L: int) -> np.ndarray:
    """Time-series momentum: close/close[t-L] - 1."""
    s = _as_series(close)
    return (s / s.shift(int(L)) - 1.0).fillna(0.0).to_numpy()


class BaseCryptoStrategy(Strategy):
    """Base for project strategies.

    Parameters (class attributes in backtesting.py):
    - start_date: trading is disabled strictly before this timestamp.
    - use_vol_target: if True, order size is scaled by target_vol / realized_vol,
      clipped to [0, 1] (no leverage).
    """

    start_date: Optional[pd.Timestamp] = None
    use_vol_target: bool = False
    vol_lookback: int = 60
    target_vol: float = 0.40
    periods_per_year: int = 365

    def init(self):
        if _IMPORT_ERROR is not None:
            raise _IMPORT_ERROR  # pragma: no cover

        if self.use_vol_target:
            self._ann_vol = self.I(ann_vol_from_close, self.data.Close, self.vol_lookback, self.periods_per_year)

    def I(self, func, *args, **kwargs):
        def wrapped(*a, **k):
            out = func(*a, **k)

            # backtesting.py поддерживает multi-output индикаторы
            if isinstance(out, (tuple, list)):
                return tuple(np.asarray(x).copy() for x in out)

            return np.asarray(out).copy()

        return super().I(wrapped, *args, **kwargs)

    def _can_trade_now(self) -> bool:
        if self.start_date is None:
            return True
        # backtesting.py exposes index via self.data.index
        try:
            ts = pd.Timestamp(self.data.index[-1])
        except Exception:
            return True
        return ts >= pd.Timestamp(self.start_date)

    def _target_size(self) -> Optional[float]:
        """Return position size fraction of equity (0..1) or None for default."""
        if not self.use_vol_target:
            return None
        vol = float(self._ann_vol[-1]) if hasattr(self, "_ann_vol") else 0.0
        if (not np.isfinite(vol)) or vol <= 0:
            return 1.0
        w = float(self.target_vol) / vol
        if not np.isfinite(w):
            return 1.0
        return float(max(0.0, min(1.0, w)))

    def _buy_full_or_scaled(self):
        sz = self._target_size()
        if sz is None:
            self.buy()
        else:
            self.buy(size=sz)

    def _flat_before_start(self):
        """Ensure we are flat before start_date (important when running with warm-up)."""
        if self.start_date is None:
            return
        if not self._can_trade_now():
            if self.position:
                self.position.close()
