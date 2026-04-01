"""Mean-reversion strategies (long/flat).

R1: RSI mean reversion (buy oversold, exit on RSI recovery)
R2: Bollinger mean reversion (buy below lower band, exit on midline or time)
R3: Z-score to SMA (buy when price is far below SMA)
"""

from __future__ import annotations

from .base import (
    BaseCryptoStrategy,
    boll_lower,
    boll_mid,
    rsi_wilder,
    zscore_to_sma,
)


class RSIMeanReversion(BaseCryptoStrategy):
    """R1: RSI mean reversion."""

    n: int = 14
    low: float = 30.0
    exit: float = 50.0

    def init(self):
        super().init()
        self._rsi = self.I(rsi_wilder, self.data.Close, self.n)

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return

        r = float(self._rsi[-1])
        if not self.position:
            if r < float(self.low):
                self._buy_full_or_scaled()
        else:
            if r > float(self.exit):
                self.position.close()


class BollingerMeanReversion(BaseCryptoStrategy):
    """R2: Bollinger band mean reversion.

    Entry: close < lower band.
    Exit options (exit_mode):
      - 'midline': close > midline
      - 'fixed':   hold for `n` bars since entry (simple time stop)

    Note: 'fixed' is included because one of the provided tables lists exit={midline,fixed}.
    If you want a simpler universe, run only exit_mode='midline'.
    """

    n: int = 20
    k: float = 2.0
    exit_mode: str = "midline"  # 'midline' or 'fixed'

    def init(self):
        super().init()
        self._mid = self.I(boll_mid, self.data.Close, self.n)
        self._low = self.I(boll_lower, self.data.Close, self.n, self.k)
        self._entry_bar: int | None = None

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return

        c = float(self.data.Close[-1])
        if not self.position:
            if c < float(self._low[-1]):
                self._buy_full_or_scaled()
                self._entry_bar = len(self.data)
        else:
            if self.exit_mode == "midline":
                if c > float(self._mid[-1]):
                    self.position.close()
                    self._entry_bar = None
            elif self.exit_mode == "fixed":
                if self._entry_bar is not None and (len(self.data) - self._entry_bar) >= int(self.n):
                    self.position.close()
                    self._entry_bar = None
            else:
                # unknown mode -> behave like midline for safety
                if c > float(self._mid[-1]):
                    self.position.close()
                    self._entry_bar = None


class ZScoreMeanReversion(BaseCryptoStrategy):
    """R3: Z-score to SMA mean reversion.

    z = (close - SMA(n)) / std(n)
    Entry: z < -entry_z
    Exit:  z > -exit_z
    """

    n: int = 100
    entry_z: float = 2.0
    exit_z: float = 0.5

    def init(self):
        super().init()
        self._z = self.I(zscore_to_sma, self.data.Close, self.n)

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return

        z = float(self._z[-1])
        if not self.position:
            if z < -float(self.entry_z):
                self._buy_full_or_scaled()
        else:
            if z > -float(self.exit_z):
                self.position.close()
