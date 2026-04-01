"""Momentum strategies (long/flat).

M1: Time-series momentum (TSMOM) - long if past L-day return is positive.
"""

from __future__ import annotations

from .base import BaseCryptoStrategy, tsmom


class TSMomentum(BaseCryptoStrategy):
    """M1: time-series momentum.

    Signal: mom(L) = close/close[t-L] - 1; long if mom(L) > 0 else flat.
    """

    L: int = 120

    def init(self):
        super().init()
        self._mom = self.I(tsmom, self.data.Close, self.L)

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return

        if self._mom[-1] > 0:
            if not self.position:
                self._buy_full_or_scaled()
        else:
            if self.position:
                self.position.close()
