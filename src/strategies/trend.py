"""Trend-following strategies (long/flat).

T1: SMA crossover
T2: EMA crossover
T3: Donchian breakout with exit channel (standard channel breakout template)
"""

from __future__ import annotations

from .base import BaseCryptoStrategy, donchian_high_prev, donchian_low_prev, ema, sma


class SMACrossover(BaseCryptoStrategy):
    """T1: fast SMA vs slow SMA."""

    fast: int = 20
    slow: int = 200

    def init(self):
        super().init()
        self._fast = self.I(sma, self.data.Close, self.fast)
        self._slow = self.I(sma, self.data.Close, self.slow)

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return

        if self._fast[-1] > self._slow[-1]:
            if not self.position:
                self._buy_full_or_scaled()
        else:
            if self.position:
                self.position.close()


class EMACrossover(BaseCryptoStrategy):
    """T2: fast EMA vs slow EMA."""

    fast: int = 20
    slow: int = 200

    def init(self):
        super().init()
        self._fast = self.I(ema, self.data.Close, self.fast)
        self._slow = self.I(ema, self.data.Close, self.slow)

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return

        if self._fast[-1] > self._slow[-1]:
            if not self.position:
                self._buy_full_or_scaled()
        else:
            if self.position:
                self.position.close()


class DonchianBreakout(BaseCryptoStrategy):
    """T3: Donchian breakout with separate exit channel.

    Entry: close > max(close_{t-1..t-N})

    Exit:  close < min(close_{t-1..t-exitN})
    """

    N: int = 100
    exit: int = 20

    def init(self):
        super().init()
        self._hi = self.I(donchian_high_prev, self.data.Close, self.N)
        self._lo = self.I(donchian_low_prev, self.data.Close, self.exit)

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return

        c = self.data.Close[-1]
        if not self.position:
            if c > self._hi[-1]:
                self._buy_full_or_scaled()
        else:
            if c < self._lo[-1]:
                self.position.close()
