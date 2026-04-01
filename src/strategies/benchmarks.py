"""Benchmark strategies.

Buy-and-hold is implemented as a Strategy so it uses the same execution and cost
model as the tested strategies, as required by the spec.
"""

from __future__ import annotations

from .base import BaseCryptoStrategy


class BuyHold(BaseCryptoStrategy):
    """Benchmark: open a long position once and hold to the end."""

    def init(self):
        super().init()
        self._entered = False

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return
        if not self._entered:
            if not self.position:
                self._buy_full_or_scaled()
            self._entered = True


class BuyHoldVolTarget(BuyHold):
    """Benchmark: buy-and-hold with (no-leverage) volatility targeting."""

    use_vol_target: bool = True
