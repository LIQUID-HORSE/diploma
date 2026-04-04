"""Synergy strategies: combinations of signals (long/flat).

The spec intentionally uses small parameter grids for synergies to limit
degrees of freedom and data-snooping risk.

S1: MA filter + RSI MR
S2: MA(200) filter + Bollinger MR
S3: Breakout + MA confirmation
S4: MA crossover + TSMOM confirmation
S5: Simple ensemble (MA regime + Breakout regime + TSMOM)
"""

from __future__ import annotations

from .base import (
    BaseCryptoStrategy,
    boll_lower,
    boll_mid,
    donchian_high_prev,
    donchian_low_prev,
    rsi_wilder,
    sma,
    tsmom,
)


class MAFilterRSI(BaseCryptoStrategy):
    """S1: Trend filter (Close > SMA(M)) + RSI mean reversion entry."""

    M: int = 200
    n: int = 14
    low: float = 30.0
    exit: float = 55.0

    def init(self):
        super().init()
        self._ma = self.I(sma, self.data.Close, self.M)
        self._rsi = self.I(rsi_wilder, self.data.Close, self.n)

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return

        c = float(self.data.Close[-1])
        ma = float(self._ma[-1])
        r = float(self._rsi[-1])

        if not self.position:
            if c > ma and r < float(self.low):
                self._buy_full_or_scaled()
        else:
            if r > float(self.exit):
                self.position.close()


class MA200FilterBollinger(BaseCryptoStrategy):
    """S2: Close > SMA(M) filter + Bollinger MR trigger."""

    M: int = 200
    n: int = 20
    k: float = 2.0

    def init(self):
        super().init()
        self._ma_filter = self.I(sma, self.data.Close, self.M)
        self._mid = self.I(boll_mid, self.data.Close, self.n)
        self._low = self.I(boll_lower, self.data.Close, self.n, self.k)

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return

        c = float(self.data.Close[-1])
        if not self.position:
            if c > float(self._ma_filter[-1]) and c < float(self._low[-1]):
                self._buy_full_or_scaled()
        else:
            if c > float(self._mid[-1]):
                self.position.close()


class BreakoutConfirmMA(BaseCryptoStrategy):
    """S3: Donchian breakout + MA confirmation.

    Entry: breakout(N) AND Close > SMA(M)
    Exit:  Donchian low with exit window = max(10, N//2)

    The spec's synergy table doesn't specify an exit parameter; using N//2 is a
    reasonable, transparent default and keeps the parameter grid small.
    """

    N: int = 100
    M: int = 200

    def init(self):
        super().init()
        self._hi = self.I(donchian_high_prev, self.data.Close, self.N)
        self._ma = self.I(sma, self.data.Close, self.M)
        self._exit_n = max(10, int(self.N) // 2)
        self._lo = self.I(donchian_low_prev, self.data.Close, self._exit_n)

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return

        c = float(self.data.Close[-1])
        if not self.position:
            if c > float(self._hi[-1]) and c > float(self._ma[-1]):
                self._buy_full_or_scaled()
        else:
            if c < float(self._lo[-1]):
                self.position.close()


class MACrossTSMOMConfirm(BaseCryptoStrategy):
    """S4: MA crossover confirmed by positive TSMOM."""

    fast: int = 20
    slow: int = 200
    L: int = 120

    def init(self):
        super().init()
        self._fast = self.I(sma, self.data.Close, self.fast)
        self._slow = self.I(sma, self.data.Close, self.slow)
        self._mom = self.I(tsmom, self.data.Close, self.L)

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return

        trend = float(self._fast[-1]) > float(self._slow[-1])
        mom_ok = float(self._mom[-1]) > 0.0

        if trend and mom_ok:
            if not self.position:
                self._buy_full_or_scaled()
        else:
            if self.position:
                self.position.close()


class SimpleEnsemble(BaseCryptoStrategy):
    """S5: Simple ensemble of 3 regime signals.

    The spec uses: pos = sign(mean([MA, Breakout, TSMOM])) -> long/flat.

    For sign(mean()) to be meaningful, component signals should be symmetric
    (+1 for long regime, -1 for flat regime). We use majority vote on {-1, +1}.

    Parameters:
    - ma_pair: tuple(fast, slow)
    - N: breakout lookback for Donchian regime (entry/exit state machine)
    - L: TSMOM lookback
    """

    ma_pair: tuple[int, int] = (20, 200)
    N: int = 200
    L: int = 252

    def init(self):
        super().init()
        fast, slow = self.ma_pair
        self._ma_fast = self.I(sma, self.data.Close, int(fast))
        self._ma_slow = self.I(sma, self.data.Close, int(slow))

        self._hi = self.I(donchian_high_prev, self.data.Close, self.N)
        self._exit_n = max(10, int(self.N) // 2)
        self._lo = self.I(donchian_low_prev, self.data.Close, self._exit_n)

        self._mom = self.I(tsmom, self.data.Close, self.L)

        self._breakout_state = False  # persistent regime

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return

        c = float(self.data.Close[-1])

        # Update breakout regime state
        if not self._breakout_state and c > float(self._hi[-1]):
            self._breakout_state = True
        elif self._breakout_state and c < float(self._lo[-1]):
            self._breakout_state = False

        sig_ma = 1.0 if float(self._ma_fast[-1]) > float(self._ma_slow[-1]) else -1.0
        sig_brk = 1.0 if self._breakout_state else -1.0
        sig_mom = 1.0 if float(self._mom[-1]) > 0.0 else -1.0

        vote = (sig_ma + sig_brk + sig_mom) / 3.0
        want_long = vote > 0.0

        if want_long:
            if not self.position:
                self._buy_full_or_scaled()
        else:
            if self.position:
                self.position.close()
