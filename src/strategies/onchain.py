"""On-chain strategy extensions (daily).

These classes extend existing price-based strategies with on-chain filters:
- Activity trend via AdrActCnt short/long SMA
- MVRV cap via CapMVRVCur

All strategies remain long/flat and keep the original exit logic unchanged.
"""

from __future__ import annotations

import numpy as np

from .base import sma
from .meanrev import RSIMeanReversion, BollingerMeanReversion, ZScoreMeanReversion
from .synergy import MAFilterRSI, MA200FilterBollinger


def _require_data_columns(data: object, cols: list[str]) -> None:
    missing = [c for c in cols if not hasattr(data, c)]
    if missing:
        raise ValueError(
            "Missing required on-chain columns: "
            f"{missing}. Ensure merged data includes these fields."
        )


def _finite(v: float) -> bool:
    return bool(np.isfinite(v))


class RSIMeanReversionOnchain(RSIMeanReversion):
    """R1OC: RSI MR + Activity + MVRV."""

    addr_short: int = 14
    addr_long: int = 60
    mvrv_cap: float = 2.5

    def init(self):
        _require_data_columns(self.data, ["AdrActCnt", "CapMVRVCur"])
        super().init()
        self._addr_short = self.I(sma, self.data.AdrActCnt, self.addr_short)
        self._addr_long = self.I(sma, self.data.AdrActCnt, self.addr_long)

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return

        r = float(self._rsi[-1])
        act_s = float(self._addr_short[-1])
        act_l = float(self._addr_long[-1])
        mvrv = float(self.data.CapMVRVCur[-1])

        if not self.position:
            activity_ok = _finite(act_s) and _finite(act_l) and (act_s > act_l)
            mvrv_ok = _finite(mvrv) and (mvrv < float(self.mvrv_cap))
            if r < float(self.low) and activity_ok and mvrv_ok:
                self._buy_full_or_scaled()
        else:
            if r > float(self.exit):
                self.position.close()


class BollingerMeanReversionOnchain(BollingerMeanReversion):
    """R2OC: Bollinger MR + Activity + MVRV."""

    addr_short: int = 14
    addr_long: int = 60
    mvrv_cap: float = 2.5

    def init(self):
        _require_data_columns(self.data, ["AdrActCnt", "CapMVRVCur"])
        super().init()
        self._addr_short = self.I(sma, self.data.AdrActCnt, self.addr_short)
        self._addr_long = self.I(sma, self.data.AdrActCnt, self.addr_long)

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return

        c = float(self.data.Close[-1])
        act_s = float(self._addr_short[-1])
        act_l = float(self._addr_long[-1])
        mvrv = float(self.data.CapMVRVCur[-1])

        if not self.position:
            activity_ok = _finite(act_s) and _finite(act_l) and (act_s > act_l)
            mvrv_ok = _finite(mvrv) and (mvrv < float(self.mvrv_cap))
            if c < float(self._low[-1]) and activity_ok and mvrv_ok:
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
                if c > float(self._mid[-1]):
                    self.position.close()
                    self._entry_bar = None


class ZScoreMeanReversionOnchain(ZScoreMeanReversion):
    """R3OC: Z-Score MR + Activity + MVRV."""

    addr_short: int = 14
    addr_long: int = 60
    mvrv_cap: float = 2.5

    def init(self):
        _require_data_columns(self.data, ["AdrActCnt", "CapMVRVCur"])
        super().init()
        self._addr_short = self.I(sma, self.data.AdrActCnt, self.addr_short)
        self._addr_long = self.I(sma, self.data.AdrActCnt, self.addr_long)

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return

        z = float(self._z[-1])
        act_s = float(self._addr_short[-1])
        act_l = float(self._addr_long[-1])
        mvrv = float(self.data.CapMVRVCur[-1])

        if not self.position:
            activity_ok = _finite(act_s) and _finite(act_l) and (act_s > act_l)
            mvrv_ok = _finite(mvrv) and (mvrv < float(self.mvrv_cap))
            if z < -float(self.entry_z) and activity_ok and mvrv_ok:
                self._buy_full_or_scaled()
        else:
            if z > -float(self.exit_z):
                self.position.close()


class MAFilterRSIOnchain(MAFilterRSI):
    """S1OC: MAFilter + RSI + MVRV."""

    mvrv_cap: float = 2.5

    def init(self):
        _require_data_columns(self.data, ["CapMVRVCur"])
        super().init()

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return

        c = float(self.data.Close[-1])
        ma = float(self._ma[-1])
        r = float(self._rsi[-1])
        mvrv = float(self.data.CapMVRVCur[-1])

        if not self.position:
            mvrv_ok = _finite(mvrv) and (mvrv < float(self.mvrv_cap))
            if c > ma and r < float(self.low) and mvrv_ok:
                self._buy_full_or_scaled()
        else:
            if r > float(self.exit):
                self.position.close()


class MA200FilterBollingerOnchain(MA200FilterBollinger):
    """S2OC: MAFilter + Bollinger + MVRV."""

    mvrv_cap: float = 2.5

    def init(self):
        _require_data_columns(self.data, ["CapMVRVCur"])
        super().init()

    def next(self):
        self._flat_before_start()
        if not self._can_trade_now():
            return

        c = float(self.data.Close[-1])
        mvrv = float(self.data.CapMVRVCur[-1])

        if not self.position:
            mvrv_ok = _finite(mvrv) and (mvrv < float(self.mvrv_cap))
            if c > float(self._ma_filter[-1]) and c < float(self._low[-1]) and mvrv_ok:
                self._buy_full_or_scaled()
        else:
            if c > float(self._mid[-1]):
                self.position.close()

