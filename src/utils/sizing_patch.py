from __future__ import annotations

from typing import Optional


def apply_order_size_buffer(max_rel_size: float = 0.999) -> None:
    """
    Apply a small safety buffer for relative-sized orders so 'all-in' orders don't get
    rejected due to commission/spread/rounding.

    This monkey-patches BaseCryptoStrategy._buy_full_or_scaled.

    Call this ONCE early in the notebook/script, after config is available.
    """
    from src.strategies.base import BaseCryptoStrategy  # local import to avoid import cycles

    orig = BaseCryptoStrategy._buy_full_or_scaled  # type: ignore[attr-defined]

    def _buy_full_or_scaled_capped(self):  # type: ignore[no-untyped-def]
        sz = self._target_size()
        if sz is None:
            return self.buy()
        try:
            sz_f = float(sz)
        except Exception:
            return self.buy(size=sz)
        return self.buy(size=min(sz_f, float(max_rel_size)))

    # Avoid double patch
    if getattr(BaseCryptoStrategy._buy_full_or_scaled, "__name__", "") == "_buy_full_or_scaled_capped":
        return

    BaseCryptoStrategy._buy_full_or_scaled = _buy_full_or_scaled_capped  # type: ignore[assignment]
    BaseCryptoStrategy._buy_full_or_scaled.__wrapped__ = orig  # type: ignore[attr-defined]