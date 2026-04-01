from __future__ import annotations

import inspect
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Type

import pandas as pd
from backtesting import Backtest
from src.experiment_config import BacktestConfig


_REQUIRED_BACKTEST_KWARGS: Set[str] = {
    "cash",
    "commission",
    "spread",
    "trade_on_close",
    "exclusive_orders",
}


def get_backtest_cls() -> Type[Backtest]:
    """Prefer FractionalBacktest if available; fallback to Backtest."""
    try:
        from backtesting.lib import FractionalBacktest  # type: ignore
        return FractionalBacktest
    except Exception:
        return Backtest


def make_backtest(
    data: pd.DataFrame,
    StrategyCls: Any,
    *,
    bt_cfg: BacktestConfig,
    commission: float,
    spread: float,
    backtest_cls: Optional[Type[Backtest]] = None,
    fail_fast_on_required_kwargs: bool = True,
) -> Backtest:
    """
    Create Backtest with version-robust kwargs handling.

    Critical idea:
    - if backtesting.py rejects required kwargs (costs/execution), raise loudly
      because silently dropping them invalidates comparisons.
    """
    BACKTEST_CLS = backtest_cls or get_backtest_cls()

    kwargs: Dict[str, Any] = {
        "cash": bt_cfg.cash,
        "commission": commission,
        "spread": spread,
        "trade_on_close": bt_cfg.trade_on_close,
        "exclusive_orders": bt_cfg.exclusive_orders,
        "hedging": bt_cfg.hedging,
        "margin": bt_cfg.margin,
        "finalize_trades": bt_cfg.finalize_trades,
    }

    dropped: list[str] = []

    # Attempt 1: signature-based filtering.
    try:
        sig = inspect.signature(BACKTEST_CLS.__init__)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not has_var_kw:
            supported = set(params.keys())
            supported.discard("self")
            unsupported = [k for k in list(kwargs.keys()) if k not in supported]
            if unsupported:
                hard = [k for k in unsupported if k in _REQUIRED_BACKTEST_KWARGS]
                if hard and fail_fast_on_required_kwargs:
                    raise RuntimeError(
                        f"Installed backtesting.py does not support required BACKTEST_CLS() kwargs {hard}. "
                        "Upgrade backtesting.py or adjust experiment assumptions (not recommended)."
                    )
                for k in unsupported:
                    kwargs.pop(k, None)
                dropped.extend(unsupported)
    except Exception:
        # If signature introspection fails, we fall back to the TypeError loop below.
        pass

    # Attempt 2: instantiate; on TypeError, drop truly unknown kwargs (but never drop required).
    while True:
        try:
            bt = BACKTEST_CLS(data, StrategyCls, **kwargs)
            break
        except TypeError as e:
            msg = str(e)
            m = re.search(r"unexpected keyword argument ['\"](?P<kw>\w+)['\"]", msg)
            if not m:
                raise
            bad_kw = m.group("kw")
            if bad_kw in _REQUIRED_BACKTEST_KWARGS and fail_fast_on_required_kwargs:
                raise RuntimeError(
                    f"Installed backtesting.py rejected a required BACKTEST_CLS() kwarg '{bad_kw}'. "
                    "This would silently change costs/execution. Please upgrade backtesting.py."
                ) from e
            if bad_kw not in kwargs:
                raise
            kwargs.pop(bad_kw)
            dropped.append(bad_kw)

    if dropped:
        warnings.warn(
            "BACKTEST_CLS() did not accept some non-critical kwargs and they were dropped: "
            + ", ".join(sorted(set(dropped))),
            RuntimeWarning,
        )

    return bt