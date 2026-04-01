from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def strategy_with_start_date(cls: Any, start_date: Optional[pd.Timestamp]) -> Any:
    """
    Create a dynamic Strategy subclass with a fixed start_date gate.
    This relies on strategies inheriting BaseCryptoStrategy with start_date handling.
    """
    if start_date is None:
        return cls
    ts = pd.Timestamp(start_date)
    return type(
        f"{cls.__name__}__Start_{ts.date()}",
        (cls,),
        {"start_date": ts},
    )