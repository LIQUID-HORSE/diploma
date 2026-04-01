from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

from ..metrics import (
    CostScenario,
    compute_metrics_from_returns,
    daily_returns_from_backtesting_stats,
    objective_calmar_with_sanity_penalties,
)
from .core import BacktestConfig, make_backtest
from .strategy_wrap import strategy_with_start_date
from src.experiment_config import ObjectiveConfig, BacktestConfig

# Keep StrategySpec as "Any" here to avoid import cycles; wf_runner will type it.
def extract_best_params(stats: Any, param_names: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    strat_obj = None

    if hasattr(stats, "_strategy"):
        strat_obj = getattr(stats, "_strategy")
    elif hasattr(stats, "get"):
        strat_obj = stats.get("_strategy", None)

    for p in param_names:
        if strat_obj is not None and hasattr(strat_obj, p):
            out[p] = getattr(strat_obj, p)

    return out


def optimize_on_train(
    train_df: pd.DataFrame,
    *,
    strategy_cls: Any,
    param_grid: Dict[str, Sequence],
    constraint: Optional[Callable[[Any], bool]],
    cost: CostScenario,
    bt_cfg: BacktestConfig,
    obj_cfg: ObjectiveConfig,
    return_heatmap: bool,
) -> Tuple[Any, Optional[pd.Series], Dict[str, Any], float]:
    """
    Optimize on train and return:
      best_stats, heatmap(optional), best_params, best_train_objective(float)
    """
    bt = make_backtest(train_df, strategy_cls, bt_cfg=bt_cfg, commission=cost.commission, spread=cost.spread)

    maximize = lambda s: objective_calmar_with_sanity_penalties(  # noqa: E731
        s,
        min_trades=obj_cfg.min_trades,
        min_exposure=obj_cfg.min_exposure,
        penalty=obj_cfg.penalty,
    )

    if return_heatmap:
        res = bt.optimize(
            **param_grid,
            maximize=maximize,
            constraint=constraint,
            return_heatmap=True,
        )
        if isinstance(res, tuple) and len(res) == 2:
            best_stats, heatmap = res
        else:
            best_stats, heatmap = res, None
    else:
        best_stats = bt.optimize(**param_grid, maximize=maximize, constraint=constraint)
        heatmap = None

    best_params = extract_best_params(best_stats, param_grid.keys())
    try:
        best_obj = float(maximize(best_stats))
    except Exception:
        best_obj = float("nan")

    return best_stats, heatmap, best_params, best_obj


def run_on_test(
    test_df_with_buffer: pd.DataFrame,
    *,
    strategy_cls: Any,
    start_date: pd.Timestamp,
    params: Dict[str, Any],
    cost: CostScenario,
    bt_cfg: BacktestConfig,
) -> Any:
    StrategyCls = strategy_with_start_date(strategy_cls, start_date)
    bt = make_backtest(test_df_with_buffer, StrategyCls, bt_cfg=bt_cfg, commission=cost.commission, spread=cost.spread)
    return bt.run(**params)


def extract_test_returns(
    stats: Any,
    *,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.Series:
    r = daily_returns_from_backtesting_stats(stats).copy()
    if not isinstance(r.index, pd.DatetimeIndex):
        raise TypeError("Expected DatetimeIndex in returns extracted from backtesting.py stats.")
    r = r.sort_index()
    r = r.loc[(r.index >= test_start) & (r.index < test_end)]
    r.name = "ret"
    return r


def metrics_row_from_oos_returns(
    oos_ret: pd.Series,
    *,
    stats: Any,
) -> Dict[str, Any]:
    """
    Compute OOS metrics from returns + extract a couple of backtesting.py fields.
    """
    m = compute_metrics_from_returns(oos_ret)

    try:
        n_trades = int(stats.get("# Trades", 0))
    except Exception:
        n_trades = 0
    try:
        exposure_pct = float(stats.get("Exposure Time [%]", 0.0))
    except Exception:
        exposure_pct = 0.0

    return {
        "oos_total_return": m["Total Return"],
        "oos_cagr": m["CAGR"],
        "oos_sharpe": m["Sharpe"],
        "oos_sortino": m["Sortino"],
        "oos_maxdd": m["MaxDD"],
        "oos_calmar": m["Calmar"],
        "oos_ann_vol": m["Ann. Vol"],
        "oos_n_bars": m["n_bars"],
        "oos_trades": n_trades,
        "oos_exposure_pct": exposure_pct,
    }