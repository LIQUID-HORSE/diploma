from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .strategy_registry import StrategySpec

import pandas as pd

from .wf import WFFold
from .metrics import CostScenario
from .experiment_config import BacktestConfig, ObjectiveConfig
from .backtest_engine.run import (
    extract_test_returns,
    metrics_row_from_oos_returns,
    optimize_on_train,
    run_on_test,
)


def run_one_fold(
    *,
    sym: str,
    data: pd.DataFrame,
    fold: WFFold,
    spec: StrategySpec,
    cost: CostScenario,
    bt_cfg: BacktestConfig,
    obj_cfg: ObjectiveConfig,
    save_heatmap: bool,
    periods_per_year: int = 365,
) -> Tuple[Dict[str, Any], pd.Series, Optional[pd.DataFrame]]:
    """
    Run exactly one (symbol, fold, strategy, cost) unit:
      - train optimize (optional)
      - test run with buffer + start_date gating
      - slice pure test returns
      - compute metrics row
      - (optional) heatmap dataframe
    """
    train_df = fold.slice_train(data)
    test_df = fold.slice_test_with_buffer(data)

    heatmap_df: Optional[pd.DataFrame] = None

    if spec.param_grid:
        _best_stats, heatmap, best_params, train_obj = optimize_on_train(
            train_df,
            strategy_cls=spec.cls,
            param_grid=spec.param_grid,
            constraint=spec.constraint,
            cost=cost,
            bt_cfg=bt_cfg,
            obj_cfg=obj_cfg,
            return_heatmap=bool(save_heatmap),
            periods_per_year=periods_per_year,
        )
        if heatmap is not None:
            hm = heatmap.rename("objective").reset_index()
            hm["symbol"] = sym
            hm["cost"] = cost.name
            hm["fold_id"] = fold.fold_id
            hm["strategy_id"] = spec.strategy_id
            heatmap_df = hm
    else:
        best_params = {}
        train_obj = float("nan")

    test_stats = run_on_test(
        test_df,
        strategy_cls=spec.cls,
        start_date=fold.test_start,
        params=best_params,
        cost=cost,
        bt_cfg=bt_cfg,
    )

    oos_ret = extract_test_returns(test_stats, test_start=fold.test_start, test_end=fold.test_end)

    row: Dict[str, Any] = {
        "symbol": sym,
        "cost": cost.name,
        "fold_id": fold.fold_id,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "buffer_start": fold.buffer_start,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "strategy_code": spec.code,
        "strategy_name": spec.name,
        "strategy_id": spec.strategy_id,
        "params": json.dumps(best_params, default=str),
        "train_objective": float(train_obj),
    }
    row.update(metrics_row_from_oos_returns(oos_ret, stats=test_stats, periods_per_year=periods_per_year))

    return row, oos_ret, heatmap_df
