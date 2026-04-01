from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .artifacts import safe_write_parquet
from .experiment_config import ExperimentConfig
from .metrics import COST_SCENARIOS, CostScenario, compute_metrics_from_returns
from .strategy_registry import StrategySpec
from .wf import WFFold
from .wf_runner import run_one_fold
from .backtest_engine.run import extract_test_returns, run_on_test


def _stitch_parts(parts: List[pd.Series]) -> pd.Series:
    stitched = pd.concat(parts).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    return stitched


def run_symbol_cost(
    *,
    sym: str,
    data: pd.DataFrame,
    folds: List[WFFold],
    cost: CostScenario,
    registry: List[StrategySpec],
    bench: StrategySpec,
    cfg: ExperimentConfig,
    logger=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """
    Run one (symbol, cost) over all folds:
      - bench OOS (stitched)
      - strategies OOS (stitched)
      - per-fold metrics table for strategies
      - per-fold metrics table for bench
      - optional train heatmaps
    Returns:
      folds_best_df, returns_oos_long, bench_oos_long, heatmaps_df_or_none, bench_folds_df
    """
    folds_rows: List[Dict[str, Any]] = []
    bench_fold_rows: List[Dict[str, Any]] = []
    heatmap_rows: List[pd.DataFrame] = []

    per_strat_parts: Dict[str, List[pd.Series]] = {s.strategy_id: [] for s in registry}
    bench_parts: List[pd.Series] = []

    for fold in folds:
        # ---- Benchmark per fold
        try:
            test_df = fold.slice_test_with_buffer(data)
            bench_stats = run_on_test(
                test_df,
                strategy_cls=bench.cls,
                start_date=fold.test_start,
                params={},  # no params
                cost=cost,
                bt_cfg=cfg.bt,
            )
            bench_ret = extract_test_returns(bench_stats, test_start=fold.test_start, test_end=fold.test_end)
            bench_parts.append(bench_ret)

            bm = compute_metrics_from_returns(bench_ret)
            bench_fold_rows.append(
                {
                    "symbol": sym,
                    "cost": cost.name,
                    "fold_id": fold.fold_id,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "bench_total_return": bm["Total Return"],
                    "bench_cagr": bm["CAGR"],
                    "bench_sharpe": bm["Sharpe"],
                    "bench_sortino": bm["Sortino"],
                    "bench_maxdd": bm["MaxDD"],
                    "bench_calmar": bm["Calmar"],
                    "bench_ann_vol": bm["Ann. Vol"],
                    "bench_n_bars": bm["n_bars"],
                }
            )
        except Exception as e:
            if cfg.artifacts.fail_fast:
                raise
            if logger is not None:
                logger.exception("Bench failed on %s %s fold=%d: %s", sym, cost.name, fold.fold_id, e)
            continue

        # ---- Strategies per fold
        for spec in registry:
            try:
                row, oos_ret, hm = run_one_fold(
                    sym=sym,
                    data=data,
                    fold=fold,
                    spec=spec,
                    cost=cost,
                    bt_cfg=cfg.bt,
                    obj_cfg=cfg.objective,
                    save_heatmap=bool(cfg.artifacts.save_heatmaps and spec.save_heatmap),
                )
                folds_rows.append(row)
                per_strat_parts[spec.strategy_id].append(oos_ret)

                if hm is not None:
                    heatmap_rows.append(hm)

            except Exception as e:
                if cfg.artifacts.fail_fast:
                    raise
                if logger is not None:
                    logger.exception(
                        "Strategy failed: %s %s fold=%d %s: %s",
                        sym, cost.name, fold.fold_id, spec.strategy_id, e
                    )
                continue

    # ---- Stitch strategies
    ret_long_parts: List[pd.DataFrame] = []
    for sid, parts in per_strat_parts.items():
        if not parts:
            continue
        stitched = _stitch_parts(parts)
        df_long = stitched.rename("ret").reset_index().rename(columns={"index": "date"})
        df_long["symbol"] = sym
        df_long["cost"] = cost.name
        df_long["strategy_id"] = sid
        ret_long_parts.append(df_long)

    returns_oos_long = (
        pd.concat(ret_long_parts, ignore_index=True)
        if ret_long_parts
        else pd.DataFrame(columns=["date", "ret", "symbol", "cost", "strategy_id"])
    )

    # ---- Stitch benchmark
    bench_oos_long: pd.DataFrame
    if bench_parts:
        bench_stitched = _stitch_parts(bench_parts)
        bench_oos_long = bench_stitched.rename("ret").reset_index().rename(columns={"index": "date"})
        bench_oos_long["symbol"] = sym
        bench_oos_long["cost"] = cost.name
        bench_oos_long["strategy_id"] = bench.strategy_id
    else:
        bench_oos_long = pd.DataFrame(columns=["date", "ret", "symbol", "cost", "strategy_id"])

    folds_best_df = pd.DataFrame(folds_rows)
    bench_folds_df = pd.DataFrame(bench_fold_rows)
    heatmaps_df = pd.concat(heatmap_rows, ignore_index=True) if heatmap_rows else None

    return folds_best_df, returns_oos_long, bench_oos_long, heatmaps_df, bench_folds_df


def run_full_experiment(
    *,
    cfg: ExperimentConfig,
    data_by_symbol: Dict[str, pd.DataFrame],
    folds_by_symbol: Dict[str, List[WFFold]],
    registry: List[StrategySpec],
    bench: StrategySpec,
    report_mod=None,
    logger=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """
    Run all symbols and all cost scenarios.
    If cfg.artifacts.use_cache_if_exists is True and report_mod is provided,
    load existing results instead of recomputing.
    """
    if cfg.artifacts.use_cache_if_exists:
        if report_mod is None:
            raise ValueError("use_cache_if_exists=True requires report_mod with load_results().")
        folds_best_all, returns_oos_all, bench_oos_all, heatmaps_all = report_mod.load_results(cfg.out_dir)
        bench_folds_path = cfg.results_dir / "bench_folds.parquet"
        bench_folds_all = pd.read_parquet(bench_folds_path) if bench_folds_path.exists() else pd.DataFrame()
        return folds_best_all, returns_oos_all, bench_oos_all, heatmaps_all, bench_folds_all

    folds_best_list: List[pd.DataFrame] = []
    returns_oos_list: List[pd.DataFrame] = []
    bench_oos_list: List[pd.DataFrame] = []
    heatmaps_list: List[pd.DataFrame] = []
    bench_folds_list: List[pd.DataFrame] = []

    for sym, df in data_by_symbol.items():
        folds = folds_by_symbol[sym]
        for cost_name in cfg.costs:
            cost = COST_SCENARIOS[cost_name]
            if logger is not None:
                logger.info("RUN: %s | cost=%s | folds=%d | strategies=%d",
                            sym, cost.name, len(folds), len(registry))

            fb, roos, boos, hm, bfolds = run_symbol_cost(
                sym=sym,
                data=df,
                folds=folds,
                cost=cost,
                registry=registry,
                bench=bench,
                cfg=cfg,
                logger=logger,
            )
            folds_best_list.append(fb)
            returns_oos_list.append(roos)
            bench_oos_list.append(boos)
            bench_folds_list.append(bfolds)
            if hm is not None:
                heatmaps_list.append(hm)

    folds_best_all = pd.concat(folds_best_list, ignore_index=True) if folds_best_list else pd.DataFrame()
    returns_oos_all = pd.concat(returns_oos_list, ignore_index=True) if returns_oos_list else pd.DataFrame()
    bench_oos_all = pd.concat(bench_oos_list, ignore_index=True) if bench_oos_list else pd.DataFrame()
    bench_folds_all = pd.concat(bench_folds_list, ignore_index=True) if bench_folds_list else pd.DataFrame()
    heatmaps_all = pd.concat(heatmaps_list, ignore_index=True) if heatmaps_list else None

    # Normalize date column type for downstream pivoting/reporting
    for df_ in (returns_oos_all, bench_oos_all):
        if "date" in df_.columns:
            df_["date"] = pd.to_datetime(df_["date"])

    safe_write_parquet(folds_best_all, cfg.results_dir / "folds_best.parquet")
    safe_write_parquet(returns_oos_all, cfg.results_dir / "returns_oos.parquet")
    safe_write_parquet(bench_oos_all, cfg.results_dir / "bench_returns_oos.parquet")
    safe_write_parquet(bench_folds_all, cfg.results_dir / "bench_folds.parquet")
    if heatmaps_all is not None:
        safe_write_parquet(heatmaps_all, cfg.results_dir / "train_heatmaps.parquet")

    if logger is not None:
        logger.info(
            "DONE. folds_best=%s | returns_oos=%s | bench=%s | heatmaps=%s",
            folds_best_all.shape, returns_oos_all.shape, bench_oos_all.shape,
            None if heatmaps_all is None else heatmaps_all.shape
        )

    return folds_best_all, returns_oos_all, bench_oos_all, heatmaps_all, bench_folds_all