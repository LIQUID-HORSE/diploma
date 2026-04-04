from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Optional
import warnings

import pandas as pd

from .artifacts import safe_write_parquet
from .data_io import load_all_symbols
from .experiment_config import (
    ExperimentConfig,
    default_config,
    periods_per_year_for_timeframe,
    save_config_json,
    warmup_bars_for_timeframe,
)
from .metrics import compute_metrics_from_returns
from .strategy_registry import build_registry, get_benchmark, maybe_filter_debug
from .utils.sizing_patch import apply_order_size_buffer
from .wf import generate_folds
from .wf_experiment import run_full_experiment


@dataclass(frozen=True)
class TimeframeContext:
    tf: str
    cfg: ExperimentConfig
    ppy: int
    symbols: list[str]
    data_by_symbol: dict[str, pd.DataFrame]
    folds_by_symbol: dict[str, list[Any]]
    registry: list[Any]
    bench: Any


@dataclass(frozen=True)
class WFOutputs:
    folds_best: pd.DataFrame
    returns_oos: pd.DataFrame
    bench_oos: pd.DataFrame
    bench_folds: pd.DataFrame
    heatmaps: Optional[pd.DataFrame]


@dataclass(frozen=True)
class ReportsOutputs:
    leaderboard: pd.DataFrame
    stitched: pd.DataFrame
    bench_stitched: pd.DataFrame
    beat: pd.DataFrame
    excess: pd.DataFrame


@dataclass(frozen=True)
class TestsOutputs:
    rcspa: pd.DataFrame
    rcspa_util: pd.DataFrame
    spa_tbl: pd.DataFrame
    spa_tbl_view: pd.DataFrame


@dataclass(frozen=True)
class RobustnessOutputs:
    sensitivity_calmar: pd.DataFrame
    sensitivity_utility: pd.DataFrame
    subperiod_calmar: pd.DataFrame
    subperiod_utility: pd.DataFrame


def _ensure_date_col(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    if "date" not in df.columns:
        if "datetime_utc" in df.columns:
            df = df.rename(columns={"datetime_utc": "date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "date"})
        else:
            raise KeyError(f"No date column found. Columns: {list(df.columns)[:30]}")
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_timeframe_context(
    *,
    project_root: Path,
    timeframe: str,
    symbol_to_active: Mapping[str, str],
    objective_min_trades: Optional[Mapping[str, int]] = None,
    rc_block_size: Optional[Mapping[str, int]] = None,
    results_root_name: str = "runner_exp",
    logger: Any = None,
) -> TimeframeContext:
    tf = timeframe.strip().lower()
    ppy = periods_per_year_for_timeframe(tf)

    cfg0 = default_config(project_root)
    data_paths = {
        symbol: (project_root / "data" / f"{active}-{tf}.csv")
        for symbol, active in symbol_to_active.items()
    }
    missing = [str(p) for p in data_paths.values() if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing data files for timeframe {tf}: {missing}")

    obj_min_trades = int((objective_min_trades or {}).get(tf, cfg0.objective.min_trades))
    block_size = int((rc_block_size or {}).get(tf, cfg0.rc_spa.block_size))

    results_dir_tf = project_root / "results" / results_root_name / tf
    figures_dir_tf = project_root / "figures" / results_root_name / tf

    cfg = replace(
        cfg0,
        data_paths=data_paths,
        out_dir=project_root,
        results_dir=results_dir_tf,
        figures_dir=figures_dir_tf,
        wf=replace(cfg0.wf, warmup_bars=warmup_bars_for_timeframe(tf)),
        objective=replace(cfg0.objective, min_trades=obj_min_trades),
        rc_spa=replace(cfg0.rc_spa, block_size=block_size),
    )

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(cfg, cfg.results_dir / f"config_notebook_{tf}.json")

    apply_order_size_buffer(max_rel_size=float(cfg.bt.max_rel_size))

    data_by_symbol = load_all_symbols(cfg.data_paths, project_root=project_root, logger=logger)

    registry = build_registry(timeframe=tf)
    bench = get_benchmark()
    if cfg.debug.enabled:
        registry = maybe_filter_debug(registry, enabled=True, only_strategies=cfg.debug.only_strategies)

    folds_by_symbol = {}
    for sym, df in data_by_symbol.items():
        folds = generate_folds(
            df.index,
            train_years=cfg.wf.train_years,
            test_months=cfg.wf.test_months,
            step_months=cfg.wf.step_months,
            warmup_bars=cfg.wf.warmup_bars,
            require_full_warmup=cfg.wf.require_full_warmup,
        )
        folds_by_symbol[sym] = folds
        if logger is not None:
            logger.info("%s | %s folds=%d", tf, sym, len(folds))

    return TimeframeContext(
        tf=tf,
        cfg=cfg,
        ppy=ppy,
        symbols=list(data_by_symbol.keys()),
        data_by_symbol=data_by_symbol,
        folds_by_symbol=folds_by_symbol,
        registry=registry,
        bench=bench,
    )


def run_wf_phase(
    ctx: TimeframeContext,
    *,
    report_mod: Any = None,
    logger: Any = None,
) -> WFOutputs:
    if report_mod is None:
        from . import report as report_mod

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        folds_best, returns_oos, bench_oos, heatmaps, bench_folds = run_full_experiment(
            cfg=ctx.cfg,
            data_by_symbol=ctx.data_by_symbol,
            folds_by_symbol=ctx.folds_by_symbol,
            registry=ctx.registry,
            bench=ctx.bench,
            report_mod=report_mod,
            logger=logger,
            periods_per_year=ctx.ppy,
        )

    return WFOutputs(
        folds_best=folds_best,
        returns_oos=returns_oos,
        bench_oos=bench_oos,
        bench_folds=bench_folds,
        heatmaps=heatmaps,
    )


def _build_benchmark_comparison(
    *,
    folds_best: pd.DataFrame,
    bench_folds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bench_folds.empty or folds_best.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = folds_best.merge(
        bench_folds[["symbol", "cost", "fold_id", "bench_calmar", "bench_total_return"]],
        on=["symbol", "cost", "fold_id"],
        how="left",
        validate="many_to_one",
    )
    merged["beat_bench_calmar"] = (merged["oos_calmar"] > merged["bench_calmar"]).astype(int)
    merged["beat_bench_total_return"] = (merged["oos_total_return"] > merged["bench_total_return"]).astype(int)
    merged["excess_calmar"] = merged["oos_calmar"] - merged["bench_calmar"]
    merged["excess_total_return"] = merged["oos_total_return"] - merged["bench_total_return"]

    beat = (
        merged.groupby(["symbol", "cost", "strategy_id"])
        .agg(
            folds=("fold_id", "nunique"),
            beat_calmar_share=("beat_bench_calmar", "mean"),
            beat_return_share=("beat_bench_total_return", "mean"),
            calmar_median=("oos_calmar", "median"),
            calmar_worst=("oos_calmar", "min"),
        )
        .reset_index()
    )
    excess = (
        merged.groupby(["symbol", "cost", "strategy_id"])
        .agg(
            folds=("fold_id", "nunique"),
            excess_calmar_median=("excess_calmar", "median"),
            excess_return_median=("excess_total_return", "median"),
        )
        .reset_index()
    )
    return beat, excess


def _plot_top_equity_drawdown(
    *,
    returns_oos: pd.DataFrame,
    bench_oos: pd.DataFrame,
    stitched: pd.DataFrame,
    bench_id: str,
    figures_dir: Path,
    top_k: int = 3,
    report_mod: Any,
) -> None:
    rets = _ensure_date_col(returns_oos)
    bench = _ensure_date_col(bench_oos)

    if stitched.empty:
        return

    ranked = stitched.sort_values(["symbol", "cost", "Calmar"], ascending=[True, True, False])
    for (sym, cost), group in ranked.groupby(["symbol", "cost"]):
        top = group.head(int(top_k))
        for _, row in top.iterrows():
            sid = str(row["strategy_id"])
            if sid.startswith("BENCH:"):
                continue

            r = rets[
                (rets["symbol"] == sym) & (rets["cost"] == cost) & (rets["strategy_id"] == sid)
            ].copy()
            b = bench[
                (bench["symbol"] == sym) & (bench["cost"] == cost) & (bench["strategy_id"] == bench_id)
            ].copy()
            if r.empty or b.empty:
                continue

            rs = r.set_index("date")["ret"].astype(float)
            bs = b.set_index("date")["ret"].astype(float)
            report_mod.plot_equity_and_drawdown(
                returns=rs,
                bench_returns=bs,
                title=f"{sid} vs Buy&Hold | {sym} | {cost}",
                out_path=figures_dir / f"equity_dd__{sym}__{cost}__{sid.replace(':', '_')}.png",
            )


def run_reports_phase(
    ctx: TimeframeContext,
    wf: WFOutputs,
    *,
    report_mod: Any = None,
    top_k_plots: int = 3,
    save_plots: bool = True,
    logger: Any = None,
) -> ReportsOutputs:
    if report_mod is None:
        from . import report as report_mod

    leaderboard = report_mod.leaderboard_by_strategy(wf.folds_best)
    stitched = report_mod.stitched_metrics(wf.returns_oos, periods_per_year=ctx.ppy)
    bench_stitched = report_mod.stitched_metrics(wf.bench_oos, periods_per_year=ctx.ppy)

    safe_write_parquet(leaderboard, ctx.cfg.results_dir / "leaderboard.parquet")
    safe_write_parquet(stitched, ctx.cfg.results_dir / "stitched_metrics.parquet")
    safe_write_parquet(bench_stitched, ctx.cfg.results_dir / "bench_stitched_metrics.parquet")

    beat, excess = _build_benchmark_comparison(folds_best=wf.folds_best, bench_folds=wf.bench_folds)
    safe_write_parquet(beat, ctx.cfg.results_dir / "beat_benchmark_summary.parquet")
    safe_write_parquet(excess, ctx.cfg.results_dir / "excess_vs_benchmark_summary.parquet")

    if save_plots:
        _plot_top_equity_drawdown(
            returns_oos=wf.returns_oos,
            bench_oos=wf.bench_oos,
            stitched=stitched,
            bench_id=ctx.bench.strategy_id,
            figures_dir=ctx.cfg.figures_dir,
            top_k=top_k_plots,
            report_mod=report_mod,
        )

    if logger is not None:
        logger.info(
            "REPORTS DONE %s | leaderboard=%s | stitched=%s",
            ctx.tf,
            leaderboard.shape,
            stitched.shape,
        )

    return ReportsOutputs(
        leaderboard=leaderboard,
        stitched=stitched,
        bench_stitched=bench_stitched,
        beat=beat,
        excess=excess,
    )


def run_tests_phase(
    ctx: TimeframeContext,
    wf: WFOutputs,
    *,
    st_module: Any = None,
    lambdas: Optional[list[float]] = None,
    lam_star: float = 1.0,
    alpha: float = 0.05,
    pvalue_type: str = "consistent",
    max_show: int = 20,
    logger: Any = None,
) -> TestsOutputs:
    if st_module is None:
        from . import stats_tests as st_module

    from .rc_spa import run_rc_spa_returns, run_rc_spa_utility, run_spa_better_strategies_table

    lambdas = lambdas or [1.0]

    rcspa = run_rc_spa_returns(
        symbols=ctx.symbols,
        costs=ctx.cfg.costs,
        returns_oos_long=wf.returns_oos,
        bench_long=wf.bench_oos,
        st_module=st_module,
        block_size=int(ctx.cfg.rc_spa.block_size),
        reps=int(ctx.cfg.rc_spa.reps),
        seed=int(ctx.cfg.rc_spa.seed),
        studentize=bool(ctx.cfg.rc_spa.studentize),
        alpha=alpha,
        logger=logger,
    )
    safe_write_parquet(rcspa, ctx.cfg.results_dir / "rc_spa_returns.parquet")

    rcspa_util = run_rc_spa_utility(
        symbols=ctx.symbols,
        costs=ctx.cfg.costs,
        returns_oos_long=wf.returns_oos,
        bench_long=wf.bench_oos,
        st_module=st_module,
        lambdas=lambdas,
        block_size=int(ctx.cfg.rc_spa.block_size),
        reps=int(ctx.cfg.rc_spa.reps),
        seed=int(ctx.cfg.rc_spa.seed),
        studentize=bool(ctx.cfg.rc_spa.studentize),
        alpha=alpha,
        logger=logger,
    )
    safe_write_parquet(rcspa_util, ctx.cfg.results_dir / "rc_spa_utility.parquet")

    try:
        metrics_fn = partial(compute_metrics_from_returns, periods_per_year=ctx.ppy)
        spa_tbl, spa_tbl_view = run_spa_better_strategies_table(
            symbols=ctx.symbols,
            costs=ctx.cfg.costs,
            returns_oos_long=wf.returns_oos,
            bench_long=wf.bench_oos,
            compute_metrics_fn=metrics_fn,
            block_size=int(ctx.cfg.rc_spa.block_size),
            reps=int(ctx.cfg.rc_spa.reps),
            seed=int(ctx.cfg.rc_spa.seed),
            studentize=bool(ctx.cfg.rc_spa.studentize),
            lam_star=float(lam_star),
            alpha=alpha,
            pvalue_type=pvalue_type,
            max_show=int(max_show),
            logger=logger,
        )
    except Exception as exc:
        if logger is not None:
            logger.warning("SPA better table skipped for %s: %s", ctx.tf, exc)
        spa_tbl, spa_tbl_view = pd.DataFrame(), pd.DataFrame()

    safe_write_parquet(spa_tbl, ctx.cfg.results_dir / "spa_better_table.parquet")
    safe_write_parquet(spa_tbl_view, ctx.cfg.results_dir / "spa_better_table_view.parquet")

    return TestsOutputs(
        rcspa=rcspa,
        rcspa_util=rcspa_util,
        spa_tbl=spa_tbl,
        spa_tbl_view=spa_tbl_view,
    )


def run_robustness_phase(
    ctx: TimeframeContext,
    wf: WFOutputs,
    reports: ReportsOutputs,
    *,
    robust_min_bars: int,
    subperiods: Mapping[str, tuple[str, str]],
    lam_star: float = 1.0,
    top_n: int = 5,
    top_k: int = 3,
    bench_id: str = "BENCH:BuyHold",
    logger: Any = None,
) -> RobustnessOutputs:
    from . import robustness as rb

    sensitivity_calmar = rb.cost_sensitivity_top_calmar(
        cfg=ctx.cfg,
        stitched=reports.stitched,
        returns_oos_all=wf.returns_oos,
        bench_oos_all=wf.bench_oos,
        symbols=ctx.symbols,
        topN=int(top_n),
        min_bars=int(robust_min_bars),
        bench_id=bench_id,
        periods_per_year=ctx.ppy,
    )
    sensitivity_utility = rb.cost_sensitivity_top_utility(
        cfg=ctx.cfg,
        stitched=reports.stitched,
        returns_oos_all=wf.returns_oos,
        bench_oos_all=wf.bench_oos,
        symbols=ctx.symbols,
        lam_star=float(lam_star),
        topN=int(top_n),
        min_bars=int(robust_min_bars),
        bench_id=bench_id,
        periods_per_year=ctx.ppy,
    )
    subperiod_calmar = rb.subperiods_top_calmar(
        cfg=ctx.cfg,
        stitched=reports.stitched,
        returns_oos_all=wf.returns_oos,
        bench_oos_all=wf.bench_oos,
        symbols=ctx.symbols,
        subperiods=dict(subperiods),
        topK=int(top_k),
        min_bars=int(robust_min_bars),
        bench_id=bench_id,
        periods_per_year=ctx.ppy,
    )
    subperiod_utility = rb.subperiods_top_utility(
        cfg=ctx.cfg,
        stitched=reports.stitched,
        returns_oos_all=wf.returns_oos,
        bench_oos_all=wf.bench_oos,
        symbols=ctx.symbols,
        subperiods=dict(subperiods),
        lam_star=float(lam_star),
        topK=int(top_k),
        min_bars=int(robust_min_bars),
        bench_id=bench_id,
        periods_per_year=ctx.ppy,
    )

    if logger is not None:
        logger.info(
            "ROBUST DONE %s | sens_cal=%s | sub_cal=%s",
            ctx.tf,
            sensitivity_calmar.shape,
            subperiod_calmar.shape,
        )

    return RobustnessOutputs(
        sensitivity_calmar=sensitivity_calmar,
        sensitivity_utility=sensitivity_utility,
        subperiod_calmar=subperiod_calmar,
        subperiod_utility=subperiod_utility,
    )
