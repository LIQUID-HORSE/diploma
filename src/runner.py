"""Experiment runner: walk-forward optimize(train) -> run(test) -> stitch OOS.

This module implements the orchestration described in the specification:
- For each fold:
    1) Optimize strategy parameters on train window (grid search; objective=Calmar with penalties)
    2) Run best parameters on (warmup buffer + test) window, but disable trading before test_start
    3) Compute OOS metrics on pure test window
- Stitch OOS daily returns across folds into one out-of-sample series per strategy.
- Persist artifacts to results/ and figures/ (plots are handled in report.py).

Data format
-----------
CSV files in data/ with at least:
    datetime_utc, open, high, low, close, volume, ...

Only OHLCV is used; other columns are ignored.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import re
import warnings
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# These kwargs are considered *required* for comparability with the project spec.
# If the installed backtesting.py does not support them, we abort rather than
# silently dropping them (which would invalidate results).
_REQUIRED_BACKTEST_KWARGS = {"cash", "commission", "spread", "trade_on_close", "exclusive_orders"}

from .metrics import (
    COST_SCENARIOS,
    CostScenario,
    compute_metrics_from_returns,
    daily_returns_from_backtesting_stats,
    objective_calmar_with_sanity_penalties,
)
from .wf import WFFold, generate_folds

from .strategies.trend import DonchianBreakout, EMACrossover, SMACrossover
from .strategies.momentum import TSMomentum
from .strategies.meanrev import BollingerMeanReversion, RSIMeanReversion, ZScoreMeanReversion
from .strategies.synergy import (
    BreakoutConfirmMA,
    MA200FilterBollinger,
    MACrossTSMOMConfirm,
    MAFilterRSI,
    SimpleEnsemble,
)
from .strategies.benchmarks import BuyHold


@dataclass(frozen=True)
class StrategySpec:
    code: str
    name: str
    cls: Any  # backtesting.Strategy subclass
    param_grid: Dict[str, Sequence[Any]]
    constraint: Optional[Callable[[Any], bool]] = None
    group: str = ""
    save_heatmap: bool = False

    @property
    def id(self) -> str:
        return f"{self.code}:{self.name}"


def strategy_registry() -> List[StrategySpec]:
    """Canonical strategy registry for the project."""
    return [
        StrategySpec(
            code="T1",
            name="SMA_Crossover",
            cls=SMACrossover,
            param_grid={
                "fast": [5, 10, 20, 30, 50],
                "slow": [60, 100, 150, 200, 250, 300],
            },
            constraint=lambda p: p.fast < p.slow,
            group="trend",
            save_heatmap=True,
        ),
        StrategySpec(
            code="T2",
            name="EMA_Crossover",
            cls=EMACrossover,
            param_grid={
                "fast": [5, 10, 20, 30, 50],
                "slow": [60, 100, 150, 200, 250, 300],
            },
            constraint=lambda p: p.fast < p.slow,
            group="trend",
        ),
        StrategySpec(
            code="T3",
            name="Donchian_Breakout",
            cls=DonchianBreakout,
            param_grid={"N": [20, 50, 100, 200], "exit": [10, 20, 50]},
            group="trend",
        ),
        StrategySpec(
            code="M1",
            name="TSMOM",
            cls=TSMomentum,
            param_grid={"L": [20, 60, 120, 252]},
            group="momentum",
        ),
        StrategySpec(
            code="R1",
            name="RSI_MR",
            cls=RSIMeanReversion,
            param_grid={"n": [7, 14, 21], "low": [20, 30, 40], "exit": [45, 50, 55]},
            group="meanrev",
        ),
        StrategySpec(
            code="R2",
            name="Bollinger_MR",
            cls=BollingerMeanReversion,
            # One of the provided tables lists exit={midline,fixed}; we implement both modes.
            param_grid={"n": [20, 50, 100], "k": [1.5, 2.0, 2.5], "exit_mode": ["midline", "fixed"]},
            group="meanrev",
        ),
        StrategySpec(
            code="R3",
            name="ZScore_MR",
            cls=ZScoreMeanReversion,
            param_grid={
                "n": [20, 50, 100, 200],
                "entry_z": [1.0, 1.5, 2.0, 2.5],
                "exit_z": [0.0, 0.5, 1.0],
            },
            group="meanrev",
        ),
        # Synergies (small grids)
        StrategySpec(
            code="S1",
            name="MAFilter_RSI_MR",
            cls=MAFilterRSI,
            param_grid={"M": [100, 200], "n": [14, 21], "low": [30, 40], "exit": [50, 55]},
            group="synergy",
        ),
        StrategySpec(
            code="S2",
            name="MA200Filter_Bollinger_MR",
            cls=MA200FilterBollinger,
            param_grid={"n": [20, 50], "k": [2.0, 2.5]},
            group="synergy",
        ),
        StrategySpec(
            code="S3",
            name="Breakout_Confirm_MA",
            cls=BreakoutConfirmMA,
            param_grid={"N": [50, 100, 200], "M": [100, 200]},
            group="synergy",
        ),
        StrategySpec(
            code="S4",
            name="MA_Confirm_TSMOM",
            cls=MACrossTSMOMConfirm,
            param_grid={"fast": [10, 20, 30], "slow": [100, 200], "L": [60, 120, 252]},
            constraint=lambda p: p.fast < p.slow,
            group="synergy",
        ),
        StrategySpec(
            code="S5",
            name="Ensemble_3Signals",
            cls=SimpleEnsemble,
            param_grid={"ma_pair": [(20, 200), (50, 200)], "N": [50, 200], "L": [120, 252]},
            group="synergy",
        ),
    ]


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime_utc" not in df.columns:
        raise ValueError(f"Expected datetime_utc column in {path}")
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.sort_values("datetime_utc").drop_duplicates("datetime_utc", keep="last")
    df = df.set_index("datetime_utc")

    rename = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    df = df.rename(columns=rename)
    missing = [c for c in ["Open", "High", "Low", "Close"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out.index = out.index.tz_convert(None)  # compatibility with backtesting.py
    return out


def discover_symbol_files(data_dir: Path) -> Dict[str, Path]:
    files = sorted(data_dir.glob("*.csv"))
    return {f.stem: f for f in files}


def _strategy_with_start_date(cls: Any, start_date: Optional[pd.Timestamp]) -> Any:
    if start_date is None:
        return cls
    return type(
        f"{cls.__name__}__Start_{pd.Timestamp(start_date).date()}",
        (cls,),
        {"start_date": pd.Timestamp(start_date)},
    )


def _make_backtest(data: pd.DataFrame, StrategyCls: Any, *, cash: float, commission: float, spread: float) -> Any:
    """Create Backtest with version-robust kwargs handling.

    Why so defensive?
    -----------------
    In some backtesting.py versions, `Backtest(...)` may not accept newer kwargs
    (e.g. `spread`). A naive "drop unknown kwargs and retry" can silently disable
    transaction costs or execution settings, producing invalid results.

    Here we:
    1) Try to filter kwargs by inspecting the Backtest.__init__ signature.
    2) If we still hit a TypeError, parse the "unexpected keyword" and retry.
    3) If a *required* kwarg cannot be passed, we raise an explicit error.
    """
    from backtesting import Backtest  # local import for better error messages

    kwargs: Dict[str, Any] = {
        "cash": cash,
        "commission": commission,
        "spread": spread,
        "trade_on_close": False,
        "exclusive_orders": True,
        "hedging": False,
        "margin": 1.0,
    }
    dropped: List[str] = []

    # First attempt: signature-based filtering (best signal of compatibility).
    try:
        sig = inspect.signature(Backtest.__init__)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not has_var_kw:
            supported = set(params.keys())
            supported.discard("self")
            unsupported = [k for k in list(kwargs.keys()) if k not in supported]
            if unsupported:
                hard = [k for k in unsupported if k in _REQUIRED_BACKTEST_KWARGS]
                if hard:
                    raise RuntimeError(
                        "Installed backtesting.py does not support required Backtest() kwargs "
                        f"{hard}. These are required by the project spec (costs/execution). "
                        "Please upgrade backtesting.py to a version that supports them."
                    )
                for k in unsupported:
                    kwargs.pop(k, None)
                dropped.extend(unsupported)
    except Exception:
        # If signature inspection fails for any reason, we fall back to runtime parsing below.
        pass

    # Second attempt: instantiate; on TypeError, drop truly unknown kwargs.
    while True:
        try:
            bt = Backtest(data, StrategyCls, **kwargs)
            break
        except TypeError as e:
            msg = str(e)
            m = re.search(r"unexpected keyword argument ['\"](?P<kw>\w+)['\"]", msg)
            if not m:
                # Some other TypeError (not about kwargs) – surface it.
                raise
            bad_kw = m.group("kw")
            if bad_kw in _REQUIRED_BACKTEST_KWARGS:
                raise RuntimeError(
                    "Installed backtesting.py rejected a required Backtest() kwarg "
                    f"'{bad_kw}'. This would silently change costs/execution. "
                    "Please upgrade backtesting.py (recommended) or adjust the project settings."
                ) from e
            if bad_kw not in kwargs:
                raise
            kwargs.pop(bad_kw)
            dropped.append(bad_kw)

    if dropped:
        warnings.warn(
            "Backtest() did not accept some kwargs and they were dropped: " + ", ".join(sorted(set(dropped))),
            RuntimeWarning,
            stacklevel=2,
        )
        logger.warning("Backtest() kwargs dropped due to version mismatch: %s", sorted(set(dropped)))

    return bt


def _extract_best_params(stats: Any, param_names: Iterable[str]) -> Dict[str, Any]:
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


def _optimize_on_train(
    train_df: pd.DataFrame,
    spec: StrategySpec,
    *,
    cost: CostScenario,
    cash: float,
    min_trades: int,
    min_exposure: float,
    return_heatmap: bool,
) -> Tuple[Any, Optional[pd.Series], Dict[str, Any]]:
    bt = _make_backtest(train_df, spec.cls, cash=cash, commission=cost.commission, spread=cost.spread)

    maximize = lambda s: objective_calmar_with_sanity_penalties(  # noqa: E731
        s, min_trades=min_trades, min_exposure=min_exposure
    )

    if return_heatmap:
        res = bt.optimize(
            **spec.param_grid,
            maximize=maximize,
            constraint=spec.constraint,
            return_heatmap=True,
        )
        if isinstance(res, tuple) and len(res) == 2:
            best_stats, heatmap = res
        else:
            best_stats, heatmap = res, None
    else:
        best_stats = bt.optimize(**spec.param_grid, maximize=maximize, constraint=spec.constraint)
        heatmap = None

    best_params = _extract_best_params(best_stats, spec.param_grid.keys())
    return best_stats, heatmap, best_params


def _run_on_test(
    test_df_with_buffer: pd.DataFrame,
    spec: StrategySpec,
    *,
    start_date: pd.Timestamp,
    params: Dict[str, Any],
    cost: CostScenario,
    cash: float,
) -> Any:
    StrategyCls = _strategy_with_start_date(spec.cls, start_date)
    bt = _make_backtest(test_df_with_buffer, StrategyCls, cash=cash, commission=cost.commission, spread=cost.spread)
    return bt.run(**params)


def _run_buyhold_on_test(
    test_df_with_buffer: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    cost: CostScenario,
    cash: float,
) -> Any:
    StrategyCls = _strategy_with_start_date(BuyHold, start_date)
    bt = _make_backtest(test_df_with_buffer, StrategyCls, cash=cash, commission=cost.commission, spread=cost.spread)
    return bt.run()


def _safe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # Fallback to CSV if parquet engine isn't installed.
        df.to_csv(path.with_suffix(".csv"), index=False)


def run_walk_forward(
    *,
    symbol: str,
    data: pd.DataFrame,
    out_dir: Path,
    cash: float = 10_000.0,
    costs: Sequence[CostScenario] = (COST_SCENARIOS["Base"],),
    strategies: Optional[Sequence[StrategySpec]] = None,
    train_years: int = 3,
    test_months: int = 6,
    step_months: int = 6,
    warmup_bars: int = 252,
    # Defaults are intentionally conservative to reduce the chance that the optimizer
    # picks a "lucky" configuration with too few trades/exposure.
    min_trades: int = 10,
    min_exposure: float = 0.10,
    save_heatmaps: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full WF pipeline for one symbol.

    Returns:
      folds_best_df: one row per strategy per fold with best params and OOS metrics
      returns_oos_df: long-format stitched OOS returns (one row per day per strategy)
      bench_oos_df: long-format stitched benchmark returns
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if strategies is None:
        strategies = strategy_registry()

    folds = generate_folds(
        data.index,
        train_years=train_years,
        test_months=test_months,
        step_months=step_months,
        warmup_bars=warmup_bars,
        require_full_warmup=True,
    )
    if not folds:
        raise ValueError(f"No folds generated for {symbol}. Check data coverage and WF parameters.")

    folds_rows: List[Dict[str, Any]] = []
    stitched_returns: List[Dict[str, Any]] = []
    stitched_bench: List[Dict[str, Any]] = []
    heatmap_rows: List[Dict[str, Any]] = []

    for cost in costs:
        # Stitch per strategy within each cost scenario
        per_strat_returns: Dict[str, List[pd.Series]] = {spec.id: [] for spec in strategies}
        bench_returns: List[pd.Series] = []

        for fold in folds:
            train_df = fold.slice_train(data)
            test_df = fold.slice_test_with_buffer(data)

            # Benchmark on test
            bench_stats = _run_buyhold_on_test(test_df, start_date=fold.test_start, cost=cost, cash=cash)
            bench_ret = daily_returns_from_backtesting_stats(bench_stats)
            bench_ret = bench_ret.loc[(bench_ret.index >= fold.test_start) & (bench_ret.index < fold.test_end)]
            bench_returns.append(bench_ret)

            for spec in strategies:
                best_train_stats, heatmap, best_params = _optimize_on_train(
                    train_df,
                    spec,
                    cost=cost,
                    cash=cash,
                    min_trades=min_trades,
                    min_exposure=min_exposure,
                    return_heatmap=bool(save_heatmaps and spec.save_heatmap),
                )

                test_stats = _run_on_test(
                    test_df,
                    spec,
                    start_date=fold.test_start,
                    params=best_params,
                    cost=cost,
                    cash=cash,
                )
                oos_ret = daily_returns_from_backtesting_stats(test_stats)
                oos_ret = oos_ret.loc[(oos_ret.index >= fold.test_start) & (oos_ret.index < fold.test_end)]
                per_strat_returns[spec.id].append(oos_ret)

                fold_metrics = compute_metrics_from_returns(oos_ret)
                # Add trade stats (should be test-only since strategy is gated by start_date)
                try:
                    n_trades = int(test_stats.get("# Trades", 0))
                except Exception:
                    n_trades = 0
                try:
                    exposure = float(test_stats.get("Exposure Time [%]", 0.0))
                except Exception:
                    exposure = 0.0

                # Train objective value
                try:
                    train_obj = float(
                        objective_calmar_with_sanity_penalties(
                            best_train_stats, min_trades=min_trades, min_exposure=min_exposure
                        )
                    )
                except Exception:
                    train_obj = float("nan")

                folds_rows.append(
                    {
                        "symbol": symbol,
                        "cost": cost.name,
                        "fold_id": fold.fold_id,
                        "train_start": fold.train_start,
                        "train_end": fold.train_end,
                        "test_start": fold.test_start,
                        "test_end": fold.test_end,
                        "strategy_code": spec.code,
                        "strategy_name": spec.name,
                        "strategy_id": spec.id,
                        "params": json.dumps(best_params, default=str),
                        "train_objective": train_obj,
                        "oos_total_return": fold_metrics["Total Return"],
                        "oos_cagr": fold_metrics["CAGR"],
                        "oos_sharpe": fold_metrics["Sharpe"],
                        "oos_sortino": fold_metrics["Sortino"],
                        "oos_maxdd": fold_metrics["MaxDD"],
                        "oos_calmar": fold_metrics["Calmar"],
                        "oos_ann_vol": fold_metrics["Ann. Vol"],
                        "oos_n_bars": fold_metrics["n_bars"],
                        "oos_trades": n_trades,
                        "oos_exposure_pct": exposure,
                    }
                )

                if heatmap is not None:
                    # heatmap is a Series with MultiIndex over params
                    try:
                        hm = heatmap.rename("objective").reset_index()
                        hm["symbol"] = symbol
                        hm["cost"] = cost.name
                        hm["fold_id"] = fold.fold_id
                        hm["strategy_id"] = spec.id
                        heatmap_rows.append(hm)
                    except Exception:
                        pass

        # Stitch returns for this cost scenario
        for strat_id, parts in per_strat_returns.items():
            if not parts:
                continue
            stitched = pd.concat(parts).sort_index()
            stitched = stitched[~stitched.index.duplicated(keep="first")]
            for dt, r in stitched.items():
                stitched_returns.append({"symbol": symbol, "cost": cost.name, "strategy_id": strat_id, "date": dt, "ret": float(r)})

        bench_stitched = pd.concat(bench_returns).sort_index()
        bench_stitched = bench_stitched[~bench_stitched.index.duplicated(keep="first")]
        for dt, r in bench_stitched.items():
            stitched_bench.append({"symbol": symbol, "cost": cost.name, "strategy_id": "BENCH:BuyHold", "date": dt, "ret": float(r)})

    folds_best_df = pd.DataFrame(folds_rows)
    returns_oos_df = pd.DataFrame(stitched_returns)
    bench_oos_df = pd.DataFrame(stitched_bench)

    _safe_to_parquet(folds_best_df, results_dir / "folds_best.parquet")
    _safe_to_parquet(returns_oos_df, results_dir / "returns_oos.parquet")
    _safe_to_parquet(bench_oos_df, results_dir / "bench_returns_oos.parquet")

    if heatmap_rows:
        heatmaps_df = pd.concat(heatmap_rows, ignore_index=True)
        _safe_to_parquet(heatmaps_df, results_dir / "train_heatmaps.parquet")

    return folds_best_df, returns_oos_df, bench_oos_df


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--symbols", nargs="*", default=None, help="Symbols to run (default: all csv stems in data-dir)")
    p.add_argument("--costs", nargs="*", default=["Base"], help="Cost scenarios: Low Base High")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--train-years", type=int, default=3)
    p.add_argument("--test-months", type=int, default=6)
    p.add_argument("--step-months", type=int, default=6)
    p.add_argument("--warmup-bars", type=int, default=252)
    p.add_argument("--min-trades", type=int, default=10)
    p.add_argument("--min-exposure", type=float, default=0.10)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    sym_files = discover_symbol_files(data_dir)
    symbols = args.symbols or sorted(sym_files.keys())
    costs = [COST_SCENARIOS[c] for c in args.costs]

    for sym in symbols:
        if sym not in sym_files:
            raise SystemExit(f"Symbol {sym} not found in {data_dir}")
        df = _load_csv(sym_files[sym])
        run_walk_forward(
            symbol=sym,
            data=df,
            out_dir=out_dir,
            cash=args.cash,
            costs=costs,
            train_years=args.train_years,
            test_months=args.test_months,
            step_months=args.step_months,
            warmup_bars=args.warmup_bars,
            min_trades=args.min_trades,
            min_exposure=args.min_exposure,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
