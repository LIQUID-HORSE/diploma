from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from functools import partial
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

from src.data_io import load_all_symbols
from src.experiment_config import default_config, save_config_json
from src.strategy_registry import StrategySpec, build_registry, get_benchmark
from src.utils.sizing_patch import apply_order_size_buffer
from src.wf import generate_folds
from src.wf_experiment import run_full_experiment


TIMEFRAME = "1d"
PERIODS_PER_YEAR = 365

SYMBOL_TO_ACTIVE = {
    "BTCUSDT": "btc",
    "ETHUSDT": "eth",
    "XRPUSDT": "xrp",
    "DOGEUSDT": "doge",
}


def _resolve_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "src").exists():
        return cwd
    if (cwd.parent / "src").exists():
        return cwd.parent
    raise FileNotFoundError(f"Cannot locate project root with 'src' directory from cwd={cwd}")


def _build_data_paths(project_root: Path) -> Dict[str, Path]:
    return {
        symbol: project_root / "data" / "merged_onchain_1d" / f"{active}-1d-merged.csv"
        for symbol, active in SYMBOL_TO_ACTIVE.items()
    }


def _filter_onchain_registry(registry: List[StrategySpec]) -> List[StrategySpec]:
    return [s for s in registry if str(s.group).lower() == "onchain"]


def _configure_backtesting_multiprocessing(
    *,
    logger: logging.Logger,
    no_mp: bool,
    mp_workers: int | None,
) -> None:
    if no_mp:
        logger.info("Multiprocessing optimization disabled via --no-mp.")
        return

    try:
        import backtesting  # Imported lazily to keep startup robust.
    except Exception as exc:
        logger.warning("Cannot import backtesting to configure multiprocessing pool: %s", exc)
        return

    if mp_workers is None:
        backtesting.Pool = mp.Pool
        logger.info("Configured backtesting.Pool = multiprocessing.Pool (default workers).")
        return

    workers = max(1, int(mp_workers))
    backtesting.Pool = partial(mp.Pool, processes=workers)
    logger.info("Configured backtesting.Pool = multiprocessing.Pool(processes=%d).", workers)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run WF experiment for on-chain strategies only (1d, 4 symbols).")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/data wiring without running WF.")
    parser.add_argument("--max-strategies", type=int, default=None, help="Optional cap for debug runs.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR).")
    parser.add_argument("--no-mp", action="store_true", help="Disable multiprocessing pool for optimization.")
    parser.add_argument(
        "--mp-workers",
        type=int,
        default=None,
        help="Optional number of worker processes for optimization (default: auto).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("runner_onchain")
    mp.freeze_support()
    _configure_backtesting_multiprocessing(
        logger=logger,
        no_mp=bool(args.no_mp),
        mp_workers=args.mp_workers,
    )

    project_root = _resolve_project_root()
    logger.info("PROJECT_ROOT=%s", project_root)

    cfg0 = default_config(project_root)
    data_paths = _build_data_paths(project_root)

    missing = [str(p) for p in data_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing merged on-chain files: {missing}")

    results_dir = project_root / "results" / "runner_onchain"
    figures_dir = project_root / "figures" / "runner_onchain"

    cfg = replace(
        cfg0,
        data_paths=data_paths,
        out_dir=project_root,
        results_dir=results_dir,
        figures_dir=figures_dir,
        wf=replace(cfg0.wf, warmup_bars=300),
        objective=replace(cfg0.objective, min_trades=10),
    )

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(cfg, cfg.results_dir / "config_runner_onchain.json")

    apply_order_size_buffer(max_rel_size=float(cfg.bt.max_rel_size))

    registry_all = build_registry(timeframe=TIMEFRAME)
    registry = _filter_onchain_registry(registry_all)
    if args.max_strategies is not None:
        registry = registry[: max(0, int(args.max_strategies))]
    if not registry:
        raise RuntimeError("On-chain registry is empty after filtering.")

    bench = get_benchmark()
    logger.info("On-chain strategies: %s", [s.strategy_id for s in registry])
    logger.info("Benchmark: %s", bench.strategy_id)

    data_by_symbol = load_all_symbols(cfg.data_paths, project_root=project_root, logger=logger)

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
        logger.info("%s | folds=%d", sym, len(folds))

    if args.dry_run:
        logger.info("Dry-run OK. Skipping run_full_experiment.")
        return 0

    folds_best_all, returns_oos_all, bench_oos_all, heatmaps_all, bench_folds_all = run_full_experiment(
        cfg=cfg,
        data_by_symbol=data_by_symbol,
        folds_by_symbol=folds_by_symbol,
        registry=registry,
        bench=bench,
        report_mod=None,
        logger=logger,
        periods_per_year=PERIODS_PER_YEAR,
    )

    logger.info(
        "DONE | folds_best=%s | returns_oos=%s | bench_oos=%s | bench_folds=%s | heatmaps=%s",
        folds_best_all.shape,
        returns_oos_all.shape,
        bench_oos_all.shape,
        bench_folds_all.shape,
        None if heatmaps_all is None else heatmaps_all.shape,
    )
    logger.info("Artifacts written to %s", cfg.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
