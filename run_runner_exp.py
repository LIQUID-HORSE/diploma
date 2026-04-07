from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path
from dataclasses import replace
from typing import Iterable

from src.pipeline_exp import build_timeframe_context, run_wf_phase


DEFAULT_TIMEFRAMES = ("1d", "4h")
SYMBOL_TO_ACTIVE = {
    "BTCUSDT": "btc",
    "ETHUSDT": "eth",
    "BNBUSDT": "bnb",
    "XRPUSDT": "xrp",
    "DOGEUSDT": "doge",
}
OBJECTIVE_MIN_TRADES = {"1d": 10, "4h": 30}
RC_BLOCK_SIZE = {"1d": 10, "4h": 60}


def _resolve_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "src").exists():
        return cwd
    if (cwd.parent / "src").exists():
        return cwd.parent
    raise FileNotFoundError(f"Cannot locate project root with 'src' directory from cwd={cwd}")


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


def _parse_timeframes(values: Iterable[str]) -> list[str]:
    allowed = {"1d", "4h"}
    out: list[str] = []
    for raw in values:
        tf = str(raw).strip().lower()
        if tf not in allowed:
            raise ValueError(f"Unsupported timeframe={raw!r}. Allowed: {sorted(allowed)}")
        if tf not in out:
            out.append(tf)
    if not out:
        raise ValueError("No valid timeframes requested.")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run runner_exp WF phase for selected timeframes.",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=list(DEFAULT_TIMEFRAMES),
        help="One or more timeframes to run: 1d 4h",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build contexts only, skip WF execution.")
    parser.add_argument("--no-mp", action="store_true", help="Disable multiprocessing pool for optimization.")
    parser.add_argument(
        "--mp-workers",
        type=int,
        default=None,
        help="Optional number of worker processes for optimization (default: auto).",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR).")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("runner_exp_wf")

    mp.freeze_support()
    _configure_backtesting_multiprocessing(
        logger=logger,
        no_mp=bool(args.no_mp),
        mp_workers=args.mp_workers,
    )

    project_root = _resolve_project_root()
    logger.info("PROJECT_ROOT=%s", project_root)

    timeframes = _parse_timeframes(args.timeframes)
    logger.info("Requested timeframes: %s", timeframes)

    summary_rows: list[dict[str, object]] = []
    for tf in timeframes:
        logger.info("=== WF RUN START | %s ===", tf)
        ctx = build_timeframe_context(
            project_root=project_root,
            timeframe=tf,
            symbol_to_active=SYMBOL_TO_ACTIVE,
            objective_min_trades=OBJECTIVE_MIN_TRADES,
            rc_block_size=RC_BLOCK_SIZE,
            results_root_name="runner_exp",
            logger=logger,
        )

        # runner_exp is price-only; on-chain strategies are run in runner_onchain pipeline.
        base_registry = [s for s in ctx.registry if str(getattr(s, "group", "")).lower() != "onchain"]
        if len(base_registry) != len(ctx.registry):
            logger.info(
                "%s | filtered on-chain strategies for runner_exp: %d -> %d",
                tf,
                len(ctx.registry),
                len(base_registry),
            )
        ctx = replace(ctx, registry=base_registry)

        if args.dry_run:
            logger.info("Dry-run mode: skipping run_wf_phase for %s", tf)
            summary_rows.append(
                {
                    "timeframe": tf,
                    "status": "dry_run",
                    "results_dir": str(ctx.cfg.results_dir),
                    "figures_dir": str(ctx.cfg.figures_dir),
                }
            )
            continue

        wf = run_wf_phase(ctx, logger=logger)
        summary_rows.append(
            {
                "timeframe": tf,
                "status": "done",
                "results_dir": str(ctx.cfg.results_dir),
                "folds_best_rows": int(wf.folds_best.shape[0]),
                "returns_oos_rows": int(wf.returns_oos.shape[0]),
                "bench_oos_rows": int(wf.bench_oos.shape[0]),
                "bench_folds_rows": int(wf.bench_folds.shape[0]),
            }
        )
        logger.info(
            "WF DONE %s | folds_best=%s | returns_oos=%s | bench_oos=%s",
            tf,
            wf.folds_best.shape,
            wf.returns_oos.shape,
            wf.bench_oos.shape,
        )

    if summary_rows:
        logger.info("Summary:")
        for row in summary_rows:
            logger.info("%s", row)

    logger.info("All requested WF runs finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
