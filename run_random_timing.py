from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import safe_write_parquet
from src.metrics import COST_SCENARIOS
from src.random_timing import (
    GROUP_KEY,
    assert_unique_by_key,
    build_random_timing_summary,
    compute_random_timing_result,
    deterministic_seed,
    normalize_positions_long,
    normalize_returns_long,
    strategy_family,
)


def _resolve_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "src").exists():
        return cwd
    if (cwd.parent / "src").exists():
        return cwd.parent
    raise FileNotFoundError(f"Cannot locate project root with 'src' directory from cwd={cwd}")


def _build_inputs(project_root: Path) -> dict[str, Path]:
    return {
        "positions_true": project_root / "results" / "poses_true" / "positions_true_oos_long.parquet",
        "exp_1d_returns": project_root / "results" / "runner_exp" / "1d" / "returns_oos.parquet",
        "exp_4h_returns": project_root / "results" / "runner_exp" / "4h" / "returns_oos.parquet",
        "onchain_1d_returns": project_root / "results" / "runner_onchain" / "returns_oos.parquet",
        "bench_1d": project_root / "results" / "runner_exp" / "1d" / "bench_returns_oos.parquet",
        "bench_4h": project_root / "results" / "runner_exp" / "4h" / "bench_returns_oos.parquet",
    }


def _prepare_returns(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    paths = _build_inputs(project_root)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    positions = normalize_positions_long(pd.read_parquet(paths["positions_true"]))
    assert_unique_by_key(positions, key=GROUP_KEY + ["date"], name="positions_true")

    exp_1d = normalize_returns_long(pd.read_parquet(paths["exp_1d_returns"]), timeframe="1d", scenario="u12")
    exp_4h = normalize_returns_long(pd.read_parquet(paths["exp_4h_returns"]), timeframe="4h", scenario="u12")
    onchain_1d = normalize_returns_long(
        pd.read_parquet(paths["onchain_1d_returns"]),
        timeframe="1d",
        scenario="u17_onchain",
    )

    returns_all = pd.concat([exp_1d, exp_4h, onchain_1d], ignore_index=True)
    assert_unique_by_key(returns_all, key=GROUP_KEY + ["date"], name="returns_all")

    bench_1d_u12 = normalize_returns_long(pd.read_parquet(paths["bench_1d"]), timeframe="1d", scenario="u12")
    bench_1d_u17 = bench_1d_u12.copy()
    bench_1d_u17["scenario"] = "u17_onchain"
    bench_4h_u12 = normalize_returns_long(pd.read_parquet(paths["bench_4h"]), timeframe="4h", scenario="u12")
    bench_all = pd.concat([bench_1d_u12, bench_1d_u17, bench_4h_u12], ignore_index=True)
    assert_unique_by_key(bench_all, key=GROUP_KEY + ["date"], name="bench_all")

    return positions, returns_all, bench_all


def _series_by_key(df: pd.DataFrame) -> dict[tuple[str, str, str, str, str], pd.DataFrame]:
    out: dict[tuple[str, str, str, str, str], pd.DataFrame] = {}
    for k, g in df.groupby(GROUP_KEY, sort=False):
        out[(str(k[0]), str(k[1]), str(k[2]), str(k[3]), str(k[4]))] = g.sort_values("date").reset_index(drop=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run random-timing benchmark on stitched OOS outputs.")
    parser.add_argument("--k", type=int, default=1000, help="Number of random realizations per test.")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.5, 1.0, 2.0], help="Lambda values.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument("--chunk-size", type=int, default=200, help="Chunk size for random simulations.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional debug cap for number of strategy groups before lambda expansion.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("random_timing")

    if args.k <= 0:
        raise ValueError("--k must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    project_root = _resolve_project_root()
    out_dir = project_root / "results" / "random_timing-test"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("PROJECT_ROOT=%s", project_root)
    logger.info("OUT_DIR=%s", out_dir)
    logger.info("Params: k=%d lambdas=%s seed=%d chunk_size=%d", args.k, args.lambdas, args.seed, args.chunk_size)

    positions, returns_all, bench_all = _prepare_returns(project_root)
    logger.info("positions shape=%s", positions.shape)
    logger.info("returns_all shape=%s", returns_all.shape)
    logger.info("bench_all shape=%s", bench_all.shape)

    cost_entry_by_name: dict[str, float] = {}
    cost_exit_by_name: dict[str, float] = {}
    for cost_name, scenario in COST_SCENARIOS.items():
        cost_entry_by_name[cost_name] = float(scenario.commission + scenario.spread)
        cost_exit_by_name[cost_name] = float(scenario.commission)

    pos_map = _series_by_key(positions)
    ret_map = _series_by_key(returns_all)
    bench_map = _series_by_key(bench_all)

    group_keys = sorted(pos_map.keys())
    if args.max_groups is not None:
        group_keys = group_keys[: max(0, int(args.max_groups))]
        logger.info("Debug cap applied: max_groups=%d -> running groups=%d", args.max_groups, len(group_keys))

    rows: list[dict[str, object]] = []
    n_total = len(group_keys) * len(args.lambdas)
    n_done = 0
    n_skipped = 0

    for tf, scenario, symbol, cost, strategy_id in group_keys:
        pkey = (tf, scenario, symbol, cost, strategy_id)
        bkey = (tf, scenario, symbol, cost, "BENCH:BuyHold")

        pos_df = pos_map.get(pkey)
        ret_df = ret_map.get(pkey)
        bench_df = bench_map.get(bkey)

        if pos_df is None or ret_df is None or bench_df is None:
            for lam in args.lambdas:
                rows.append(
                    {
                        "timeframe": tf,
                        "scenario": scenario,
                        "symbol": symbol,
                        "cost": cost,
                        "strategy_id": strategy_id,
                        "family": strategy_family(strategy_id),
                        "lam": float(lam),
                        "n_bars": 0,
                        "exposure": np.nan,
                        "n_trades": np.nan,
                        "avg_trade_len": np.nan,
                        "p_enter": np.nan,
                        "p_exit": np.nan,
                        "excess_vs_BH": np.nan,
                        "excess_random_median": np.nan,
                        "excess_random_mean": np.nan,
                        "excess_random_std": np.nan,
                        "p_value": np.nan,
                        "timing_significant": False,
                        "skip_reason": "missing_series",
                    }
                )
                n_skipped += 1
                n_done += 1
            continue

        merged = (
            pos_df[["date", "position"]]
            .merge(ret_df[["date", "ret"]], on="date", how="inner")
            .merge(
                bench_df[["date", "ret"]].rename(columns={"ret": "ret_bench"}),
                on="date",
                how="inner",
            )
            .sort_values("date")
            .reset_index(drop=True)
        )

        pos_arr = merged["position"].to_numpy(dtype=np.int8)
        ret_arr = merged["ret"].to_numpy(dtype=float)
        bench_arr = merged["ret_bench"].to_numpy(dtype=float)
        if str(cost) not in cost_entry_by_name:
            raise KeyError(f"Unknown cost scenario in random timing inputs: {cost!r}")
        entry_cost = float(cost_entry_by_name[str(cost)])
        exit_cost = float(cost_exit_by_name[str(cost)])

        for lam in args.lambdas:
            case_seed = deterministic_seed(
                int(args.seed),
                [tf, scenario, symbol, cost, strategy_id, float(lam)],
            )
            res = compute_random_timing_result(
                position=pos_arr,
                ret_strategy=ret_arr,
                ret_bench=bench_arr,
                lam=float(lam),
                k=int(args.k),
                chunk_size=int(args.chunk_size),
                entry_cost=float(entry_cost),
                exit_cost=float(exit_cost),
                seed=int(case_seed),
            )

            if res.get("skip_reason") is not None:
                n_skipped += 1

            rows.append(
                {
                    "timeframe": tf,
                    "scenario": scenario,
                    "symbol": symbol,
                    "cost": cost,
                    "strategy_id": strategy_id,
                    "family": strategy_family(strategy_id),
                    "lam": float(lam),
                    **res,
                }
            )
            n_done += 1

            if n_done % 100 == 0 or n_done == n_total:
                logger.info("Progress: %d/%d tests done (skipped=%d)", n_done, n_total, n_skipped)

    results = pd.DataFrame(rows)
    if not results.empty:
        key = ["timeframe", "scenario", "symbol", "cost", "strategy_id", "lam"]
        assert_unique_by_key(results, key=key, name="random_timing_results")

    summary = build_random_timing_summary(results)

    safe_write_parquet(results, out_dir / "random_timing_results.parquet")
    safe_write_parquet(summary, out_dir / "random_timing_summary.parquet")

    run_info = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "out_dir": str(out_dir),
        "params": {
            "k": int(args.k),
            "lambdas": [float(x) for x in args.lambdas],
            "seed": int(args.seed),
            "chunk_size": int(args.chunk_size),
            "cost_model": "entry=commission+spread, exit=commission",
            "entry_bps_by_cost": {k: float(v * 10000.0) for k, v in cost_entry_by_name.items()},
            "exit_bps_by_cost": {k: float(v * 10000.0) for k, v in cost_exit_by_name.items()},
            "roundtrip_bps_by_cost": {
                k: float((cost_entry_by_name[k] + cost_exit_by_name[k]) * 10000.0)
                for k in cost_entry_by_name
            },
        },
        "coverage": {
            "positions_rows": int(positions.shape[0]),
            "returns_rows": int(returns_all.shape[0]),
            "bench_rows": int(bench_all.shape[0]),
            "groups": int(len(group_keys)),
            "tests_total": int(n_total),
            "tests_skipped": int(n_skipped),
            "results_rows": int(results.shape[0]),
            "summary_rows": int(summary.shape[0]),
        },
    }
    (out_dir / "random_timing_run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "DONE | results=%s | summary=%s | skipped=%d",
        results.shape,
        summary.shape,
        n_skipped,
    )
    logger.info("Artifacts written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
