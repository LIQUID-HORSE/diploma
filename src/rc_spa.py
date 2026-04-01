# src/rc_spa.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

from .metrics import CostScenario


def build_universe_matrix(
    returns_oos_long: pd.DataFrame,
    bench_long: pd.DataFrame,
    *,
    symbol: str,
    cost: str,
    drop_bench_like_cols: bool = True,
    fillna: float = 0.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build model universe matrix (T x M) and benchmark series (T).

    Accepts legacy date column names:
      - 'date' (preferred)
      - 'datetime_utc' (legacy)
      - 'index' (legacy)

    Guardrails:
      - (date, strategy_id) must be unique (no silent aggregation in pivot)
      - align models and bench on inner-joined dates
      - optional drop BENCH:* columns if they slipped into returns_oos_long
    """
    def _ensure_date_col(df_in: pd.DataFrame, *, who: str) -> pd.DataFrame:
        df = df_in.copy()
        if "date" not in df.columns:
            if "datetime_utc" in df.columns:
                df = df.rename(columns={"datetime_utc": "date"})
            elif "index" in df.columns:
                df = df.rename(columns={"index": "date"})
            else:
                raise ValueError(
                    f"{who} must contain a 'date' column (or legacy 'datetime_utc'/'index'). "
                    f"Got columns: {list(df.columns)[:30]}"
                )
        df["date"] = pd.to_datetime(df["date"])
        return df

    # --- Normalize inputs
    df = _ensure_date_col(returns_oos_long, who="returns_oos_long")
    df = df[(df["symbol"] == symbol) & (df["cost"] == cost)]

    # Guardrail: each (date,strategy_id) must be unique; otherwise pivot_table would silently aggregate.
    dups = df.duplicated(subset=["date", "strategy_id"]).sum()
    if int(dups) > 0:
        raise RuntimeError(
            f"Duplicate returns rows for {symbol}/{cost}: {dups} duplicate (date,strategy_id) keys. Fix stitching."
        )

    wide = (
        df.pivot_table(index="date", columns="strategy_id", values="ret", aggfunc="mean")
        .sort_index()
    )

    b = _ensure_date_col(bench_long, who="bench_long")
    b = b[(b["symbol"] == symbol) & (b["cost"] == cost)]
    bench = b.set_index("date")["ret"].sort_index()

    # Align on shared dates only
    wide, bench = wide.align(bench, join="inner", axis=0)

    # Clean inf/nan
    wide = wide.replace([np.inf, -np.inf], np.nan).fillna(fillna)
    bench = bench.replace([np.inf, -np.inf], np.nan).fillna(fillna)

    # Drop any benchmark-like columns if they got into returns_oos_long
    if drop_bench_like_cols:
        wide = wide[[c for c in wide.columns if not str(c).startswith("BENCH:")]]

    return wide, bench


def hansen_spa(
    returns_alt: pd.DataFrame,
    returns_bench: pd.Series,
    *,
    block_size: int,
    reps: int,
    seed: int,
    studentize: bool = True,
    alpha: float = 0.05,
) -> Optional[Dict[str, float]]:
    """
    Hansen SPA test using arch.bootstrap.SPA.

    We test models vs benchmark using excess returns:
      d_{t,j} = r_{t,j} - r_{t,bench}

    SPA expects losses (lower is better). We use:
      losses = -d_{t,j}, benchmark losses = 0
    """
    try:
        from arch.bootstrap import SPA  # type: ignore
    except Exception:
        return None

    df, bench = returns_alt.align(returns_bench, join="inner", axis=0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bench = bench.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    excess = df.sub(bench, axis=0)  # d_{t,j}

    models = (-excess).to_numpy()
    benchmark = np.zeros(models.shape[0], dtype=float)

    T = models.shape[0]
    bs = int(block_size)
    if bs >= T:
        bs = max(2, T // 5)

    spa = SPA(
        benchmark,
        models,
        block_size=int(bs),
        reps=int(reps),
        seed=int(seed),
        studentize=bool(studentize),
    )
    spa.compute()

    try:
        pvals = np.asarray(spa.pvalues, dtype=float)
    except Exception:
        return None

    out: Dict[str, float] = {
        "spa_pvalue_lower": float(pvals[0]) if pvals.size > 0 else float("nan"),
        "spa_pvalue_consistent": float(pvals[1]) if pvals.size > 1 else float("nan"),
        "spa_pvalue_upper": float(pvals[2]) if pvals.size > 2 else float("nan"),
        "reps": float(reps),
        "block_size": float(bs),
        "studentize": float(bool(studentize)),
    }

    try:
        better = spa.better_models(pvalue=float(alpha), pvalue_type="consistent")
        out["spa_n_better_models"] = float(len(better))
    except Exception:
        out["spa_n_better_models"] = float("nan")

    return out


def run_rc_spa_returns(
    *,
    symbols: List[str],
    costs: List[str],
    returns_oos_long: pd.DataFrame,
    bench_long: pd.DataFrame,
    st_module: Any,  # expects white_reality_check
    block_size: int,
    reps: int,
    seed: int,
    studentize: bool,
    alpha: float = 0.05,
    logger: Any = None,
) -> pd.DataFrame:
    """
    Run White RC + Hansen SPA (if arch installed) for each (symbol, cost) on raw returns.
    """
    rows: List[Dict[str, Any]] = []

    for sym in symbols:
        for cost_name in costs:
            returns_alt, returns_bench = build_universe_matrix(
                returns_oos_long, bench_long, symbol=sym, cost=cost_name
            )

            if logger is not None:
                logger.info(
                    "RC/SPA universe: %s | %s | T=%d | models=%d",
                    sym, cost_name, len(returns_alt), returns_alt.shape[1]
                )

            rc = st_module.white_reality_check(
                returns_alt=returns_alt,
                returns_bench=returns_bench,
                n_boot=int(reps),
                avg_block_len=int(block_size),
                seed=int(seed),
            )

            spa_res = hansen_spa(
                returns_alt,
                returns_bench,
                block_size=int(block_size),
                reps=int(reps),
                seed=int(seed),
                studentize=bool(studentize),
                alpha=float(alpha),
            )

            row: Dict[str, Any] = {
                "symbol": sym,
                "cost": cost_name,
                "T": int(len(returns_alt)),
                "universe_size": int(returns_alt.shape[1]),
                "rc_pvalue": float(rc.p_value),
                "rc_stat": float(rc.statistic),
                "rc_best_model": str(rc.best_model),
            }

            if spa_res is not None:
                row.update(spa_res)
            else:
                row.update(
                    {
                        "spa_pvalue_lower": np.nan,
                        "spa_pvalue_consistent": np.nan,
                        "spa_pvalue_upper": np.nan,
                        "spa_n_better_models": np.nan,
                    }
                )

            rows.append(row)

    return pd.DataFrame(rows)


# =========================
# Utility-based RC / SPA
# =========================

def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown as a negative fraction (e.g., -0.25 means -25% from peak)."""
    r = returns.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return dd


def utility_penalized_return(returns: pd.Series, *, lam: float) -> pd.Series:
    """
    Penalized utility using worsening drawdown only.

    Steps:
      - compute drawdown depth DD_t = 1 - equity_t/peak_t  (>=0)
      - penalty_t = max(0, DD_t - DD_{t-1})  (only when drawdown deepens)
      - u_t = r_t - lam * penalty_t
    """
    r = returns.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # drawdown depth (positive)
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd_depth = 1.0 - (eq / peak)

    # worsening only (first diff, clipped at 0)
    dd_worsen = dd_depth.diff().fillna(0.0).clip(lower=0.0)

    u = r - float(lam) * dd_worsen
    return u


def build_utility_matrix(
    returns_alt: pd.DataFrame,
    returns_bench: pd.Series,
    *,
    lam: float,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convert returns into utility series for models and bench, aligned on common dates.
    """
    df, bench = returns_alt.align(returns_bench, join="inner", axis=0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bench = bench.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    u_bench = utility_penalized_return(bench, lam=float(lam))

    U = {c: utility_penalized_return(df[c], lam=float(lam)) for c in df.columns}
    u_models = pd.DataFrame(U, index=df.index)

    return u_models, u_bench


def run_rc_spa_utility(
    *,
    symbols: List[str],
    costs: List[str],
    returns_oos_long: pd.DataFrame,
    bench_long: pd.DataFrame,
    st_module: Any,
    lambdas: List[float],
    block_size: int,
    reps: int,
    seed: int,
    studentize: bool,
    alpha: float = 0.05,
    logger: Any = None,
) -> pd.DataFrame:
    """
    Run White RC + Hansen SPA on utility series:
      u_t = r_t - lam * drawdown_depth_t
    for each (symbol, cost, lam).
    """
    rows: List[Dict[str, Any]] = []

    for sym in symbols:
        for cost_name in costs:
            # universe in return space
            returns_alt, returns_bench = build_universe_matrix(
                returns_oos_long, bench_long, symbol=sym, cost=cost_name
            )

            for lam in lambdas:
                u_alt, u_bench = build_utility_matrix(returns_alt, returns_bench, lam=float(lam))

                if logger is not None:
                    logger.info(
                        "Risk-RC/SPA universe: %s | %s | lam=%.2f | T=%d | models=%d",
                        sym, cost_name, float(lam), len(u_alt), u_alt.shape[1]
                    )

                rc = st_module.white_reality_check(
                    returns_alt=u_alt,
                    returns_bench=u_bench,
                    n_boot=int(reps),
                    avg_block_len=int(block_size),
                    seed=int(seed),
                )

                spa_res = hansen_spa(
                    u_alt,
                    u_bench,
                    block_size=int(block_size),
                    reps=int(reps),
                    seed=int(seed),
                    studentize=bool(studentize),
                    alpha=float(alpha),
                )

                row: Dict[str, Any] = {
                    "symbol": sym,
                    "cost": cost_name,
                    "lam": float(lam),
                    "T": int(len(u_alt)),
                    "universe_size": int(u_alt.shape[1]),
                    "rc_pvalue": float(rc.p_value),
                    "rc_stat": float(rc.statistic),
                    "rc_best_model": str(rc.best_model),
                }

                if spa_res is not None:
                    row.update(spa_res)
                else:
                    row.update(
                        {
                            "spa_pvalue_lower": np.nan,
                            "spa_pvalue_consistent": np.nan,
                            "spa_pvalue_upper": np.nan,
                            "spa_n_better_models": np.nan,
                        }
                    )

                rows.append(row)

    return pd.DataFrame(rows)
    
    
# =========================
# Variant B: SPA selection + comparison table
# =========================

def _series_from_long(
    df_long: pd.DataFrame, *, symbol: str, cost: str, strategy_id: str
) -> pd.Series:
    d = df_long[
        (df_long["symbol"] == symbol)
        & (df_long["cost"] == cost)
        & (df_long["strategy_id"] == strategy_id)
    ].copy()
    if d.empty:
        return pd.Series(dtype=float)

    # accept legacy column names too
    if "date" not in d.columns:
        if "datetime_utc" in d.columns:
            d = d.rename(columns={"datetime_utc": "date"})
        elif "index" in d.columns:
            d = d.rename(columns={"index": "date"})
        else:
            return pd.Series(dtype=float)

    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    return d.set_index("date")["ret"].astype(float)


def _metrics_row_from_returns(
    returns: pd.Series,
    *,
    compute_metrics_fn: Any,
) -> Dict[str, Any]:
    """
    compute_metrics_fn is usually src.metrics.compute_metrics_from_returns
    """
    m = compute_metrics_fn(returns)
    return {
        "TotalReturn": float(m.get("Total Return", np.nan)),
        "CAGR": float(m.get("CAGR", np.nan)),
        "VolAnn": float(m.get("Ann. Vol", np.nan)),
        "Sharpe": float(m.get("Sharpe", np.nan)),
        "Sortino": float(m.get("Sortino", np.nan)),
        "MaxDD": float(m.get("MaxDD", np.nan)),
        "Calmar": float(m.get("Calmar", np.nan)),
        "n_bars": int(m.get("n_bars", 0)),
    }


def spa_better_models_by_utility(
    returns_alt: pd.DataFrame,
    returns_bench: pd.Series,
    *,
    lam: float,
    block_size: int,
    reps: int,
    seed: int,
    studentize: bool,
    alpha: float = 0.05,
    pvalue_type: str = "consistent",
) -> List[str]:
    """
    Return list of strategy_ids (column names) that SPA considers superior under utility (lam).
    """
    from arch.bootstrap import SPA  # type: ignore

    df, bench = returns_alt.align(returns_bench, join="inner", axis=0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bench = bench.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # utility bench and models
    u_bench = utility_penalized_return(bench, lam=float(lam))
    u_models = pd.DataFrame({c: utility_penalized_return(df[c], lam=float(lam)) for c in df.columns}, index=df.index)

    # utility excess vs bench
    util_excess = u_models.sub(u_bench, axis=0)

    # SPA expects losses: lower is better; use benchmark=0 and model losses=-util_excess
    models = (-util_excess).to_numpy()
    benchmark = np.zeros(models.shape[0], dtype=float)

    T = models.shape[0]
    bs = int(block_size)
    if bs >= T:
        bs = max(2, T // 5)

    spa = SPA(benchmark, models, block_size=bs, reps=int(reps), seed=int(seed), studentize=bool(studentize))
    spa.compute()

    better_idx = spa.better_models(pvalue=float(alpha), pvalue_type=str(pvalue_type))
    better_idx = list(map(int, better_idx)) if better_idx is not None else []
    cols = list(df.columns)
    return [str(cols[i]) for i in better_idx if 0 <= i < len(cols)]


def run_spa_better_strategies_table(
    *,
    symbols: List[str],
    costs: List[str],
    returns_oos_long: pd.DataFrame,
    bench_long: pd.DataFrame,
    compute_metrics_fn: Any,  # usually compute_metrics_from_returns
    block_size: int,
    reps: int,
    seed: int,
    studentize: bool,
    lam_star: float = 1.0,
    alpha: float = 0.05,
    pvalue_type: str = "consistent",
    max_show: int = 20,
    logger: Any = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each (symbol, cost):
      1) build returns universe
      2) select SPA 'better models' under utility at lam_star
      3) build a comparison table with BENCH + selected strategies

    Returns:
      tbl (full table), tbl_view (BENCH + up to max_show per group)
    """
    rows: List[Dict[str, Any]] = []

    for sym in symbols:
        for cost_name in costs:
            returns_alt, returns_bench = build_universe_matrix(
                returns_oos_long, bench_long, symbol=sym, cost=cost_name
            )

            # if arch isn't installed, this will raise at import; let it bubble (or handle outside)
            better = spa_better_models_by_utility(
                returns_alt,
                returns_bench,
                lam=float(lam_star),
                block_size=int(block_size),
                reps=int(reps),
                seed=int(seed),
                studentize=bool(studentize),
                alpha=float(alpha),
                pvalue_type=str(pvalue_type),
            )

            if logger is not None:
                logger.info(
                    "SPA better models: %s | %s | lam=%.2f | alpha=%.3f | n=%d",
                    sym, cost_name, float(lam_star), float(alpha), len(better)
                )

            # --- Benchmark: use returns_bench from build_universe_matrix (already aligned to universe dates)
            bench_series = returns_bench.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            u_bench = utility_penalized_return(bench_series, lam=float(lam_star))
            bench_util_mean = float(u_bench.mean())

            bench_m = _metrics_row_from_returns(bench_series, compute_metrics_fn=compute_metrics_fn)

            rows.append({
                "symbol": sym,
                "cost": cost_name,
                "lam": float(lam_star),
                "strategy_id": "BENCH:BuyHold",
                "utility_mean": bench_util_mean,
                "utility_mean_excess": 0.0,
                **bench_m,
            })

            # strategy rows (only those SPA marked better)
            for sid in better:
                r = _series_from_long(returns_oos_long, symbol=sym, cost=cost_name, strategy_id=str(sid))
                if r.empty:
                    continue

                u = utility_penalized_return(r, lam=float(lam_star))
                # align utility with bench utility (same dates)
                u, u_bench_al = u.align(u_bench, join="inner")
                util_mean = float(u.mean())
                util_excess_mean = float((u - u_bench_al).mean())

                m = _metrics_row_from_returns(r, compute_metrics_fn=compute_metrics_fn)

                rows.append({
                    "symbol": sym,
                    "cost": cost_name,
                    "lam": float(lam_star),
                    "strategy_id": str(sid),
                    "utility_mean": util_mean,
                    "utility_mean_excess": util_excess_mean,
                    **m,
                })

    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return tbl, tbl

    # sort so BENCH is first, then strategies by utility excess desc
    tbl["is_bench"] = (tbl["strategy_id"] == "BENCH:BuyHold").astype(int)
    tbl = tbl.sort_values(
        ["symbol", "cost", "is_bench", "utility_mean_excess"],
        ascending=[True, True, False, False],
    ).drop(columns=["is_bench"])

    # view: BENCH + up to max_show strategies
    out_parts = []
    for (sym, cost_name), g in tbl.groupby(["symbol", "cost"], sort=True):
        bench_rows = g[g["strategy_id"] == "BENCH:BuyHold"]
        others = g[g["strategy_id"] != "BENCH:BuyHold"].head(int(max_show))
        out_parts.append(pd.concat([bench_rows, others], axis=0))
    tbl_view = pd.concat(out_parts, axis=0) if out_parts else tbl.head(0)

    return tbl, tbl_view
