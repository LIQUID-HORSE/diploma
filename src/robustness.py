# src/robustness.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import safe_write_parquet
from .experiment_config import ExperimentConfig
from .metrics import compute_metrics_from_returns


# -----------------------
# Helpers
# -----------------------

def ensure_date_col(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the long-returns dataframe has a 'date' column (supports legacy names).
    Mutates a copy only if rename is needed; always converts to datetime.
    """
    df = df_in.copy()
    if "date" not in df.columns:
        if "datetime_utc" in df.columns:
            df = df.rename(columns={"datetime_utc": "date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "date"})
        else:
            raise ValueError(f"Can't find date column. Columns: {list(df.columns)[:30]}")
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_stitched_series(
    df_long: pd.DataFrame,
    *,
    symbol: str,
    cost: str,
    strategy_id: str,
) -> pd.Series:
    """
    Extract stitched return series for (symbol, cost, strategy_id) from a long dataframe.
    Expects df_long to already have a 'date' datetime column.
    """
    g = df_long[
        (df_long["symbol"] == symbol)
        & (df_long["cost"] == cost)
        & (df_long["strategy_id"] == strategy_id)
    ].sort_values("date")

    if g.empty:
        return pd.Series(dtype=float)

    return g.set_index("date")["ret"].astype(float)


def stitched_metrics_for_subset_costs(
    *,
    symbol: str,
    strategies: list[str],
    costs: list[str],
    returns_oos_long: pd.DataFrame,
    bench_long: pd.DataFrame,
    bench_id: str = "BENCH:BuyHold",
    min_bars: int = 50,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for cost in costs:
        bench = get_stitched_series(bench_long, symbol=symbol, cost=cost, strategy_id=bench_id)
        if len(bench) >= min_bars:
            rows.append(
                {"symbol": symbol, "cost": cost, "strategy_id": bench_id, **compute_metrics_from_returns(bench)}
            )

        for sid in strategies:
            r = get_stitched_series(returns_oos_long, symbol=symbol, cost=cost, strategy_id=sid)
            if len(r) < min_bars:
                continue
            rows.append(
                {"symbol": symbol, "cost": cost, "strategy_id": sid, **compute_metrics_from_returns(r)}
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["symbol", "strategy_id", "cost"])
    return out


# -----------------------
# Utility: penalty on worsening drawdown only
# -----------------------

def utility_penalized_return_worsen_dd(returns: pd.Series, lam: float) -> pd.Series:
    """
    u_t = r_t - lam * max(0, DDdepth_t - DDdepth_{t-1})
    where DDdepth_t = 1 - equity_t/peak_t >= 0
    """
    r = returns.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd_depth = 1.0 - (eq / peak)

    dd_worsen = dd_depth.diff().fillna(0.0).clip(lower=0.0)
    return r - float(lam) * dd_worsen


def mean_utility_excess_for_strategy(
    *,
    symbol: str,
    cost: str,
    strategy_id: str,
    lam: float,
    returns_oos_long: pd.DataFrame,
    bench_long: pd.DataFrame,
    bench_id: str = "BENCH:BuyHold",
) -> float:
    r = get_stitched_series(returns_oos_long, symbol=symbol, cost=cost, strategy_id=strategy_id)
    b = get_stitched_series(bench_long, symbol=symbol, cost=cost, strategy_id=bench_id)
    if r.empty or b.empty:
        return float("nan")

    u = utility_penalized_return_worsen_dd(r, lam)
    ub = utility_penalized_return_worsen_dd(b, lam)

    u, ub = u.align(ub, join="inner")
    if len(u) == 0:
        return float("nan")

    return float((u - ub).mean())


# -----------------------
# Robustness 1/4: cost sensitivity (top by Calmar on Base)
# -----------------------

def cost_sensitivity_top_calmar(
    *,
    cfg: ExperimentConfig,
    stitched: pd.DataFrame,
    returns_oos_all: pd.DataFrame,
    bench_oos_all: pd.DataFrame,
    symbols: list[str],
    topN: int = 5,
    costs_all: list[str] | None = None,
    bench_id: str = "BENCH:BuyHold",
    min_bars: int = 50,
    out_name: str = "robust_cost_sensitivity_top_calmar.parquet",
) -> pd.DataFrame:
    costs_all = costs_all or ["Low", "Base", "High"]

    returns_oos_all = ensure_date_col(returns_oos_all)
    bench_oos_all = ensure_date_col(bench_oos_all)

    sens_tables: list[pd.DataFrame] = []
    for sym in symbols:
        stitched_sym_base = (
            stitched[(stitched["symbol"] == sym) & (stitched["cost"] == "Base")]
            .sort_values("Calmar", ascending=False)
        )
        top_strats = stitched_sym_base.head(int(topN))["strategy_id"].astype(str).tolist()
        top_strats = [sid for sid in top_strats if not sid.startswith("BENCH:")]

        t = stitched_metrics_for_subset_costs(
            symbol=sym,
            strategies=top_strats,
            costs=costs_all,
            returns_oos_long=returns_oos_all,
            bench_long=bench_oos_all,
            bench_id=bench_id,
            min_bars=min_bars,
        )
        t["selection_rule"] = f"Top{topN}_by_Calmar_on_Base"
        sens_tables.append(t)

    out = pd.concat(sens_tables, ignore_index=True) if sens_tables else pd.DataFrame()
    safe_write_parquet(out, cfg.results_dir / out_name)
    return out


# -----------------------
# Robustness 2/4: cost sensitivity (top by UtilityExcess on Base)
# -----------------------

def cost_sensitivity_top_utility(
    *,
    cfg: ExperimentConfig,
    stitched: pd.DataFrame,
    returns_oos_all: pd.DataFrame,
    bench_oos_all: pd.DataFrame,
    symbols: list[str],
    lam_star: float = 1.0,
    topN: int = 5,
    costs_all: list[str] | None = None,
    bench_id: str = "BENCH:BuyHold",
    min_bars: int = 50,
    out_name: str = "robust_cost_sensitivity_top_utility.parquet",
) -> pd.DataFrame:
    costs_all = costs_all or ["Low", "Base", "High"]

    returns_oos_all = ensure_date_col(returns_oos_all)
    bench_oos_all = ensure_date_col(bench_oos_all)

    sens_tables: list[pd.DataFrame] = []
    for sym in symbols:
        base_strats = (
            stitched[(stitched["symbol"] == sym) & (stitched["cost"] == "Base")]["strategy_id"]
            .astype(str)
            .tolist()
        )
        base_strats = [sid for sid in base_strats if not sid.startswith("BENCH:")]

        scores: list[tuple[str, float]] = []
        for sid in base_strats:
            s = mean_utility_excess_for_strategy(
                symbol=sym,
                cost="Base",
                strategy_id=sid,
                lam=float(lam_star),
                returns_oos_long=returns_oos_all,
                bench_long=bench_oos_all,
                bench_id=bench_id,
            )
            if np.isfinite(s):
                scores.append((sid, float(s)))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_strats = [sid for sid, _ in scores[: int(topN)]]

        t = stitched_metrics_for_subset_costs(
            symbol=sym,
            strategies=top_strats,
            costs=costs_all,
            returns_oos_long=returns_oos_all,
            bench_long=bench_oos_all,
            bench_id=bench_id,
            min_bars=min_bars,
        )
        t["selection_rule"] = f"Top{topN}_by_UtilityExcessWorsenDD_lam{lam_star}_on_Base"
        t["lam"] = float(lam_star)
        sens_tables.append(t)

    out = pd.concat(sens_tables, ignore_index=True) if sens_tables else pd.DataFrame()
    safe_write_parquet(out, cfg.results_dir / out_name)
    return out


# -----------------------
# Robustness 3/4: subperiods (top by Calmar on Base)
# -----------------------

def metrics_on_slice(
    r: pd.Series,
    *,
    start: str,
    end: str,
    min_bars: int = 50,
) -> dict[str, Any] | None:
    rs = r.loc[(r.index >= pd.Timestamp(start)) & (r.index < pd.Timestamp(end))]
    if len(rs) < min_bars:
        return None
    return compute_metrics_from_returns(rs)


def subperiods_top_calmar(
    *,
    cfg: ExperimentConfig,
    stitched: pd.DataFrame,
    returns_oos_all: pd.DataFrame,
    bench_oos_all: pd.DataFrame,
    symbols: list[str],
    subperiods: dict[str, tuple[str, str]],
    topK: int = 3,
    bench_id: str = "BENCH:BuyHold",
    min_bars: int = 50,
    out_name: str = "robust_subperiods_top_calmar.parquet",
) -> pd.DataFrame:
    returns_oos_all = ensure_date_col(returns_oos_all)
    bench_oos_all = ensure_date_col(bench_oos_all)

    rows: list[dict[str, Any]] = []

    for sym in symbols:
        top = (
            stitched[(stitched["symbol"] == sym) & (stitched["cost"] == "Base")]
            .sort_values("Calmar", ascending=False)
            .head(int(topK))
        )
        top_strats = top["strategy_id"].astype(str).tolist()
        top_strats = [sid for sid in top_strats if not sid.startswith("BENCH:")]

        series_map: dict[str, pd.Series] = {
            bench_id: get_stitched_series(bench_oos_all, symbol=sym, cost="Base", strategy_id=bench_id)
        }
        for sid in top_strats:
            series_map[sid] = get_stitched_series(returns_oos_all, symbol=sym, cost="Base", strategy_id=sid)

        for sid, r in series_map.items():
            if r.empty:
                continue
            for label, (start, end) in subperiods.items():
                m = metrics_on_slice(r, start=start, end=end, min_bars=min_bars)
                if m is None:
                    continue
                rows.append({"symbol": sym, "cost": "Base", "strategy_id": sid, "subperiod": label, **m})

    out = pd.DataFrame(rows)
    safe_write_parquet(out, cfg.results_dir / out_name)
    return out


# -----------------------
# Robustness 4/4: subperiods (top by UtilityExcess on Base)
# -----------------------

def subperiods_top_utility(
    *,
    cfg: ExperimentConfig,
    stitched: pd.DataFrame,
    returns_oos_all: pd.DataFrame,
    bench_oos_all: pd.DataFrame,
    symbols: list[str],
    subperiods: dict[str, tuple[str, str]],
    lam_star: float = 1.0,
    topK: int = 3,
    bench_id: str = "BENCH:BuyHold",
    min_bars: int = 50,
    out_name: str = "robust_subperiods_top_utility.parquet",
) -> pd.DataFrame:
    returns_oos_all = ensure_date_col(returns_oos_all)
    bench_oos_all = ensure_date_col(bench_oos_all)

    rows: list[dict[str, Any]] = []

    for sym in symbols:
        base_strats = (
            stitched[(stitched["symbol"] == sym) & (stitched["cost"] == "Base")]["strategy_id"]
            .astype(str)
            .tolist()
        )
        base_strats = [sid for sid in base_strats if not sid.startswith("BENCH:")]

        scores: list[tuple[str, float]] = []
        for sid in base_strats:
            s = mean_utility_excess_for_strategy(
                symbol=sym,
                cost="Base",
                strategy_id=sid,
                lam=float(lam_star),
                returns_oos_long=returns_oos_all,
                bench_long=bench_oos_all,
                bench_id=bench_id,
            )
            if np.isfinite(s):
                scores.append((sid, float(s)))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_strats = [sid for sid, _ in scores[: int(topK)]]

        series_map: dict[str, pd.Series] = {
            bench_id: get_stitched_series(bench_oos_all, symbol=sym, cost="Base", strategy_id=bench_id)
        }
        for sid in top_strats:
            series_map[sid] = get_stitched_series(returns_oos_all, symbol=sym, cost="Base", strategy_id=sid)

        for sid, r in series_map.items():
            if r.empty:
                continue
            for label, (start, end) in subperiods.items():
                m = metrics_on_slice(r, start=start, end=end, min_bars=min_bars)
                if m is None:
                    continue
                rows.append(
                    {
                        "symbol": sym,
                        "cost": "Base",
                        "strategy_id": sid,
                        "subperiod": label,
                        "lam": float(lam_star),
                        **m,
                    }
                )

    out = pd.DataFrame(rows)
    safe_write_parquet(out, cfg.results_dir / out_name)
    return out