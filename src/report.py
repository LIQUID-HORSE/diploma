"""
Reporting: tables and plots for the course write-up.

Reads artifacts created by runner/notebook:
- results/folds_best.parquet
- results/returns_oos.parquet
- results/bench_returns_oos.parquet
- (optional) results/train_heatmaps.parquet

Generates:
- leaderboards (aggregated fold metrics)
- stitched OOS equity vs buy-and-hold
- stitched OOS drawdown
- selected heatmaps
- sensitivity comparisons across cost scenarios
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import compute_metrics_from_returns, equity_from_returns, drawdown_from_equity

# For strategies with >2 params, explicitly choose heatmap axes (x,y).
# This prevents confusing plots when there are many parameters.
HEATMAP_AXES: Dict[str, Tuple[str, str]] = {
    "T1:SMA_Crossover": ("fast", "slow"),
    "T2:EMA_Crossover": ("fast", "slow"),
    "T3:Donchian_Breakout": ("N", "exit"),
    "R1:RSI_MeanReversion": ("low", "exit"),
    "R2:Bollinger_MeanReversion": ("n", "k"),
    "R3:ZScore_MeanReversion": ("entry", "exit"),
    "S1:MAFilter_RSI": ("low", "exit"),
    "S2:MA200Filter_Bollinger": ("n", "k"),
    "S3:BreakoutConfirm_MA": ("N", "M"),
    "S4:MACross_TSMOMConfirm": ("fast", "slow"),
    "S5:Simple_Ensemble": ("N", "L"),
}


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize date column name used across artifacts.

    Some artifacts store timestamp as `datetime_utc`, others as `date`.
    Reporting expects `date`.
    """
    out = df.copy()
    if "date" not in out.columns and "datetime_utc" in out.columns:
        out = out.rename(columns={"datetime_utc": "date"})
    return out


def safe_read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        # fallback
        return pd.read_csv(path)


def load_results(out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    results_dir = out_dir / "results"
    folds = safe_read_parquet(results_dir / "folds_best.parquet")
    rets = safe_read_parquet(results_dir / "returns_oos.parquet")
    bench = safe_read_parquet(results_dir / "bench_returns_oos.parquet")

    heatmaps_path = results_dir / "train_heatmaps.parquet"
    heatmaps = safe_read_parquet(heatmaps_path) if heatmaps_path.exists() else None
    return folds, rets, bench, heatmaps


def leaderboard_by_strategy(folds_best: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level OOS metrics into per-strategy leaderboard."""
    df = folds_best.copy()
    keys = ["symbol", "cost", "strategy_id"]
    num_cols = [
        "oos_total_return",
        "oos_cagr",
        "oos_sharpe",
        "oos_sortino",
        "oos_maxdd",
        "oos_calmar",
        "oos_trades",
        "oos_exposure_pct",
    ]
    existing = [c for c in num_cols if c in df.columns]

    agg = {c: ["median", "mean", "min", "max"] for c in existing}
    agg["fold_id"] = ["nunique"]

    out = (
        df.groupby(keys, dropna=False)
        .agg(agg)
        .reset_index()
    )

    # flatten columns
    out.columns = [
        f"{a}_{b}" if b else a
        for (a, b) in out.columns.to_flat_index()
    ]

    # rename fold count
    if "fold_id_nunique" in out.columns:
        out = out.rename(columns={"fold_id_nunique": "folds"})

    # sort (best first)
    sort_key = "oos_calmar_median" if "oos_calmar_median" in out.columns else out.columns[-1]
    out = out.sort_values(["symbol", "cost", sort_key], ascending=[True, True, False])
    return out


def stitched_metrics(returns_oos: pd.DataFrame, *, periods_per_year: int = 365) -> pd.DataFrame:
    """Compute metrics for stitched OOS return series for each strategy."""
    df = _ensure_date_column(returns_oos)
    if "date" not in df.columns:
        raise KeyError("returns_oos must contain a 'date' column (or 'datetime_utc' that can be renamed to 'date').")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "cost", "strategy_id", "date"])

    rows = []
    for (sym, cost, sid), g in df.groupby(["symbol", "cost", "strategy_id"], dropna=False):
        r = g.set_index("date")["ret"].astype(float)
        m = compute_metrics_from_returns(r, periods_per_year=periods_per_year)
        rows.append({"symbol": sym, "cost": cost, "strategy_id": sid, **m})
    out = pd.DataFrame(rows)
    if not out.empty and "Calmar" in out.columns:
        out = out.sort_values(["symbol", "cost", "Calmar"], ascending=[True, True, False])
    return out


def plot_equity_and_drawdown(
    *,
    returns: pd.Series,
    bench_returns: pd.Series,
    title: str,
    out_path: Path,
) -> None:
    """Plot equity and drawdown for strategy vs benchmark."""
    import matplotlib.pyplot as plt

    eq = equity_from_returns(returns)
    beq = equity_from_returns(bench_returns)

    dd = drawdown_from_equity(eq)
    bdd = drawdown_from_equity(beq)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(eq.index, eq.values, label="Strategy")
    ax1.plot(beq.index, beq.values, label="Buy&Hold")
    ax1.set_title(title)
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(dd.index, dd.values, label="Strategy DD")
    ax2.plot(bdd.index, bdd.values, label="Buy&Hold DD")
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_heatmap(
    heatmaps: pd.DataFrame,
    *,
    symbol: str,
    cost: str,
    fold_id: int,
    strategy_id: str,
    out_path: Path,
    axes: Optional[Tuple[str, str]] = None,
) -> None:
    """Plot a 2D heatmap from a saved train heatmap table.

    Works best for 2-parameter grids (e.g. SMA crossover fast/slow).
    If the filtered surface degenerates to 1xN / Nx1 / 1x1 we skip plotting
    to avoid matplotlib 'identical x/y limits' warnings and blank figures.
    """
    import matplotlib.pyplot as plt

    df = heatmaps.copy()
    df = df[
        (df["symbol"] == symbol)
        & (df["cost"] == cost)
        & (df["fold_id"] == fold_id)
        & (df["strategy_id"] == strategy_id)
    ]
    if df.empty:
        return

    # meta cols
    meta_cols = {"symbol", "cost", "fold_id", "strategy_id", "objective"}

    # параметр-колонки (убираем те, которые полностью NaN)
    param_cols_all = [c for c in df.columns if c not in meta_cols]
    param_cols = [c for c in param_cols_all if not df[c].isna().all()]

    if len(param_cols) < 2:
        return  # не из чего строить 2D поверхность

    # оси
    if axes is None:
        axes = HEATMAP_AXES.get(strategy_id)

    if axes is None:
        # выбрать две колонки с наибольшим числом уникальных значений (игнорируя NaN)
        uniq = sorted(((c, df[c].nunique(dropna=True)) for c in param_cols), key=lambda t: t[1], reverse=True)
        axes = (uniq[0][0], uniq[1][0])

    x, y = axes
    # если в HEATMAP_AXES указаны оси, но в df они оказались полностью NaN — подстрахуемся
    if (x not in df.columns) or df[x].isna().all():
        x = param_cols[0]
    if (y not in df.columns) or df[y].isna().all():
        y = param_cols[1]

    # фиксируем остальные параметры (кроме x,y) на best значениях
    other = [c for c in param_cols if c not in (x, y)]
    if other:
        try:
            best_row = df.loc[df["objective"].astype(float).idxmax()]
            for c in other:
                # если best_row[c] NaN, то фиксировать бессмысленно
                if pd.isna(best_row[c]):
                    continue
                df = df[df[c] == best_row[c]]
        except Exception:
            pass

    pivot = df.pivot_table(index=y, columns=x, values="objective", aggfunc="mean")

    # если после фильтрации выродилось — просто не рисуем
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    ax.set_title(f"Heatmap {strategy_id} | {symbol} | {cost} | fold {fold_id}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Objective (Calmar+penalties)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def generate_all_reports(out_dir: Path, figures_dir: Optional[Path] = None, *, periods_per_year: int = 365) -> None:
    """Convenience wrapper: build leaderboards and common figures."""
    folds, rets, bench, heatmaps = load_results(out_dir)
    figures_dir = figures_dir or (out_dir / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Normalize date columns for report module
    rets = _ensure_date_column(rets)
    bench = _ensure_date_column(bench)

    # Save CSV fallbacks
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    lb = leaderboard_by_strategy(folds)
    try:
        lb.to_parquet(results_dir / "leaderboard.parquet", index=False)
    except Exception:
        lb.to_csv(results_dir / "leaderboard.csv", index=False)

    stitched = stitched_metrics(rets, periods_per_year=periods_per_year)
    try:
        stitched.to_parquet(results_dir / "stitched_metrics.parquet", index=False)
    except Exception:
        stitched.to_csv(results_dir / "stitched_metrics.csv", index=False)

    # Plot top-3 per (symbol,cost)
    if stitched.empty:
        return

    stitched = stitched.sort_values(["symbol", "cost", "Calmar"], ascending=[True, True, False])
    for (sym, cost), g in stitched.groupby(["symbol", "cost"]):
        top = g.head(3)
        for _, row in top.iterrows():
            sid = str(row["strategy_id"])
            r = rets[(rets["symbol"] == sym) & (rets["cost"] == cost) & (rets["strategy_id"] == sid)].copy()
            b = bench[(bench["symbol"] == sym) & (bench["cost"] == cost)].copy()
            if r.empty or b.empty:
                continue

            r["date"] = pd.to_datetime(r["date"])
            b["date"] = pd.to_datetime(b["date"])
            rs = r.set_index("date")["ret"].astype(float)
            bs = b.set_index("date")["ret"].astype(float)

            plot_equity_and_drawdown(
                returns=rs,
                bench_returns=bs,
                title=f"{sid} vs Buy&Hold | {sym} | {cost}",
                out_path=figures_dir / f"equity_dd__{sym}__{cost}__{sid.replace(':','_')}.png",
            )

    # One heatmap example if available (choose a non-degenerate surface)
    if heatmaps is not None and not heatmaps.empty:
        chosen = None

        meta_cols = {"symbol", "cost", "fold_id", "strategy_id", "objective"}
        for i in range(min(len(heatmaps), 2000)):
            first = heatmaps.iloc[i]
            df = heatmaps[
                (heatmaps["symbol"] == first["symbol"])
                & (heatmaps["cost"] == first["cost"])
                & (heatmaps["fold_id"] == first["fold_id"])
                & (heatmaps["strategy_id"] == first["strategy_id"])
            ]
            param_cols = [c for c in df.columns if c not in meta_cols]
            if len(param_cols) < 2 or df.empty:
                continue

            axes = HEATMAP_AXES.get(str(first["strategy_id"]))
            if axes is None:
                uniq = sorted(((c, df[c].nunique(dropna=False)) for c in param_cols), key=lambda t: t[1], reverse=True)
                axes = (uniq[0][0], uniq[1][0])
            x, y = axes
            if x not in df.columns or y not in df.columns:
                x, y = param_cols[0], param_cols[1]

            other = [c for c in param_cols if c not in (x, y)]
            if other:
                try:
                    best_row = df.loc[df["objective"].astype(float).idxmax()]
                    for c in other:
                        df = df[df[c] == best_row[c]]
                except Exception:
                    pass

            pivot = df.pivot_table(index=y, columns=x, values="objective", aggfunc="mean")
            if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
                chosen = first
                break

        if chosen is not None:
            plot_heatmap(
                heatmaps,
                symbol=str(chosen["symbol"]),
                cost=str(chosen["cost"]),
                fold_id=int(chosen["fold_id"]),
                strategy_id=str(chosen["strategy_id"]),
                out_path=figures_dir / "heatmap_example.png",
            )
