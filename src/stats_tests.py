"""Statistical tests for data-snooping control.

The spec mentions White's Reality Check and Hansen's SPA (via arch.bootstrap.SPA).
In many environments `arch` might not be installed, so we:

1) Provide an adapter that uses `arch.bootstrap.SPA` if available.
2) Provide a standalone block-bootstrap implementation of White's Reality Check
   (RC) as a robust fallback.

Important:
- These tests require the *universe* of strategies that were considered (multiple
  comparisons). At minimum, you can apply RC to the set of final candidates
  (one stitched OOS return series per strategy family), which is what the runner
  produces by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UniverseSpec:
    """Definition of the model universe used for multiple-comparison control.

    In RC/SPA, the p-value is only interpretable relative to the *exact* set of
    models that were considered/selected from.
    """

    mode: str
    description: str
    models: Tuple[str, ...]


def build_universe_from_stitched_returns(
    *,
    returns_oos_long: pd.DataFrame,
    bench_oos_long: pd.DataFrame,
    symbol: str,
    cost: str,
    universe_mode: str = "strategies",
) -> Tuple[pd.DataFrame, pd.Series, UniverseSpec]:
    """Build (returns_alt, returns_bench) for RC/SPA from runner artifacts.

    Notes
    -----
    - The default universe_mode="strategies" treats each *walk-forward tuned*
      strategy family (one stitched OOS series per strategy_id) as one model.
      This matches what runner.py persists by default.
    - If you want a universe that also counts parameter search (strategy×params)
      as separate models, you must persist OOS return series for each parameter
      configuration that was in the selection set. That is a heavier artifact and
      is not produced in the MVP pipeline.
    """
    if universe_mode != "strategies":
        raise ValueError(
            f"Unsupported universe_mode={universe_mode!r}. Currently supported: 'strategies'. "
            "(To include strategy×params universes, persist per-parameter OOS returns.)"
        )

    df = returns_oos_long.copy()
    b = bench_oos_long.copy()

    for x in (df, b):
        if "date" not in x.columns:
            raise ValueError("Expected a 'date' column in long returns table")
        x["date"] = pd.to_datetime(x["date"])

    df = df[(df["symbol"] == symbol) & (df["cost"] == cost)]
    b = b[(b["symbol"] == symbol) & (b["cost"] == cost)]
    if df.empty:
        raise ValueError(f"No strategy returns found for symbol={symbol}, cost={cost}")
    if b.empty:
        raise ValueError(f"No benchmark returns found for symbol={symbol}, cost={cost}")

    # One column per strategy_id
    returns_alt = (
        df.pivot_table(index="date", columns="strategy_id", values="ret", aggfunc="mean")
        .sort_index()
        .astype(float)
    )
    # Benchmark: runner uses one benchmark id, but we aggregate defensively.
    returns_bench = b.groupby("date")["ret"].mean().sort_index().astype(float)

    models = tuple(map(str, returns_alt.columns.tolist()))
    spec = UniverseSpec(
        mode="strategies",
        description="WF-tuned strategies: one stitched OOS return series per strategy_id (runner output)",
        models=models,
    )
    return returns_alt, returns_bench, spec


def _stationary_bootstrap_indices(n: int, *, rng: np.random.Generator, avg_block_len: int) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices.

    Each step:
    - with prob p = 1/avg_block_len start a new block at a random position
    - otherwise continue sequentially (wrapping around)

    Returns an array of indices of length n.
    """
    if n <= 0:
        return np.array([], dtype=int)
    p = 1.0 / max(1, int(avg_block_len))
    idx = np.empty(n, dtype=int)
    idx[0] = int(rng.integers(0, n))
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = int(rng.integers(0, n))
        else:
            idx[t] = (idx[t - 1] + 1) % n
    return idx


@dataclass(frozen=True)
class RealityCheckResult:
    p_value: float
    statistic: float
    best_model: str
    universe_size: int
    mean_diffs: pd.Series


def white_reality_check(
    *,
    returns_alt: pd.DataFrame,
    returns_bench: pd.Series,
    n_boot: int = 2000,
    avg_block_len: int = 10,
    seed: int = 42,
) -> RealityCheckResult:
    """White's Reality Check (RC) with stationary block bootstrap.

    IMPORTANT
    ---------
    The interpretation of the p-value depends on the *universe* (set) of models
    included in `returns_alt`. This universe should match the set you actually
    selected from (e.g. all candidates in a model-selection procedure).

    Parameters
    ----------
    returns_alt : DataFrame
        Columns are alternative models/strategies. Values are daily returns.
    returns_bench : Series
        Benchmark daily returns.
    n_boot : int
        Bootstrap repetitions.
    avg_block_len : int
        Average block length (in bars) for stationary bootstrap.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    RealityCheckResult
    """
    if returns_alt.empty:
        raise ValueError("returns_alt is empty")
    # Align
    df = returns_alt.copy()
    bench = returns_bench.copy()
    df, bench = df.align(bench, join="inner", axis=0)
    df = df.dropna(how="all")
    bench = bench.loc[df.index].fillna(0.0)
    df = df.fillna(0.0)

    n = len(df)
    if n < 50:
        # With too few points, bootstrap p-values are not very meaningful; still compute.
        pass

    d = df.sub(bench, axis=0)  # outperformance series d_{t,j}
    mean_d = d.mean(axis=0)
    best = str(mean_d.idxmax())
    stat_obs = float(mean_d.max())

    # RC bootstrap: center d by subtracting sample means
    d_centered = d - mean_d

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot, dtype=float)

    d_vals = d_centered.to_numpy()
    for b in range(n_boot):
        idx = _stationary_bootstrap_indices(n, rng=rng, avg_block_len=avg_block_len)
        sample = d_vals[idx, :]
        boot_stats[b] = float(np.mean(sample, axis=0).max())
    count = int(np.sum(boot_stats >= stat_obs))
    p_val = (count + 1.0) / (n_boot + 1.0)
    return RealityCheckResult(
        p_value=p_val,
        statistic=stat_obs,
        best_model=best,
        universe_size=df.shape[1],
        mean_diffs=mean_d.sort_values(ascending=False),
    )


def hansen_spa_via_arch(
    *,
    returns_alt: pd.DataFrame,
    returns_bench: pd.Series,
    avg_block_len: int = 10,
    seed: int = 42,
    studentize: bool = True,
) -> Optional[Dict[str, float]]:
    """Hansen SPA / Reality Check via arch.bootstrap.SPA if `arch` is installed.

    Returns dict with key stats (p-value, etc.) or None if `arch` isn't available.
    """
    try:
        from arch.bootstrap import SPA  # type: ignore
    except Exception:
        return None

    df = returns_alt.copy()
    bench = returns_bench.copy()
    df, bench = df.align(bench, join="inner", axis=0)
    df = df.fillna(0.0)
    bench = bench.fillna(0.0)

    # arch expects losses; for returns, losses = -excess_returns is typical.
    # Excess relative to benchmark:
    excess = df.sub(bench, axis=0)
    losses = -excess

    spa = SPA(losses.to_numpy(), block_size=avg_block_len, reps=2000, seed=seed, studentize=studentize)
    res = spa.compute()
    # res is a dict-like with keys. We expose the most common ones.
    out: Dict[str, float] = {}
    for k in ["pvalue", "stat", "lower", "upper"]:
        if k in res:
            out[k] = float(res[k])
    return out
