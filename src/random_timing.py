from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Sequence

import numpy as np
import pandas as pd

GROUP_KEY = ["timeframe", "scenario", "symbol", "cost", "strategy_id"]
SERIES_KEY = ["date", "timeframe", "scenario", "symbol", "cost", "strategy_id"]


@dataclass(frozen=True)
class PositionStats:
    exposure: float
    n_trades: float
    avg_trade_len: float
    switches: float


@dataclass(frozen=True)
class MarkovParams:
    p_enter: float
    p_exit: float
    exposure_check: float


def ensure_date_col(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    if "date" not in df.columns:
        if "datetime_utc" in df.columns:
            df = df.rename(columns={"datetime_utc": "date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "date"})
        else:
            raise KeyError(f"No date-like column in dataframe. Columns: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def normalize_returns_long(
    df_in: pd.DataFrame,
    *,
    timeframe: str,
    scenario: str,
) -> pd.DataFrame:
    df = ensure_date_col(df_in)
    need_cols = {"date", "ret", "symbol", "cost", "strategy_id"}
    missing = sorted(need_cols - set(df.columns))
    if missing:
        raise KeyError(f"Missing required returns columns: {missing}")

    out = df[["date", "ret", "symbol", "cost", "strategy_id"]].copy()
    out["ret"] = pd.to_numeric(out["ret"], errors="coerce").fillna(0.0).astype(float)
    out["timeframe"] = str(timeframe)
    out["scenario"] = str(scenario)
    out = out[SERIES_KEY + ["ret"]].sort_values(SERIES_KEY).reset_index(drop=True)
    return out


def normalize_positions_long(df_in: pd.DataFrame) -> pd.DataFrame:
    df = ensure_date_col(df_in)
    need_cols = {"date", "position", "timeframe", "scenario", "symbol", "cost", "strategy_id"}
    missing = sorted(need_cols - set(df.columns))
    if missing:
        raise KeyError(f"Missing required positions columns: {missing}")

    out = df[list(need_cols)].copy()
    out["position"] = pd.to_numeric(out["position"], errors="coerce").fillna(0.0)
    out["position"] = (out["position"].abs() > 0.5).astype(np.int8)
    out = out.sort_values(SERIES_KEY).reset_index(drop=True)
    return out


def assert_unique_by_key(df: pd.DataFrame, *, key: list[str], name: str) -> None:
    dups = int(df.duplicated(subset=key).sum())
    if dups > 0:
        raise RuntimeError(f"{name}: duplicate rows by key={key}: {dups}")


def compute_utility_series(ret: np.ndarray | pd.Series, lam: float) -> np.ndarray:
    r = np.asarray(ret, dtype=float)
    if r.ndim != 1:
        raise ValueError("compute_utility_series expects 1D returns array.")
    if r.size == 0:
        return np.empty(0, dtype=float)

    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = (peak / np.maximum(equity, 1e-16)) - 1.0
    delta_dd = np.diff(dd, prepend=0.0)
    penalty = np.maximum(0.0, delta_dd)
    return r - float(lam) * penalty


def _utility_mean_from_returns_matrix(ret_mx: np.ndarray, lam: float) -> np.ndarray:
    if ret_mx.ndim != 2:
        raise ValueError("ret_mx must be 2D with shape [batch, T].")
    if ret_mx.size == 0:
        return np.empty(ret_mx.shape[0], dtype=float)

    equity = np.cumprod(1.0 + ret_mx, axis=1)
    peak = np.maximum.accumulate(equity, axis=1)
    dd = (peak / np.maximum(equity, 1e-16)) - 1.0
    delta_dd = np.diff(dd, axis=1, prepend=np.zeros((dd.shape[0], 1), dtype=float))
    penalty = np.maximum(0.0, delta_dd)
    utility = ret_mx - float(lam) * penalty
    return utility.mean(axis=1)


def compute_position_stats(position: np.ndarray | pd.Series) -> PositionStats:
    pos = np.asarray(position, dtype=np.int8)
    if pos.ndim != 1:
        raise ValueError("position must be a 1D array.")
    if pos.size == 0:
        return PositionStats(exposure=float("nan"), n_trades=float("nan"), avg_trade_len=float("nan"), switches=0.0)

    pos = np.where(pos > 0, 1, 0).astype(np.int8)
    switches = float(np.abs(np.diff(pos, prepend=0)).sum())
    n_trades = switches / 2.0
    exposure = float(pos.mean())
    avg_trade_len = float(pos.sum() / n_trades) if n_trades > 0 else 0.0
    return PositionStats(
        exposure=exposure,
        n_trades=float(n_trades),
        avg_trade_len=avg_trade_len,
        switches=switches,
    )


def calibrate_markov_params(*, exposure: float, n_trades: float, n_bars: int) -> MarkovParams:
    if n_bars <= 0:
        raise ValueError("n_bars must be positive.")

    denom_exit = max(1e-16, float(exposure) * float(n_bars))
    denom_enter = max(1e-16, float(1.0 - exposure) * float(n_bars))
    p_exit = float(np.clip(float(n_trades) / denom_exit, 0.0, 1.0))
    p_enter = float(np.clip(float(n_trades) / denom_enter, 0.0, 1.0))
    denom = p_enter + p_exit
    e_check = float(p_enter / denom) if denom > 0 else float(exposure)
    return MarkovParams(p_enter=p_enter, p_exit=p_exit, exposure_check=e_check)


def deterministic_seed(base_seed: int, parts: Sequence[Any]) -> int:
    key = "|".join(str(x) for x in parts)
    digest = sha256(f"{base_seed}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**63 - 1)


def _generate_markov_positions_batch(
    *,
    rng: np.random.Generator,
    batch_size: int,
    n_bars: int,
    exposure: float,
    p_enter: float,
    p_exit: float,
) -> np.ndarray:
    pos = np.empty((batch_size, n_bars), dtype=np.int8)
    pos[:, 0] = (rng.random(batch_size) < float(exposure)).astype(np.int8)

    for t in range(1, n_bars):
        u = rng.random(batch_size)
        prev = pos[:, t - 1]
        enter = (u < float(p_enter)).astype(np.int8)
        stay_long = (u >= float(p_exit)).astype(np.int8)
        pos[:, t] = np.where(prev == 0, enter, stay_long)
    return pos


def simulate_random_excess_distribution(
    *,
    ret_bench: np.ndarray,
    exposure: float,
    p_enter: float,
    p_exit: float,
    entry_cost: float,
    exit_cost: float,
    lam: float,
    k: int,
    chunk_size: int,
    seed: int,
) -> np.ndarray:
    r_b = np.asarray(ret_bench, dtype=float)
    if r_b.ndim != 1:
        raise ValueError("ret_bench must be 1D.")
    if r_b.size == 0:
        return np.empty(0, dtype=float)
    if k <= 0:
        return np.empty(0, dtype=float)

    n_bars = int(r_b.size)
    u_bench_mean = float(compute_utility_series(r_b, lam=float(lam)).mean())

    rng = np.random.default_rng(int(seed))
    out = np.empty(int(k), dtype=float)
    ptr = 0
    chunk = max(1, int(chunk_size))

    while ptr < int(k):
        bs = min(chunk, int(k) - ptr)
        pos = _generate_markov_positions_batch(
            rng=rng,
            batch_size=bs,
            n_bars=n_bars,
            exposure=float(exposure),
            p_enter=float(p_enter),
            p_exit=float(p_exit),
        )

        prev = np.concatenate([np.zeros((bs, 1), dtype=np.int8), pos[:, :-1]], axis=1)
        enters = ((prev == 0) & (pos == 1)).astype(float)
        exits = ((prev == 1) & (pos == 0)).astype(float)
        r_rand = (
            (pos.astype(float) * r_b.reshape(1, -1))
            - (enters * float(entry_cost))
            - (exits * float(exit_cost))
        )

        u_rand_mean = _utility_mean_from_returns_matrix(r_rand, lam=float(lam))
        out[ptr : ptr + bs] = u_rand_mean - u_bench_mean
        ptr += bs

    return out


def compute_random_timing_result(
    *,
    position: np.ndarray,
    ret_strategy: np.ndarray,
    ret_bench: np.ndarray,
    lam: float,
    k: int,
    chunk_size: int,
    entry_cost: float,
    exit_cost: float,
    seed: int,
    eps: float = 1e-12,
) -> dict[str, Any]:
    pos = np.asarray(position, dtype=np.int8)
    rs = np.asarray(ret_strategy, dtype=float)
    rb = np.asarray(ret_bench, dtype=float)

    if pos.ndim != 1 or rs.ndim != 1 or rb.ndim != 1:
        raise ValueError("position/ret_strategy/ret_bench must be 1D arrays.")
    if not (len(pos) == len(rs) == len(rb)):
        raise ValueError("position, ret_strategy and ret_bench must have equal lengths after alignment.")
    if len(pos) == 0:
        return {
            "n_bars": 0,
            "exposure": float("nan"),
            "n_trades": float("nan"),
            "avg_trade_len": float("nan"),
            "p_enter": float("nan"),
            "p_exit": float("nan"),
            "excess_vs_BH": float("nan"),
            "excess_random_median": float("nan"),
            "excess_random_mean": float("nan"),
            "excess_random_std": float("nan"),
            "p_value": float("nan"),
            "timing_significant": False,
            "skip_reason": "empty_series",
        }

    stats = compute_position_stats(pos)
    if (not np.isfinite(stats.exposure)) or stats.exposure <= eps:
        return {
            "n_bars": int(len(pos)),
            "exposure": float(stats.exposure),
            "n_trades": float(stats.n_trades),
            "avg_trade_len": float(stats.avg_trade_len),
            "p_enter": float("nan"),
            "p_exit": float("nan"),
            "excess_vs_BH": float("nan"),
            "excess_random_median": float("nan"),
            "excess_random_mean": float("nan"),
            "excess_random_std": float("nan"),
            "p_value": float("nan"),
            "timing_significant": False,
            "skip_reason": "zero_exposure",
        }
    if (not np.isfinite(stats.n_trades)) or stats.n_trades <= eps:
        return {
            "n_bars": int(len(pos)),
            "exposure": float(stats.exposure),
            "n_trades": float(stats.n_trades),
            "avg_trade_len": float(stats.avg_trade_len),
            "p_enter": float("nan"),
            "p_exit": float("nan"),
            "excess_vs_BH": float("nan"),
            "excess_random_median": float("nan"),
            "excess_random_mean": float("nan"),
            "excess_random_std": float("nan"),
            "p_value": float("nan"),
            "timing_significant": False,
            "skip_reason": "zero_trades",
        }

    mk = calibrate_markov_params(exposure=stats.exposure, n_trades=stats.n_trades, n_bars=len(pos))

    u_strategy = compute_utility_series(rs, lam=float(lam))
    u_bench = compute_utility_series(rb, lam=float(lam))
    excess_real = float((u_strategy - u_bench).mean())

    random_excess = simulate_random_excess_distribution(
        ret_bench=rb,
        exposure=stats.exposure,
        p_enter=mk.p_enter,
        p_exit=mk.p_exit,
        entry_cost=float(entry_cost),
        exit_cost=float(exit_cost),
        lam=float(lam),
        k=int(k),
        chunk_size=int(chunk_size),
        seed=int(seed),
    )

    if random_excess.size == 0:
        p_value = float("nan")
        ex_med = float("nan")
        ex_mean = float("nan")
        ex_std = float("nan")
    else:
        ex_med = float(np.median(random_excess))
        ex_mean = float(np.mean(random_excess))
        ex_std = float(np.std(random_excess, ddof=0))
        p_value = float((1.0 + np.sum(random_excess >= excess_real)) / (1.0 + float(len(random_excess))))

    return {
        "n_bars": int(len(pos)),
        "exposure": float(stats.exposure),
        "n_trades": float(stats.n_trades),
        "avg_trade_len": float(stats.avg_trade_len),
        "p_enter": float(mk.p_enter),
        "p_exit": float(mk.p_exit),
        "excess_vs_BH": float(excess_real),
        "excess_random_median": float(ex_med),
        "excess_random_mean": float(ex_mean),
        "excess_random_std": float(ex_std),
        "p_value": float(p_value),
        "timing_significant": bool(np.isfinite(p_value) and p_value < 0.05),
        "skip_reason": None,
    }


def strategy_family(strategy_id: str) -> str:
    code = str(strategy_id).split(":", 1)[0].upper()
    if code in {"T1", "T2", "T3"}:
        return "Trend"
    if code in {"M1"}:
        return "Momentum"
    if code in {"R1", "R2", "R3"}:
        return "MeanRev"
    if code in {"R1OC", "R2OC", "R3OC"}:
        return "MeanRev+OC"
    if code in {"S1", "S2", "S3", "S4", "S5"}:
        return "Synergy"
    if code in {"S1OC", "S2OC"}:
        return "Synergy+OC"
    if code.startswith("BENCH"):
        return "Benchmark"
    return "Other"


def build_random_timing_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "timeframe",
                "scenario",
                "lam",
                "n_tests",
                "n_significant",
                "pct_significant",
                "median_p_value",
                "median_exposure",
            ]
        )

    x = results.copy()
    x = x[x["skip_reason"].isna()].copy()
    if x.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "timeframe",
                "scenario",
                "lam",
                "n_tests",
                "n_significant",
                "pct_significant",
                "median_p_value",
                "median_exposure",
            ]
        )

    agg = (
        x.groupby(["family", "timeframe", "scenario", "lam"], as_index=False)
        .agg(
            n_tests=("strategy_id", "size"),
            n_significant=("timing_significant", "sum"),
            median_p_value=("p_value", "median"),
            median_exposure=("exposure", "median"),
        )
        .reset_index(drop=True)
    )
    agg["n_significant"] = agg["n_significant"].astype(int)
    agg["n_tests"] = agg["n_tests"].astype(int)
    agg["pct_significant"] = np.where(
        agg["n_tests"] > 0,
        agg["n_significant"] / agg["n_tests"],
        np.nan,
    )
    return agg.sort_values(["timeframe", "scenario", "lam", "family"]).reset_index(drop=True)
