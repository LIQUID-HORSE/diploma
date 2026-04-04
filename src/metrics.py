"""Metrics & objectives.

This project evaluates strategies out-of-sample (walk-forward) and reports
common portfolio metrics: Return, CAGR, Sharpe, Sortino, Max Drawdown, Calmar,
#Trades, Exposure.

We keep two layers of metrics:
1) backtesting.py's built-in stats (used on each fold run);
2) a lightweight return-series based calculator (used for stitched OOS).

The primary optimization metric on train windows is Calmar, with sanity penalties
(min trades / min exposure) to avoid degenerate solutions, as required by the spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# Crypto trades 24/7; with daily bars, annualization by 365 is a reasonable default.
DEFAULT_PERIODS_PER_YEAR = 365


@dataclass(frozen=True)
class CostScenario:
    name: str
    commission: float  # proportion per trade (e.g., 0.001 = 10 bps)
    spread: float      # proportion of price (e.g., 0.0002 = 2 bps)


COST_SCENARIOS = {
    "Low": CostScenario("Low", commission=0.00075, spread=0.00010),
    "Base": CostScenario("Base", commission=0.00100, spread=0.00020),
    "High": CostScenario("High", commission=0.00150, spread=0.00050),
}


def equity_from_returns(returns: pd.Series, start: float = 1.0) -> pd.Series:
    r = returns.fillna(0.0).astype(float)
    return (1.0 + r).cumprod() * float(start)


def drawdown_from_equity(equity: pd.Series) -> pd.Series:
    eq = equity.astype(float)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return dd


def max_drawdown(returns: pd.Series) -> float:
    eq = equity_from_returns(returns, start=1.0)
    dd = drawdown_from_equity(eq)
    return float(dd.min()) if len(dd) else 0.0


def cagr(returns: pd.Series, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> float:
    r = returns.fillna(0.0).astype(float)
    if len(r) == 0:
        return 0.0
    total = float((1.0 + r).prod() - 1.0)
    years = len(r) / float(periods_per_year)
    if years <= 0:
        return 0.0
    # Guard against negative base with fractional power when total < -1
    base = max(1e-12, 1.0 + total)
    return float(base ** (1.0 / years) - 1.0)


def annual_vol(returns: pd.Series, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> float:
    r = returns.dropna().astype(float)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR, rf: float = 0.0) -> float:
    r = returns.dropna().astype(float)
    if len(r) < 2:
        return 0.0
    # Convert annual rf to per-period rf approximately
    rf_p = (1.0 + rf) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_p
    sd = excess.std(ddof=0)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR, rf: float = 0.0) -> float:
    r = returns.dropna().astype(float)
    if len(r) < 2:
        return 0.0
    rf_p = (1.0 + rf) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_p
    downside = excess.copy()
    downside[downside > 0] = 0.0
    dd = downside.std(ddof=0)
    if dd == 0:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def calmar_ratio(returns: pd.Series, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> float:
    dd = abs(max_drawdown(returns))
    if dd == 0:
        return 0.0
    return float(cagr(returns, periods_per_year=periods_per_year) / dd)


def compute_metrics_from_returns(
    returns: pd.Series,
    *,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    rf: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute a compact set of metrics from a daily returns series."""
    r = returns.dropna().astype(float)
    if len(r) == 0:
        out: Dict[str, Any] = {
            "n_bars": 0,
            "Total Return": 0.0,
            "CAGR": 0.0,
            "Ann. Vol": 0.0,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "MaxDD": 0.0,
            "Calmar": 0.0,
            "CDaR_95": 0.0,
        }
    else:
        tot = float((1.0 + r).prod() - 1.0)
        out = {
            "n_bars": int(len(r)),
            "Total Return": tot,
            "CAGR": cagr(r, periods_per_year=periods_per_year),
            "Ann. Vol": annual_vol(r, periods_per_year=periods_per_year),
            "Sharpe": sharpe_ratio(r, periods_per_year=periods_per_year, rf=rf),
            "Sortino": sortino_ratio(r, periods_per_year=periods_per_year, rf=rf),
            "MaxDD": float(max_drawdown(r)),
            "Calmar": calmar_ratio(r, periods_per_year=periods_per_year),
            "CDaR_95": cdar_from_returns(returns, alpha=0.95),
        }
    if extra:
        out.update(extra)
    return out


def _get_stat(stats: Any, key: str) -> Optional[Any]:
    """Best-effort getter for backtesting.py stats."""
    try:
        if isinstance(stats, dict) and key in stats:
            return stats[key]
        # stats is usually a pandas Series
        if hasattr(stats, "get"):
            return stats.get(key, None)
        return None
    except Exception:
        return None


def daily_returns_from_backtesting_stats(stats: Any) -> pd.Series:
    """Extract daily returns from backtesting.py stats (equity curve)."""
    eq = None
    if isinstance(stats, dict) and "_equity_curve" in stats:
        eq = stats["_equity_curve"]
    elif hasattr(stats, "_equity_curve"):
        eq = stats._equity_curve  # type: ignore[attr-defined]
    else:
        eq = _get_stat(stats, "_equity_curve")

    if eq is None:
        raise ValueError("Can't find equity curve in stats; expected '_equity_curve'.")

    if isinstance(eq, pd.DataFrame):
        if "Equity" in eq.columns:
            equity = eq["Equity"]
        else:
            # fall back to first column
            equity = eq.iloc[:, 0]
    else:
        raise TypeError("_equity_curve must be a pandas DataFrame")

    r = equity.pct_change().fillna(0.0)
    r.name = "ret"
    # backtesting.py uses the same index as input data; keep it.
    return r


def objective_calmar_with_sanity_penalties(
    stats: Any,
    *,
    # Conservative defaults: reduce risk of selecting "lucky" low-activity configs.
    min_trades: int = 10,
    min_exposure: float = 0.10,
    penalty: float = 1e6,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Objective for train optimization.

    Primary target: Calmar (Annualized return / |MaxDD|), per spec.

    Sanity penalties:
    - too few trades on train
    - too low exposure on train

    The idea is explicitly mentioned in the spec. The exact penalty shape is a design
    choice; here we use a hard rejection with a large negative score.
    """
    # Prefer backtesting.py's own Calmar Ratio if available, else compute from returns.
    cal = _get_stat(stats, "Calmar Ratio")
    if cal is None:
        # backtesting.py versions differ in key naming; try a few
        cal = _get_stat(stats, "Calmar") or _get_stat(stats, "Calmar ratio")
    try:
        calmar = (
            float(cal)
            if cal is not None
            else calmar_ratio(daily_returns_from_backtesting_stats(stats), periods_per_year=periods_per_year)
        )
    except Exception:
        calmar = -np.inf

    n_trades = _get_stat(stats, "# Trades")
    exposure = _get_stat(stats, "Exposure Time [%]")
    try:
        n_trades_i = int(n_trades) if n_trades is not None else 0
    except Exception:
        n_trades_i = 0
    try:
        exposure_f = float(exposure) / 100.0 if exposure is not None else 0.0
    except Exception:
        exposure_f = 0.0

    if n_trades_i < min_trades:
        return -float(penalty) - float(min_trades - n_trades_i) * 1e3
    if exposure_f < min_exposure:
        return -float(penalty) - float(min_exposure - exposure_f) * 1e3

    # Additional guards for pathological values
    if not np.isfinite(calmar):
        return -float(penalty)

    return float(calmar)


def slice_returns_after(returns: pd.Series, start: pd.Timestamp) -> pd.Series:
    r = returns.copy()
    if not isinstance(r.index, pd.DatetimeIndex):
        return r
    return r.loc[r.index >= start]


def drawdown_depth_series(returns: pd.Series) -> pd.Series:
    """
    Drawdown depth as a positive fraction in [0, 1+): 0 means at peak, 0.2 means 20% below peak.
    """
    r = returns.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd_depth = 1.0 - (eq / peak)
    return dd_depth


def cdar_from_returns(returns: pd.Series, *, alpha: float = 0.95) -> float:
    """
    Conditional Drawdown-at-Risk (CDaR) at level alpha.
    We take the mean of drawdown depths above the alpha-quantile.
    """
    dd = drawdown_depth_series(returns)
    dd = dd.replace([np.inf, -np.inf], np.nan).dropna()
    if dd.empty:
        return float("nan")

    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    q = float(dd.quantile(a))
    tail = dd[dd >= q]
    if tail.empty:
        return float("nan")
    return float(tail.mean())
