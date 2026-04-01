from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class StrategySpec:
    code: str
    name: str
    cls: Any                         # backtesting.Strategy subclass
    param_grid: Dict[str, Sequence]  # grid for bt.optimize
    constraint: Optional[Callable[[Any], bool]] = None
    group: str = ""
    save_heatmap: bool = False

    @property
    def strategy_id(self) -> str:
        return f"{self.code}:{self.name}"


def build_registry(
    *,
    strategies: Optional[List[StrategySpec]] = None,
) -> List[StrategySpec]:
    """
    Return the default registry. If `strategies` is provided, it is returned as-is.
    This hook makes it easy to unit test / override.
    """
    if strategies is not None:
        return strategies

    # Import strategy classes locally to avoid import-time side effects / cycles.
    from src.strategies.trend import SMACrossover, EMACrossover, DonchianBreakout
    from src.strategies.momentum import TSMomentum
    from src.strategies.meanrev import RSIMeanReversion, BollingerMeanReversion, ZScoreMeanReversion
    from src.strategies.synergy import (
        MAFilterRSI,
        MA200FilterBollinger,
        BreakoutConfirmMA,
        MACrossTSMOMConfirm,
        SimpleEnsemble,
    )
    from src.strategies.benchmarks import BuyHold

    return [
        StrategySpec(
            code="T1",
            name="SMA_Crossover",
            cls=SMACrossover,
            param_grid={"fast": [5, 10, 20, 30, 50], "slow": [60, 100, 150, 200, 250, 300]},
            constraint=lambda p: p.fast < p.slow,
            group="trend",
            save_heatmap=True,
        ),
        StrategySpec(
            code="T2",
            name="EMA_Crossover",
            cls=EMACrossover,
            param_grid={"fast": [5, 10, 20, 30, 50], "slow": [60, 100, 150, 200, 250, 300]},
            constraint=lambda p: p.fast < p.slow,
            group="trend",
        ),
        StrategySpec(
            code="T3",
            name="Donchian_Breakout",
            cls=DonchianBreakout,
            param_grid={"N": [20, 50, 100, 200], "exit": [10, 20, 50]},
            group="trend",
            save_heatmap=True,
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
            param_grid={"n": [20, 50, 100], "k": [1.5, 2.0, 2.5], "exit_mode": ["midline", "fixed"]},
            group="meanrev",
        ),
        StrategySpec(
            code="R3",
            name="ZScore_MR",
            cls=ZScoreMeanReversion,
            param_grid={"n": [20, 50, 100, 200], "entry_z": [1.0, 1.5, 2.0, 2.5], "exit_z": [0.0, 0.5, 1.0]},
            group="meanrev",
        ),
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


def get_benchmark() -> StrategySpec:
    from src.strategies.benchmarks import BuyHold
    return StrategySpec(code="BENCH", name="BuyHold", cls=BuyHold, param_grid={}, group="benchmark")


def maybe_filter_debug(registry: List[StrategySpec], *, enabled: bool, only_strategies: List[str]) -> List[StrategySpec]:
    if not enabled:
        return registry
    allow = set(only_strategies)
    return [s for s in registry if s.strategy_id in allow]