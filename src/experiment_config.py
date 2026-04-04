from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass(frozen=True)
class WFConfig:
    train_years: int = 3
    test_months: int = 6
    step_months: int = 6
    warmup_bars: int = 300
    require_full_warmup: bool = True


def warmup_bars_for_timeframe(timeframe: str) -> int:
    tf = timeframe.strip().lower()
    mapping = {"1d": 300, "4h": 1800}
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe={timeframe!r}. Expected one of: {sorted(mapping)}")
    return mapping[tf]


def periods_per_year_for_timeframe(timeframe: str) -> int:
    tf = timeframe.strip().lower()
    mapping = {"1d": 365, "4h": 365 * 6}
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe={timeframe!r}. Expected one of: {sorted(mapping)}")
    return mapping[tf]


@dataclass(frozen=True)
class BacktestConfig:
    cash: float = 2_000_000.0
    trade_on_close: bool = False
    exclusive_orders: bool = True
    hedging: bool = False
    margin: float = 1.0
    max_rel_size: float = 0.999  # sizing safety buffer
    finalize_trades: bool = True


@dataclass(frozen=True)
class ObjectiveConfig:
    min_trades: int = 10
    min_exposure: float = 0.10
    penalty: float = 1e6


@dataclass(frozen=True)
class RCSPAConfig:
    block_size: int = 10
    reps: int = 2000
    seed: int = 42
    studentize: bool = True


@dataclass(frozen=True)
class ArtifactsConfig:
    save_heatmaps: bool = True
    fail_fast: bool = True
    use_cache_if_exists: bool = False


@dataclass(frozen=True)
class DebugConfig:
    enabled: bool = False
    only_strategies: List[str] = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ExperimentConfig:
    data_paths: Dict[str, Path]
    out_dir: Path
    results_dir: Path
    figures_dir: Path
    wf: WFConfig = WFConfig()
    bt: BacktestConfig = BacktestConfig()
    costs: List[str] = None  # type: ignore[assignment]
    objective: ObjectiveConfig = ObjectiveConfig()
    rc_spa: RCSPAConfig = RCSPAConfig()
    artifacts: ArtifactsConfig = ArtifactsConfig()
    debug: DebugConfig = DebugConfig()

    def __post_init__(self):
        # dataclasses with mutable defaults: ensure lists are not shared
        object.__setattr__(self, "costs", self.costs or ["Low", "Base", "High"])
        object.__setattr__(self, "debug", DebugConfig(
            enabled=self.debug.enabled,
            only_strategies=self.debug.only_strategies or ["T1:SMA_Crossover", "BENCH:BuyHold"]
        ))


def default_config(project_root: Path) -> ExperimentConfig:
    data_paths = {
        "BTCUSDT": Path("../data/data_raw/BTCUSDT_spot_1d_monthly_2020-01_2025-12.csv"),
        "ETHUSDT": Path("../data/data_raw/ETHUSDT_spot_1d_monthly_2020-01_2025-12.csv"),
    }
    out_dir = project_root
    results_dir = out_dir / "results"
    figures_dir = out_dir / "figures"

    return ExperimentConfig(
        data_paths=data_paths,
        out_dir=out_dir,
        results_dir=results_dir,
        figures_dir=figures_dir,
    )


def _json_default(x: Any):
    if isinstance(x, Path):
        return str(x)
    return str(x)


def save_config_json(cfg: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(cfg), default=_json_default, indent=2),
        encoding="utf-8",
    )
