import dataclasses
from typing import List, Tuple, Dict
import numpy as np


@dataclasses.dataclass
class ModelParams:
    mu1: float = 1.0
    mu2: float = 1.0
    c1: float = 1.0
    c2: float = 1.0
    c3: float = 10.0
    c4: float = 10.0
    gamma: float = 1.0
    d: float = 1.0
    x_max: float = 10.0
    v_max: float = 2.0
    v_min: float = -2.0
    s1: float = 0.5
    s2: float = 0.5
    s3: float = 0.5
    const: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class TrainingConfig:
    M: int = 100
    N: int = 100
    D: int = 4
    T_standard: float = 12.0
    T_total: float = 48.0
    block_size: float = 12.0
    
    passes: int = 2
    empirical_jitter_scale: float = 0.02
    pass1_warm_start_from_next: bool = False
    exact_solution: str = "none"
    training_plan_csv: str = ""
    
    # Plans
    stage_plan: List[Tuple[int, float]] = dataclasses.field(
        default_factory=lambda: [(5000, 1e-3), (5000, 5e-4), (5000, 1e-4), (5000, 5e-5)]
    )
    final_plan: List[Tuple[int, float]] = dataclasses.field(
        default_factory=lambda: [(5000, 1e-5), (5000, 5e-6)]
    )

    @property
    def layers(self) -> List[int]:
        return [self.D + 1] + 4 * [256] + [1]
