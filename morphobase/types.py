from dataclasses import dataclass
from enum import Enum
from typing import Any


class Stage(str, Enum):
    SEED = 'seed'
    EXPLORATORY = 'exploratory'
    DIFFERENTIATING = 'differentiating'
    MATURE = 'mature'
    DEDIFFERENTIATING = 'dedifferentiating'
    PRUNABLE = 'prunable'


class RunVerdict(str, Enum):
    PASS = 'pass'
    UNKNOWN = 'unknown'
    UNSTABLE = 'unstable'
    DEGENERATE_LOCK = 'degenerate_lock'
    CHRONIC_GROWTH = 'chronic_growth'
    PLASTICITY_LOSS = 'plasticity_loss'
    DEAD_FIELD = 'dead_field'
    PSEUDO_MATURITY = 'pseudo_maturity'


@dataclass(slots=True)
class Alert:
    name: str
    triggered: bool
    message: str = ''
    payload: dict[str, Any] | None = None
