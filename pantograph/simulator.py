"""Forward kinematics: од генома и угла crank-а до путање tracer чвора (README 1.2).

Ово је најтоплија петља пројекта и уједно јединица трошка буџета (README 3.3):
један позив `simulate` = један позив симулатора.
"""

import numpy as np

from .genome import Topology
from .validation import Step


class CallCounter:
    """Експлицитан бројач позива симулатора — једина дозвољена глобална статистика."""

    def __init__(self) -> None:
        self.calls = 0

    def increment(self, by: int = 1) -> None:
        self.calls += by

    def reset(self) -> None:
        self.calls = 0


def circle_intersect(
    p1: np.ndarray,
    p2: np.ndarray,
    r1: float,
    r2: float,
    previous: np.ndarray,
) -> np.ndarray | None:
    """Пресек два круга; бира грану ближу претходној позицији чвора.

    Враћа `None` ако пресека нема (d > r1+r2 или d < |r1−r2|) — то је невалидна
    геометрија у том тренутку, коју фитнес претвара у коначну казну (README 1.6).

    Хеуристика избора гране је извор circuit defect проблема (README 4.4, отворено 4.1) —
    не додавати проверу скока док одлука не буде донета.
    """
    raise NotImplementedError


def positions_at(
    topology: Topology,
    lengths: dict[tuple[int, int], float],
    previous: np.ndarray,
    order: list[Step],
    angle: float,
) -> np.ndarray | None:
    """Позиције свих чворова за дати угао crank-а; `None` ако пресек не постоји."""
    raise NotImplementedError


def simulate(
    topology: Topology,
    coords: np.ndarray,
    order: list[Step],
    n: int,
) -> np.ndarray | None:
    """Путања tracer чвора при пуној ротацији crank-а, `n` равномерних корака по θ ∈ [0, 2π).

    Враћа низ облика (n, 2) или `None` ако геометрија у неком тренутку није решива.
    """
    raise NotImplementedError
