"""Forward kinematics: од генома и угла crank-а до путање tracer чвора (README 1.2).

Ово је најтоплија петља пројекта и уједно јединица трошка буџета (README 3.3):
један позив `simulate` = један позив симулатора.
"""

import numpy as np

from .genome import CRANK, FIXED_A, Topology, link_lengths, tracer
from .validation import Step


def _link_length(lengths: dict[tuple[int, int], float], i: int, j: int) -> float:
    """Дужина крака (i,j) без обзира на редослед у кључу речника."""
    return lengths[(i, j)] if (i, j) in lengths else lengths[(j, i)]


class CallCounter:
    """Експлицитан бројач позива симулатора — једина дозвољена глобална статистика."""

    def __init__(self) -> None:
        self.calls = 0

    def increment(self, by: int = 1) -> None:
        self.calls += by

    def reset(self) -> None:
        self.calls = 0


def circle_intersect_pair(
    p1: np.ndarray,
    p2: np.ndarray,
    r1: float,
    r2: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Обе тачке пресека кругова (p1,r1) и (p2,r2); прва одговара грани s=+1, друга s=−1.

    Пресек не постоји ако је `d > r1+r2` или `d < |r1−r2|` (README 1.4) — тада враћа `None`.
    Конвенција гране (README 1.2.1): нека је `u = (p2−p1)/d`; `s=+1` је тачка помакнута
    од средишње тачке дужи p1p2 у смеру ротације `u` за +90°, `s=−1` у супротном смеру.
    Ово је основна геометрија коју користе и `circle_intersect` (симулација) и
    `genome.to_sequence`/`from_sequence` (канонизација, README 1.2.1).
    """
    d = float(np.linalg.norm(p2 - p1))
    if d == 0.0:
        return None
    if d > r1 + r2 or d < abs(r1 - r2):
        return None
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = float(np.sqrt(max(r1**2 - a**2, 0.0)))
    u = (p2 - p1) / d
    midpoint = p1 + a * u
    perp = np.array([-u[1], u[0]])
    return midpoint + h * perp, midpoint - h * perp


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
    pair = circle_intersect_pair(p1, p2, r1, r2)
    if pair is None:
        return None
    p_plus, p_minus = pair
    d_plus = np.linalg.norm(p_plus - previous)
    d_minus = np.linalg.norm(p_minus - previous)
    return p_plus if d_plus <= d_minus else p_minus


def positions_at(
    topology: Topology,
    lengths: dict[tuple[int, int], float],
    previous: np.ndarray,
    order: list[Step],
    angle: float,
) -> np.ndarray | None:
    """Позиције свих чворова за дати угао crank-а; `None` ако пресек не постоји.

    Fixed чворови (0,1) се преузимају непромењени из `previous`. Crank(2) се поставља
    директно на растојању `l(0,2)` под апсолутним углом `angle` (нема drift-а — не рачуна
    се инкрементално). Остали чворови се решавају редом из `order` (README 1.2), бирајући
    грану ближу претходном положају истог чвора (`circle_intersect`).
    """
    positions = previous.copy()
    r02 = _link_length(lengths, FIXED_A, CRANK)
    positions[CRANK] = positions[FIXED_A] + r02 * np.array([np.cos(angle), np.sin(angle)])

    for step in order:
        u = step.target
        a, b = step.parents
        ra = _link_length(lengths, a, u)
        rb = _link_length(lengths, b, u)
        point = circle_intersect(positions[a], positions[b], ra, rb, previous[u])
        if point is None:
            return None
        positions[u] = point

    return positions


def simulate(
    topology: Topology,
    coords: np.ndarray,
    order: list[Step],
    n: int,
) -> np.ndarray | None:
    """Путања tracer чвора при пуној ротацији crank-а, `n` равномерних корака по θ ∈ [0, 2π).

    Дужине кракова се рачунају једном из `coords` (θ=0 геометрија) и остају константне
    (README 1.2.1). Апсолутни угао crank-а у θ=0 (`theta0`) се чита из `coords` — прва
    тачка путање тако тачно репродукује улазну геометрију, без вештачког скока на почетку.

    Враћа низ облика (n, 2) или `None` ако геометрија у неком тренутку није решива.
    """
    lengths = link_lengths(topology, coords)
    tracer_index = tracer(topology.n_nodes)
    theta0 = float(np.arctan2(*(coords[CRANK] - coords[FIXED_A])[::-1]))

    positions = coords.copy()
    path = np.zeros((n, 2))
    for i, phi in enumerate(np.linspace(0.0, 2 * np.pi, n, endpoint=False)):
        positions = positions_at(topology, lengths, positions, order, theta0 + phi)
        if positions is None:
            return None
        path[i] = positions[tracer_index]

    return path
