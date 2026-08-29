"""Буџет, метрике, семе, логовање и плато-детекција (README 2.4, 3.2, 3.3, 4.3)."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Budget:
    """Позив симулатора је јединица трошка, заједничка за оба метода (README 3.3)."""

    max_calls: int = 150
    spent: int = 0

    def spend(self, amount: int = 1) -> None:
        raise NotImplementedError

    @property
    def exhausted(self) -> bool:
        raise NotImplementedError


def plateau_detected(history: list[float], window: int, epsilon: float) -> bool:
    """Клизни прозор W и праг ε над кривом фитнеса (README 2.4).

    Исти механизам се користи на ТРИ места — динамички N, динамички K и критеријум
    заустављања — али прагови не морају бити исти; калибришу се одвојено
    (README 4.2, отворено питање 3.1).
    """
    raise NotImplementedError


def resolution_schedule(generation: int, history: list[float]) -> int:
    """Динамички N: раст 90 → 720 кроз генерације, вођен плато-детекцијом (README 2.4).

    Тачан распоред раста је отворено питање 3.1.
    """
    raise NotImplementedError


@dataclass
class RunLog:
    """Запис једног покретања — основа за све три метрике (README 3.2)."""

    method: str
    curve: str
    seed: int
    error_curve: list[tuple[int, float]] = field(default_factory=list)  # (позиви, најбољи фитнес)
    final_error: float = float("nan")

    def record(self, calls: int, fitness: float) -> None:
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Снима у `results/`. Шта се тачно логује је отворено питање 5.1."""
        raise NotImplementedError


def make_rng(seed: int) -> np.random.Generator:
    """Јединствен извор случајности по покретању.

    Шта тачно семе контролише (иницијализација популације, ЦМА-ЕС узорковање,
    редослед мутација) је ОТВОРЕНО ПИТАЊЕ 5.1 — а конзистентност је једна од три метрике.
    """
    raise NotImplementedError
