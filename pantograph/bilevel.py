"""Наш метод: спољашњи ГА над топологијом + унутрашњи ЦМА-ЕС над геометријом (README 2.3).

Фитнес јединке спољашњег ГА = фитнес најбоље јединке коју унутрашњи ЦМА-ЕС нађе
за ту топологију. Спољашњи ГА по default-у користи само само-мутацију, без crossover-а (H2).
"""

import numpy as np

from .curve import TargetCurve
from .genome import Topology


def inner_cmaes(
    topology: Topology,
    initial_mean: np.ndarray,
    initial_sigma: float,
    target: TargetCurve,
    n: int,
    budget,
):
    """ЦМА-ЕС над вектором геометрије x ∈ R^(2n−2) за фиксну топологију (README 2.3).

    Број итерација K је динамички — плато-детекција као early stopping (README 2.4).
    Референтне фиксне вредности за поређење: 5 / 15 / 40 (H3).
    Враћа најбољи вектор, најбољи фитнес и стварно потрошен број позива симулатора.
    """
    raise NotImplementedError


def warm_start(parent_geometry: np.ndarray, new_topology: Topology) -> np.ndarray:
    """Почетна средина ЦМА-ЕС расподеле наслеђена од родитеља (README 2.4).

    Механизам при ДОДАВАЊУ ЧВОРА је ОТВОРЕНО ПИТАЊЕ 2.1 — између којих постојећих
    чворова се интерполира нова позиција и са којим почетним σ за нове димензије.
    Не бирати самоиницијативно.
    """
    raise NotImplementedError


def outer_ga(target: TargetCurve, budget, seed: int, population_size: int):
    """Главна петља bilevel метода; троши стварно потрошен K по евалуацији (README 3.3)."""
    raise NotImplementedError
