"""Baseline: чист ГА над монолитним геномом — истовремена оптимизација (README 2.2).

Baseline мора бити best-effort, не намерно ослабљен — поређење H1 губи смисао
ако baseline нема праведну шансу.

Генерацијска шема из оригиналног предлога: 20% елите пролази непромењено;
преосталих 80% настаје укрштањем — 30% деца два родитеља из горњих 20%,
30% деца из мешовитог пара (горњих 20% × средњих 20–80%), 20% деца из средњих 20–80%.
Мутацији су подложне само новонастале јединке, не и пренета елита.
"""

import numpy as np

from .curve import TargetCurve
from .genome import Genome


def select(population: list[Genome], scores: np.ndarray) -> list[Genome]:
    """Рангирање и издвајање елите (горњих 20%) и средњег слоја (20–80%)."""
    raise NotImplementedError


def crossover(parent_a: Genome, parent_b: Genome, rng: np.random.Generator) -> Genome:
    """Укрштање два монолитна генома (топологија и координате заједно)."""
    raise NotImplementedError


def evolve(target: TargetCurve, budget, seed: int, population_size: int):
    """Главна петља baseline ГА; троши буџет по 1 позиву симулатора по јединки (README 3.3)."""
    raise NotImplementedError
