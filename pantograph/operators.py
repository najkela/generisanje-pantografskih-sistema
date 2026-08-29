"""Иницијализација и мутациони оператори (README 2.3, 2.4, 4.1; одлука од 09.08.).

Улоге чворова (fixed/crank/tracer) се НИКАД не мењају мутацијом.
Невалидна мутација остаје у популацији са казном — мутација се не понавља до валидне.
"""

import numpy as np

from .genome import Genome, Topology


def random_initial_topology(n_nodes: int, rng: np.random.Generator) -> Topology:
    """Конструктивна иницијализација: језгро {0,1,2} + ивица (0,2), па floating чворови.

    Сваки нови floating чвор се везује за тачно 2 постојећа — иста логика која
    гарантује BFS-решивост, па је свака иницијализована јединка валидна (README 4.1).
    """
    raise NotImplementedError


def random_initial_coords(n_nodes: int, span: float, rng: np.random.Generator) -> np.ndarray:
    """Почетне координате; `span` је отворена калибрациона ставка (README 4.1, питање 3.3)."""
    raise NotImplementedError


# --- тополошки оператори (закључано, README 2.3) ---------------------------------


def add_node(topology: Topology, coords: np.ndarray, rng: np.random.Generator):
    """Нови floating чвор везан за 2 постојећа.

    Почетна позиција новог чвора је ОТВОРЕНО ПИТАЊЕ 2.1 (warm-start при додавању чвора) —
    не бирати механизам самоиницијативно.
    """
    raise NotImplementedError


def remove_node(topology: Topology, coords: np.ndarray, rng: np.random.Generator):
    """Уклања floating чвор; fixed, crank и tracer се никад не бришу."""
    raise NotImplementedError


def add_edge(topology: Topology, rng: np.random.Generator) -> Topology:
    """Додаје ивицу између два постојећа чвора."""
    raise NotImplementedError


def remove_edge(topology: Topology, rng: np.random.Generator) -> Topology:
    """Брише ивицу; ивица (0,2) је обавезна и не сме бити обрисана."""
    raise NotImplementedError


# --- координатни оператор (baseline; одлука од 09.08.) ---------------------------


def mutate_coords(
    coords: np.ndarray,
    k: float,
    per_node_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-node изотропни Гаусов шум над свим чворовима осим чвора 0 (одлука 09.08.).

    Сваки чвор независно, вероватноћом `per_node_prob`, добија Δ ~ N(0, σ²I).
    σ = `k` · просечна дужина крака те јединке у тренутку мутације — сразмерно величини
    јединке, јер координате нису нормализоване (нормализује се само циљна крива, README 1.5).
    Коефицијент `k` опада кроз генерације: веће мутације рано, финије касно.

    Спрега са тополошком мутацијом је НЕЗАВИСНА: свака новоукрштена јединка засебно
    „баца новчић" за тополошку и за координатну мутацију.
    """
    raise NotImplementedError


def mutate_baseline(
    genome: Genome,
    p_topo: float,
    p_coord: float,
    k: float,
    rng: np.random.Generator,
) -> Genome:
    """Једна мутација монолитног генома: независни новчићи за топологију и координате."""
    raise NotImplementedError
