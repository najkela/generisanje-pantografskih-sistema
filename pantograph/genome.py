"""Структуре генома: топологија, координате, пресликавање вектор ↔ чворови (README 1.2, 1.3)."""

from dataclasses import dataclass

import numpy as np

# Конвенција индекса (README 1.3) — улога чвора је одређена искључиво индексом.
FIXED_A = 0   # закуцан у (0,0)
FIXED_B = 1   # геометријски слободан: носи ротацију и скалирање механизма
CRANK = 2     # обавезна ивица (0,2); никад везан за чвор 1


def tracer(n_nodes: int) -> int:
    """Индекс tracer чвора — увек последњи чвор (README 1.3)."""
    return n_nodes - 1


def role(node: int, n_nodes: int) -> str:
    """Враћа 'фиксни' | 'ручица' | 'трагач' | 'слободни' за дати индекс (README 1.3).

    Повратна вредност је намењена испису кориснику, отуда ћирилица.
    """
    if node in (FIXED_A, FIXED_B):
        return "фиксни"
    if node == CRANK:
        return "ручица"
    if node == tracer(n_nodes):
        return "трагач"
    return "слободни"


@dataclass
class Topology:
    """Дискретни део решења: број чворова и скуп неусмерених ивица (README 1.2)."""

    n_nodes: int
    edges: list[tuple[int, int]]

    def neighbors(self, node: int) -> set[int]:
        """Скуп суседа датог чвора."""
        return {b for a, b in self.edges if a == node} | {a for a, b in self.edges if b == node}

    def degree(self, node: int) -> int:
        """Укупан степен чвора."""
        return len(self.neighbors(node))

    def copy(self) -> "Topology":
        """Дубока копија — оператори мутације не смеју мењати родитеља на месту."""
        return Topology(n_nodes=self.n_nodes, edges=[edge for edge in self.edges])


@dataclass
class Genome:
    """Монолитни геном за baseline: топологија + координате у тренутку θ=0 (README 1.2, 2.2)."""

    topology: Topology
    coords: np.ndarray  # облика (n_nodes, 2)


def to_vector(coords: np.ndarray) -> np.ndarray:
    """Координате → раван вектор дужине 2n−2, без чвора 0 (README 1.2).

    Пресликавање мора бити детерминистичко и фиксно за дату топологију: редом
    чворови 1..n−1, за сваки чвор прво x па y координата.
    """
    return coords[1:].reshape(-1)


def from_vector(vector: np.ndarray, n_nodes: int) -> np.ndarray:
    """Инверз функције `to_vector`; чвор 0 се враћа на (0,0)."""
    coords = np.zeros((n_nodes, 2))
    coords[1:] = np.asarray(vector).reshape(n_nodes - 1, 2)
    return coords


def link_lengths(topology: Topology, coords: np.ndarray) -> dict[tuple[int, int], float]:
    """Дужине кракова из почетне геометрије; константне током целе симулације (README 1.2)."""
    return {
        (i, j): float(np.linalg.norm(coords[i] - coords[j]))
        for i, j in topology.edges
    }


@dataclass
class SequenceGene:
    """Ген једног корака секвенце склапања (README 1.2.1).

    `a`, `b` су ПОЗИЦИЈЕ ослонаца у секвенци (не индекси чворова из оригиналног генома).
    `rho_a`, `rho_b` су односи дужина полуга према размаку ослонаца у θ=0; важе искључиво
    у том тренутку (README 1.2.1, упозорење) — једном фиксирани, симулација их не мења.
    `s` бира грану circle-circle пресека (+1 или −1).
    """

    a: int
    b: int
    rho_a: float
    rho_b: float
    s: int


@dataclass
class Sequence:
    """Канонски запис јединке: језгро у апсолутним координатама + гени корака (README 1.2.1)."""

    core: np.ndarray  # облика (3, 2): q0(0), q0(1), q0(2), дословне координате
    genes: list[SequenceGene]


def to_sequence(topology: Topology, coords: np.ndarray) -> Sequence:
    """Канонизује геном у секвенцу склапања преко BFS solving order-а (README 1.2.1).

    Позиције `a`, `b` у сваком гену показују на место ослонца У СЕКВЕНЦИ — мапирање је
    потребно јер BFS solving order не мора да поклопи нумеричке индексе чворова.
    """
    # Локални увоз: `validation` и `simulator` увозе из `genome`, циклични увоз на врху фајла.
    from .simulator import circle_intersect_pair
    from .validation import solving_order

    order = solving_order(topology)

    position_of = {FIXED_A: 0, FIXED_B: 1, CRANK: 2}
    for offset, step in enumerate(order):
        position_of[step.target] = offset + 3

    genes: list[SequenceGene] = []
    for step in order:
        u = step.target
        a_node, b_node = step.parents
        pa, pb = coords[a_node], coords[b_node]
        d = float(np.linalg.norm(pb - pa))
        rho_a = float(np.linalg.norm(coords[u] - pa) / d)
        rho_b = float(np.linalg.norm(coords[u] - pb) / d)
        p_plus, p_minus = circle_intersect_pair(pa, pb, rho_a * d, rho_b * d)
        s = 1 if np.linalg.norm(coords[u] - p_plus) <= np.linalg.norm(coords[u] - p_minus) else -1
        genes.append(
            SequenceGene(a=position_of[a_node], b=position_of[b_node], rho_a=rho_a, rho_b=rho_b, s=s)
        )

    return Sequence(core=coords[[FIXED_A, FIXED_B, CRANK]].copy(), genes=genes)


def from_sequence(seq: Sequence) -> tuple[Topology, np.ndarray] | None:
    """Реконструише топологију и координате у θ=0 из секвенце склапања (README 1.2.1, 2.2 корак 3).

    Дужине кракова се фиксирају једном, из размака ослонаца у реконструисаној геометрији.
    Враћа `None` ако у неком кораку circle-circle пресек не постоји (README 1.4).
    """
    from .simulator import circle_intersect_pair

    n_nodes = 3 + len(seq.genes)
    coords = np.zeros((n_nodes, 2))
    coords[0:3] = seq.core
    edges: list[tuple[int, int]] = [(FIXED_A, CRANK)]

    for k, gene in enumerate(seq.genes, start=3):
        pa, pb = coords[gene.a], coords[gene.b]
        d = float(np.linalg.norm(pb - pa))
        pair = circle_intersect_pair(pa, pb, gene.rho_a * d, gene.rho_b * d)
        if pair is None:
            return None
        p_plus, p_minus = pair
        coords[k] = p_plus if gene.s == 1 else p_minus
        edges.append((gene.a, k))
        edges.append((gene.b, k))

    return Topology(n_nodes=n_nodes, edges=edges), coords
