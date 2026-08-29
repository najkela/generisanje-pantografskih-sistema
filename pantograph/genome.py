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
