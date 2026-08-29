"""Провере ваљаности топологије и конструкција solving order-а (README 1.4).

Редослед провера је намерно јефтино → скупо: структурне провере, провера степена,
провера ослонца crank-а, тек онда BFS конструкција solving order-а.
DOF=1 се обезбеђује имплицитно кроз BFS, не рачунањем Грублерове формуле.
"""

from dataclasses import dataclass

from .genome import Topology


class InvalidTopology(Exception):
    """Топологија не задовољава неки од услова из README 1.4.

    Порука изузетка се пише ћирилицом — иде директно кориснику у лог.
    """


@dataclass
class Step:
    """Један корак solving order-а: циљни чвор и два већ позната родитеља."""

    target: int
    parents: tuple[int, int]


def check_structure(topology: Topology) -> None:
    """Без петљи ка себи, без дуплираних ивица, без изолованих чворова (README 1.4, корак 1)."""
    raise NotImplementedError


def check_degrees(topology: Topology) -> None:
    """Тачно 2 fixed, тачно 1 crank, сваки floating чвор укупног степена ≥ 2.

    Пажња: ово је услов *степена*, не услов решивости — в. `solving_order`
    (README 1.4, корак 2, исправка од 07.08.).
    """
    raise NotImplementedError


def check_crank_support(topology: Topology) -> None:
    """Тачно једна ивица crank–fixed, и то мора бити (0,2) (README 1.4, корак 3)."""
    raise NotImplementedError


def solving_order(topology: Topology) -> list[Step]:
    """BFS конструкција solving order-а (README 1.4, корак 4).

    Креће од `known = {0, 1, 2}`; у сваком пролазу тражи floating чвор са
    *тачно 2 позната суседа* и премешта га у `known`. Ако пролаз не направи
    напредак, а нису сви чворови обиђени → циклична зависност → невалидно.
    """
    raise NotImplementedError


def validate(topology: Topology) -> list[Step]:
    """Пуна валидација; враћа solving order или подиже `InvalidTopology`."""
    raise NotImplementedError


def degrees_of_freedom(topology: Topology) -> int:
    """Грублер–Куцбах: DOF = 3(n−1) − 2j. Само дијагностика, не улази у валидацију."""
    raise NotImplementedError
