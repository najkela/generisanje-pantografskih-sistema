"""Провере ваљаности топологије и конструкција solving order-а (README 1.4).

Редослед провера је намерно јефтино → скупо: структурне провере, провера степена,
провера ослонца crank-а, тек онда BFS конструкција solving order-а, па провера
положаја tracer-a (провера 5, README 1.4, BASELINE_SPEC §2).
DOF=1 се обезбеђује имплицитно кроз BFS, не рачунањем формуле — `degrees_of_freedom`
је чиста дијагностика.
"""

from dataclasses import dataclass

from .genome import CRANK, FIXED_A, FIXED_B, Topology, tracer


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
    seen: set[tuple[int, int]] = set()
    connected: set[int] = set()
    for a, b in topology.edges:
        if a == b:
            raise InvalidTopology(f"Ивица чини петљу на чвору {a}.")
        edge = (a, b) if a < b else (b, a)
        if edge in seen:
            raise InvalidTopology(f"Ивица {edge} је дуплирана.")
        seen.add(edge)
        connected.add(a)
        connected.add(b)
    isolated = set(range(topology.n_nodes)) - connected
    if isolated:
        raise InvalidTopology(f"Изоловани чворови: {sorted(isolated)}.")


def check_degrees(topology: Topology) -> None:
    """Тачно 2 fixed, тачно 1 crank, сваки floating чвор укупног степена ≥ 2.

    Пажња: ово је услов *степена*, не услов решивости — в. `solving_order`
    (README 1.4, корак 2, исправка од 07.08.). Fixed, crank и tracer немају услов
    степена овде; улоге су одређене искључиво индексом (README 1.3).
    """
    n = topology.n_nodes
    t = tracer(n)
    for node in range(n):
        if node in (FIXED_A, FIXED_B, CRANK, t):
            continue
        if topology.degree(node) < 2:
            raise InvalidTopology(f"Floating чвор {node} има степен < 2.")


def check_crank_support(topology: Topology) -> None:
    """Тачно једна ивица crank–fixed, и то мора бити (0,2) (README 1.4, корак 3)."""
    crank_fixed_edges = [
        (a, b)
        for a, b in topology.edges
        if {a, b} & {CRANK} and {a, b} & {FIXED_A, FIXED_B}
    ]
    if len(crank_fixed_edges) != 1:
        raise InvalidTopology(
            f"Crank мора имати тачно један ослонац на fixed чвору, нађено {len(crank_fixed_edges)}."
        )
    if set(crank_fixed_edges[0]) != {FIXED_A, CRANK}:
        raise InvalidTopology("Crank мора бити ослоњен искључиво на чвор 0, никад на чвор 1.")


def solving_order(topology: Topology) -> list[Step]:
    """BFS конструкција solving order-а (README 1.4, корак 4; једнозначност — README 1.2.1).

    Креће од `known = {0, 1, 2}`; у сваком пролазу тражи чвор са *тачно 2 позната
    суседа* и премешта га у `known` — детерминистички tie-break бира најмањи индекс
    ако их има више. Ако пролаз не направи напредак, а нису сви чворови обиђени →
    циклична зависност → невалидно.
    """
    known = {FIXED_A, FIXED_B, CRANK}
    remaining = set(range(topology.n_nodes)) - known
    order: list[Step] = []

    while remaining:
        candidates = sorted(
            node for node in remaining if len(topology.neighbors(node) & known) == 2
        )
        if not candidates:
            raise InvalidTopology(
                "Циклична зависност — ниједан преостали чвор нема тачно 2 позната суседа."
            )
        node = candidates[0]
        parents = tuple(sorted(topology.neighbors(node) & known))
        order.append(Step(target=node, parents=parents))
        known.add(node)
        remaining.remove(node)

    return order


def validate(topology: Topology) -> list[Step]:
    """Пуна валидација; враћа solving order или подиже `InvalidTopology` (README 1.4).

    Провера 5 (положај tracer-a, BASELINE_SPEC §2): solving order се мора завршавати
    чвором `n−1` — ниједан чвор не сме висити на tracer-у (README 1.3).
    """
    check_structure(topology)
    check_degrees(topology)
    check_crank_support(topology)
    order = solving_order(topology)
    if not order or order[-1].target != tracer(topology.n_nodes):
        raise InvalidTopology("Tracer мора бити последњи решен чвор у solving order-у.")
    return order


def degrees_of_freedom(topology: Topology) -> int:
    """DOF = 2(n−2) − j (README 1.4, исправка 30.08.). Само дијагностика, не улази у валидацију."""
    return 2 * (topology.n_nodes - 2) - len(topology.edges)
