"""Провере ваљаности топологије и solving order преко инваријанте редоследа (README 1.4).

Редослед провера је намерно јефтино → скупо: структурне провере, провера степена,
провера ослонца crank-а, тек онда провера инваријанте редоследа, па провера положаја
tracer-a (провера 5, README 1.4, BASELINE_SPEC §2).

Инваријанта редоследа (измењено 31.08., DECISIONS §16): за сваки чвор `k ≥ 3`, скуп
суседа мањег индекса мора имати тачно 2 елемента — то су ослонци чвора `k`. Solving
order је тада тривијално `[3, ..., n-1]`, исти као редослед индекса — BFS претрага и
tie-break из старије верзије су повучени као беспредметни (сваки оператор из 2.3 већ
гради чворове тим редом). Провера остаје јер је `Topology` обичан скуп ивица који се
склапа и ручно (тестови, `run.py demo`, сутра `bilevel.py`).
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
    """Solving order преко инваријанте редоследа (README 1.4 корак 4, 1.2.1; измењено 31.08.).

    За сваки чвор `k` од 3 до `n-1`: скуп суседа мањег индекса мора имати тачно 2
    елемента — то су његови ослонци, читају се директно (`sorted(m for m in
    neighbors(k) if m < k)`), без претраге. Мање од 2 → чвор неодређен; више → преодређен
    → у оба случаја `InvalidTopology`. Резултујући редослед је увек `[3, ..., n-1]`.
    """
    order: list[Step] = []
    for k in range(3, topology.n_nodes):
        parents = sorted(m for m in topology.neighbors(k) if m < k)
        if len(parents) != 2:
            raise InvalidTopology(
                f"Чвор {k} нема тачно 2 суседа мањег индекса (нађено {len(parents)})."
            )
        order.append(Step(target=k, parents=(parents[0], parents[1])))
    return order


def check_fixed_b_ancestor(topology: Topology, order: list[Step]) -> None:
    """Провера 6 (ново 01.09., docs/NALAZ_31_08.md НАЛАЗ 1): чвор 1 (FIXED_B) мора бити
    предак трагача.

    Ако чвор 1 нигде не учествује у предачком стаблу трагача, механизам има само један
    ослонац (чвор 0) — круто тело које се врти око чвора 0 и по конструкцији описује
    ТАЧНУ кружницу, не праву путању: за задат угао crank-а, сваки следећи предак се
    одређује пресеком кругова фиксних полупречника око већ одређених чворова, па ротација
    целог склопа за Δ чува сва растојања и даје конгруентну конфигурацију. Ово је хард
    одбијање, не казна — кружна јединка није лоше решење него механизам који по
    конструкцији не може ништа осим круга (апсорбујући локални оптимум, мерено НАЛАЗ 1).

    Прима већ израчунат `order` из `validate()` — solving order се не рачуна двапут на
    најтоплијој путањи (сваки позив `fitness.evaluate`).
    """
    parents_of = {step.target: step.parents for step in order}
    stack = [tracer(topology.n_nodes)]
    seen = set(stack)
    while stack:
        node = stack.pop()
        if node == FIXED_B:
            return
        for parent in parents_of.get(node, ()):
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    raise InvalidTopology(
        "Чвор 1 (FIXED_B) није предак трагача — механизам има само један ослонац "
        "(чвор 0) и по конструкцији описује кружницу, не праву путању."
    )


def validate(topology: Topology) -> list[Step]:
    """Пуна валидација; враћа solving order или подиже `InvalidTopology` (README 1.4).

    Провера 5 (положај tracer-a, BASELINE_SPEC §2): solving order се мора завршавати
    чвором `n−1` — ниједан чвор не сме висити на tracer-у (README 1.3).

    Провера 6 (`check_fixed_b_ancestor`, ново 01.09.): чвор 1 мора бити предак трагача,
    иначе је механизам структурно осуђен на кружницу (docs/NALAZ_31_08.md НАЛАЗ 1).
    """
    check_structure(topology)
    check_degrees(topology)
    check_crank_support(topology)
    order = solving_order(topology)
    if not order or order[-1].target != tracer(topology.n_nodes):
        raise InvalidTopology("Tracer мора бити последњи решен чвор у solving order-у.")
    check_fixed_b_ancestor(topology, order)
    return order


def degrees_of_freedom(topology: Topology) -> int:
    """DOF = 2(n−2) − j (README 1.4, исправка 30.08.). Само дијагностика, не улази у валидацију."""
    return 2 * (topology.n_nodes - 2) - len(topology.edges)
