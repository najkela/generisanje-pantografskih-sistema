"""Иницијализација и мутациони оператори (README 2.3, 2.4, 4.1; BASELINE_SPEC §4).

Улоге чворова се изводе из позиције у секвенци склапања (fixed=0,1; crank=2; tracer=
последња позиција) — мутација их никад не додељује директно (README 1.3, измењено 30.08.).
Невалидна мутација остаје у популацији са казном — мутација се не понавља до валидне.

Оператора има само два, оба над чворовима (`add_node`, `delete_node`); ивичних оператора
нема — ивице настају и нестају искључиво заједно са чвором (BASELINE_SPEC §4).
"""

import numpy as np

from .genome import CRANK, FIXED_A, FIXED_B, Genome, Topology, link_lengths, tracer
from .simulator import circle_intersect_pair
from .validation import InvalidTopology, solving_order, validate


def random_initial_topology(n_nodes: int, rng: np.random.Generator) -> Topology:
    """Застарео потпис — топологија се не може градити независно од геометрије, BASELINE_SPEC
    §8 захтева спрегнуту конструкцију (circle-circle пресек по кораку). Замењено са
    `random_initial_genome` (корак 2.2, 30.08.). Не користити у новом коду.
    """
    raise NotImplementedError


def random_initial_coords(n_nodes: int, span: float, rng: np.random.Generator) -> np.ndarray:
    """Застарео потпис — исто као `random_initial_topology`, замењено са `random_initial_genome`."""
    raise NotImplementedError


def random_initial_genome(
    n_nodes: int, rng: np.random.Generator
) -> tuple[Topology, np.ndarray] | None:
    """Конструктивна иницијализација преко секвенце (README 4.1, BASELINE_SPEC §8).

    Језгро: чвор 0 у (0,0); чвор 1 у `U(-1.5,1.5)^2`; crank на растојању `U(0.2,0.8)` од
    чвора 0, под насумичним углом. Затим `n_nodes - 3` пута: два ослонца равномерно међу
    постојећим позицијама, `rho_a, rho_b ~ U(0.7,1.3)`, грана насумична, чвор се дописује
    на крај. Сваки нови чвор се везује за тачно 2 постојећа, па је solving order увек
    тривијалан (поклапа се с редоследом индекса), а tracer (последњи додат) увек последњи
    у њему — провере 3–5 (README 1.4) су тако задовољене конструкцијом.

    Провера 2 (степен) НИЈЕ гарантована: чвор 1 се бира као ослонац сасвим случајно, па по
    несрећи може остати без иједне ивице (изолован, провера 1). Враћа `None` у том случају,
    као и у дегенерисаном случају (два ослонца се поклапају у координатама) — оба се третирају
    као невалидна иницијализација коју позивалац понавља до валидне (README 2.3, „Остало").
    """
    if n_nodes < 4:
        raise ValueError("Механизам мора имати бар 4 чвора (README 1.4).")

    coords = np.zeros((n_nodes, 2))
    coords[FIXED_B] = rng.uniform(-1.5, 1.5, size=2)
    crank_radius = rng.uniform(0.2, 0.8)
    crank_angle = rng.uniform(0.0, 2 * np.pi)
    coords[CRANK] = crank_radius * np.array([np.cos(crank_angle), np.sin(crank_angle)])

    edges: list[tuple[int, int]] = [(FIXED_A, CRANK)]

    for k in range(3, n_nodes):
        a, b = (int(x) for x in rng.choice(k, size=2, replace=False))
        pa, pb = coords[a], coords[b]
        d = float(np.linalg.norm(pb - pa))
        if d == 0.0:
            return None
        rho_a = rng.uniform(0.7, 1.3)
        rho_b = rng.uniform(0.7, 1.3)
        pair = circle_intersect_pair(pa, pb, rho_a * d, rho_b * d)
        if pair is None:
            return None
        p_plus, p_minus = pair
        coords[k] = p_plus if rng.choice([True, False]) else p_minus
        edges.append((a, k))
        edges.append((b, k))

    topology = Topology(n_nodes=n_nodes, edges=edges)
    try:
        validate(topology)
    except InvalidTopology:
        return None
    return topology, coords


# --- тополошки оператори (закључано 30.08., README 2.3, BASELINE_SPEC §4) --------


def _average_link_length(topology: Topology, coords: np.ndarray) -> float:
    """Просечна дужина крака јединке у тренутку позива — скала за `alpha`/`k` (README 2.2)."""
    return float(np.mean(list(link_lengths(topology, coords).values())))


def _add_node_mode_a(
    topology: Topology, coords: np.ndarray, rng: np.random.Generator
) -> tuple[Topology, np.ndarray]:
    """Начин А — допиши на крај; нови чвор постаје tracer (BASELINE_SPEC §4.1)."""
    n = topology.n_nodes
    old_tracer = tracer(n)
    other_positions = [node for node in range(n) if node != old_tracer]
    second_anchor = int(rng.choice(other_positions))

    length_scale = _average_link_length(topology, coords)
    alpha = rng.uniform(0.10, 0.40)
    radius = alpha * length_scale
    angle = rng.uniform(0.0, 2 * np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    new_point = coords[old_tracer] + radius * direction

    new_edges = topology.edges + [(old_tracer, n), (second_anchor, n)]
    new_topology = Topology(n_nodes=n + 1, edges=new_edges)
    new_coords = np.vstack([coords, new_point])
    return new_topology, new_coords


def _add_node_mode_b(
    topology: Topology, coords: np.ndarray, rng: np.random.Generator
) -> tuple[Topology, np.ndarray] | None:
    """Начин Б — убаци непосредно пре tracer-a; неутрална мутација (BASELINE_SPEC §4.2)."""
    n = topology.n_nodes
    old_tracer = tracer(n)
    candidates = list(range(old_tracer))  # позиције пре места убацивања
    a, b = (int(x) for x in rng.choice(candidates, size=2, replace=False))
    pa, pb = coords[a], coords[b]
    d = float(np.linalg.norm(pb - pa))
    if d == 0.0:
        return None

    rho_a = rng.uniform(0.7, 1.3)
    rho_b = rng.uniform(0.7, 1.3)
    pair = circle_intersect_pair(pa, pb, rho_a * d, rho_b * d)
    if pair is None:
        return None
    p_plus, p_minus = pair
    s = int(rng.choice([1, -1]))
    new_point = p_plus if s == 1 else p_minus

    new_node = old_tracer      # ново место, непосредно пре (новог) tracer-a
    new_tracer_index = n       # стари tracer се помера на n

    def relabel(node: int) -> int:
        return new_tracer_index if node == old_tracer else node

    new_edges = [(relabel(x), relabel(y)) for x, y in topology.edges]
    new_edges.append((a, new_node))
    new_edges.append((b, new_node))

    new_coords = np.zeros((n + 1, 2))
    new_coords[:old_tracer] = coords[:old_tracer]
    new_coords[new_node] = new_point
    new_coords[new_tracer_index] = coords[old_tracer]

    new_topology = Topology(n_nodes=n + 1, edges=new_edges)
    return new_topology, new_coords


def add_node(
    topology: Topology,
    coords: np.ndarray,
    rng: np.random.Generator,
    mode: str,
) -> tuple[Topology, np.ndarray] | None:
    """Нови чвор — начин А (допиши на крај, постаје tracer) или начин Б (убаци пре
    tracer-a, неутрална мутација) (BASELINE_SPEC §4.1, §4.2).

    `mode` је `"A"` или `"B"`. Враћа `None` само у изузетном дегенерисаном случају
    начина Б (два ослонца се поклапају у координатама) — то се третира као невалидна
    мутација (README 2.3, „Остало").
    """
    if mode == "A":
        return _add_node_mode_a(topology, coords, rng)
    if mode == "B":
        return _add_node_mode_b(topology, coords, rng)
    raise ValueError(f"Непознат mode: {mode!r} — очекивано 'A' или 'B'.")


def delete_node(
    topology: Topology, coords: np.ndarray, rng: np.random.Generator
) -> tuple[Topology, np.ndarray]:
    """Брише чвор са позиције бране равномерно из {3, ..., n-1}; захтева n >= 5.

    Зависници обрисаног чвора преспајају се на преостали ослонац (BASELINE_SPEC §4.3);
    сви остали чворови задржавају позицију у θ=0 — мења се кретање, не почетни облик.
    Ако је обрисан последњи чвор, нови последњи аутоматски постаје tracer (README 1.3),
    јер тада нема зависника па реиндексирање само скраћује низ.
    """
    n = topology.n_nodes
    if n < 5:
        raise ValueError("delete_node захтева n >= 5 (README 2.3, BASELINE_SPEC §4.3).")

    order = solving_order(topology)
    parents_of = {step.target: step.parents for step in order}
    dependents_of: dict[int, list[int]] = {node: [] for node in range(n)}
    for step in order:
        for parent in step.parents:
            dependents_of[parent].append(step.target)

    u = int(rng.integers(3, n))
    a_u, b_u = parents_of[u]

    kept_edges = [(x, y) for x, y in topology.edges if u not in (x, y)]
    for w in dependents_of[u]:
        y_w = next(p for p in parents_of[w] if p != u)
        choices = [c for c in (a_u, b_u) if c != y_w]
        r = int(rng.choice(choices))
        kept_edges.append((w, r))

    def reindex(node: int) -> int:
        return node - 1 if node > u else node

    new_edges = [(reindex(x), reindex(y)) for x, y in kept_edges]
    new_coords = np.delete(coords, u, axis=0)
    new_topology = Topology(n_nodes=n - 1, edges=new_edges)
    return new_topology, new_coords


# --- координатни оператор (baseline; одлука од 09.08.) ---------------------------


def mutate_coords(
    topology: Topology,
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

    Потпис допуњен параметром `topology` (30.08.) — без ње се не може израчунати
    просечна дужина полуге; исти образац као `mode`-параметар `add_node`-а.
    """
    sigma = k * _average_link_length(topology, coords)
    new_coords = coords.copy()
    for node in range(1, topology.n_nodes):
        if rng.uniform() < per_node_prob:
            new_coords[node] = new_coords[node] + rng.normal(0.0, sigma, size=2)
    return new_coords


def mutate_baseline(
    genome: Genome,
    p_topo: float,
    p_coord: float,
    k: float,
    rng: np.random.Generator,
) -> Genome:
    """Застарео потпис — замењено са `baseline.mutate` (BASELINE_SPEC корак 3.2, 30.08.),
    које користи операторе одавде (`add_node`, `delete_node`, `mutate_coords`) плус
    `_choose_topology_operator` из табеле 7. Не користити у новом коду.
    """
    raise NotImplementedError
