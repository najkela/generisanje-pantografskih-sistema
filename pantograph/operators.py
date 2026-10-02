"""Иницијализација и мутациони оператори (README 2.3, 2.4, 4.1; BASELINE_SPEC §4).

Улоге чворова се изводе из позиције у секвенци склапања (fixed=0,1; crank=2; tracer=
последња позиција) — мутација их никад не додељује директно (README 1.3, измењено 30.08.).
Невалидна мутација остаје у популацији са казном — мутација се не понавља до валидне.

При променљивом `n` оператора има само два, оба над чворовима (`add_node`, `delete_node`);
ивичних оператора нема — ивице настају и нестају искључиво заједно са чвором (BASELINE_SPEC
§4). У режиму фиксног `n` (DECISIONS §26.3) `add_node`/`delete_node` постају недостижни и
једини тополошки потез је `reconnect_node` — он мења ивице (пар ослонаца или знак гране)
без промене броја чворова.

Уз то, „ивичних оператора нема" из претходног пасуса важи само за промену БРОЈА чворова —
`reconnect_node` јесте ивични оператор, само не мења `n`.
"""

import dataclasses

import numpy as np

from .genome import (
    CRANK,
    FIXED_A,
    FIXED_B,
    Genome,
    Sequence,
    Topology,
    from_sequence,
    link_lengths,
    to_sequence,
    tracer,
)
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
        p_plus, p_minus, _h = pair
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
    """Начин А — допиши на крај; нови чвор постаје tracer (BASELINE_SPEC §4.1).

    Одржава инваријанту „чвор 1 је предак трагача" (docs/NALAZ_31_08.md НАЛАЗ 1, ново
    01.09.): обавезан ослонац новог чвора је стари tracer, чије предачко стабло (ако је
    улазни геном био валидан) већ садржи чвор 1 — нови tracer то наслеђује директно.
    """
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
    """Начин Б — убаци непосредно пре tracer-a; неутрална мутација (BASELINE_SPEC §4.2).

    Одржава инваријанту „чвор 1 је предак трагача" (docs/NALAZ_31_08.md НАЛАЗ 1, ново
    01.09.): tracer остаје ИСТИ чвор — само му се индекс помера релабелингом (n−1 → n),
    ослонци и ивице му се не дирају — предачко стабло је структурно идентично пре и после
    потеза.
    """
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
    p_plus, p_minus, _h = pair
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


def delete_node_with_target(
    topology: Topology, coords: np.ndarray, rng: np.random.Generator
) -> tuple[Topology, np.ndarray, int]:
    """Као `delete_node`, али ВРАЋА и обрисану позицију `u` — потребно bilevel-у
    (`pantograph/bilevel.py::_abs_map_delete_node`, DECISIONS §22, В3) да изгради мапу
    димензија родитељ→дете за warm-start. `delete_node` је тањи омотач око ове функције;
    ово је чиста ЕКСТРАКЦИЈА, понашање и потрошња `rng`-а су НЕПРОМЕЊЕНИ.

    Брише чвор са позиције бране равномерно из {3, ..., n-1}; захтева n >= 5.

    Зависници обрисаног чвора преспајају се на преостали ослонац (BASELINE_SPEC §4.3);
    сви остали чворови задржавају позицију у θ=0 — мења се кретање, не почетни облик.
    Ако је обрисан последњи чвор, нови последњи аутоматски постаје tracer (README 1.3),
    јер тада нема зависника па реиндексирање само скраћује низ.

    НЕ одржава инваријанту „чвор 1 је предак трагача" (docs/NALAZ_31_08.md НАЛАЗ 1, ново
    01.09.): преспајање бира ЈЕДАН од два ослонца обрисаног чвора насумично — ако је чвор
    1 био предак искључиво преко изгубљене гране, tracer остаје без њега. НЕ поправља се,
    исти третман као circuit defect (BASELINE_SPEC §5, „без поправке") — таква јединка
    постаје неважећа тек при `validate()` унутар `fitness.evaluate`, добија казну.
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
    return new_topology, new_coords, u


def delete_node(
    topology: Topology, coords: np.ndarray, rng: np.random.Generator
) -> tuple[Topology, np.ndarray]:
    """Брише чвор са позиције бране равномерно из {3, ..., n-1}; захтева n >= 5.

    Тањи омотач око `delete_node_with_target` — потпис и понашање непромењени од 30.08.
    (baseline и даље не зна ништа о обрисаној позицији, само bilevel-у треба).
    """
    new_topology, new_coords, _u = delete_node_with_target(topology, coords, rng)
    return new_topology, new_coords


# --- оператор преповезивања (режим фиксног n, DECISIONS §26.3) -------------------


def _reconnect_change_anchor(
    topology: Topology, coords: np.ndarray, rng: np.random.Generator, k: int
) -> tuple[Topology, np.ndarray] | None:
    """Подпотез „промена ослонца" (DECISIONS §26.3) — директно над ивицама, координате се
    НЕ дирају. Намерно НЕ иде преко `Sequence`/`from_sequence` са задржаним `ρ`: `ρ_a`,
    `ρ_b` су односи према размаку ослонаца `D_k` у θ=0, а нов пар ослонаца значи нов `D_k`.
    Задржан `ρ` би за нови размак бацио чвор `k` на друго место и повукао низводне
    зависнике — то није ситан потез (преседан: `delete_node_with_target` ради исто, никад
    не дира `ρ`, само преспаја ивице). `ρ` и знак `s` се сами пресрачунавају при следећој
    канонизацији (`genome.to_sequence`) — овде их нема у чему ни да се пишу.

    Ослонци чвора `k` (његова два суседа мањег индекса, инваријанта §16) читају се преко
    `solving_order`, исти образац као `delete_node_with_target`. Нови ослонац је равномерно
    из `{0,...,k-1} \\ {a,b}` — увек `< k` конструкцијом (инваријанта редоследа одржана),
    увек различит од преосталог ослонца (`a_k ≠ b_k` одржано).

    За `k = 3` избор није насумичан него принудан: `range(3) \\ {a,b}` има тачно један
    елемент. Ако су ослонци чвора 3 баш `{0,1}` (оба фиксна), потез га форсирано пребацује
    на пар који укључује crank (2) — исти ризик „оба ослонца фиксна" постоји већ данас преко
    `add_node` начина Б; овде постаје системски јер је при фиксном `n` ово једини тополошки
    потез (в. `docs/IZVESTAJ_FIKSNO_N.md`, мерено по `k`).

    Враћа `None` само у случају празног скупа кандидата — структурно недостижно за `k≥3`
    (`range(k)` за `k≥3` има бар 3 елемента, минус 2 заузета остаје бар 1), задржано као
    изричита одбрана која пада гласно, не тихо. Никад не пада на circuit defect у тренутку
    потеза (нема `from_sequence` реконструкције) — резултат ипак може касније бити невалидан
    (нпр. чвор 1 престане да буде предак трагача, или се не може канонизовати при следећем
    `to_sequence` позиву), исти третман као и код `delete_node` (README 2.3).
    """
    try:
        order = solving_order(topology)
    except InvalidTopology:
        return None
    parents_of = {step.target: step.parents for step in order}
    a, b = parents_of[k]

    change_a = bool(rng.integers(0, 2))
    old_anchor = a if change_a else b

    candidates = [c for c in range(k) if c not in (a, b)]
    if not candidates:
        return None

    new_anchor = int(rng.choice(candidates))
    new_edges = [(x, y) for x, y in topology.edges if {x, y} != {old_anchor, k}]
    new_edges.append((new_anchor, k))
    new_topology = Topology(n_nodes=topology.n_nodes, edges=new_edges)
    return new_topology, coords.copy()


def _reconnect_flip_sign(
    topology: Topology, coords: np.ndarray, rng: np.random.Generator, k: int
) -> tuple[Topology, np.ndarray] | None:
    """Подпотез „обртање знака" (DECISIONS §26.3) — преко секвенце склапања, намерно
    другачије имплементиран од „промена ослонца". Знак `s` није самостално представљен у
    `Topology`/`coords` — изводи се из геометрије тек при канонизацији (`to_sequence`), па је
    round-trip преко `Sequence` једини начин да се овај подпотез уопште изрази. За разлику
    од „промена ослонца", овај потез физички помера чвор `k` (огледа га преко праве `a→b`)
    и реконструише све његове зависнике од те тачке надаље — зато МОЖЕ пасти на circuit
    defect (`from_sequence` врати `None`), што „промена ослонца" не може у тренутку потеза.

    Ослонци (a, b) и `ρ_a`, `ρ_b` остају нетакнути — мења се само `s_k → -s_k`.
    """
    try:
        seq = to_sequence(topology, coords)
    except InvalidTopology:
        return None
    idx = k - 3
    new_genes = list(seq.genes)
    new_genes[idx] = dataclasses.replace(new_genes[idx], s=-new_genes[idx].s)
    new_seq = Sequence(core=seq.core.copy(), genes=new_genes)
    return from_sequence(new_seq)


def reconnect_node(
    topology: Topology,
    coords: np.ndarray,
    rng: np.random.Generator,
    p_change_anchor: float,
) -> tuple[Topology, np.ndarray] | None:
    """Оператор преповезивања — једини тополошки потез у режиму фиксног `n` (DECISIONS
    §26.3, README 2.3 допуна). Бира један чвор `k ∈ {3,...,n-1}` равномерно (заједничко за
    оба подпотеза), па подпотез „промена ослонца" (вероватноћа `p_change_anchor`,
    подразумевано `Config.p_reconnect_anchor = 0.70`) или „обртање знака" (комплемент).
    `n` се никад не мења; улоге (fixed 0,1, crank 2, tracer n-1) остају какве јесу.

    Подела вероватноће прати величину категоричког простора кроз који потез креће: за
    чвор `k` постоји `C(k,2)` избора пара ослонаца наспрам свега 2 избора знака (в.
    `Config.p_reconnect_anchor` за пун коментар) — не „крупноћу" потеза, обртање знака је
    геометријски крупнији потез (физички помера чвор и низводни ланац), промена ослонца
    ситнији (чува облик у θ=0 непромењен осим за сам чвор `k`).

    Враћа `None` само кад сам подпотез пријави дегенерисан/недостижан случај — третира се
    као свака друга невалидна мутација (README 2.3, „Остало"): без понављања.
    """
    n = topology.n_nodes
    k = int(rng.integers(3, n))
    if rng.uniform() < p_change_anchor:
        return _reconnect_change_anchor(topology, coords, rng, k)
    return _reconnect_flip_sign(topology, coords, rng, k)


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


# --- мртав терет (закључано 30.08., README 2.3, BASELINE_SPEC §4.4) --------------


def prune_dead_nodes(topology: Topology, coords: np.ndarray) -> tuple[Topology, np.ndarray]:
    """Уклања чворове ван предачког стабла tracer-a (README 2.3, „Мртав терет").

    Позива се САМО над коначним резултатом, за приказ и извештај — никад унутар петље
    претраге (симулатор за фитнес већ решава само претке tracer-a преко `solving_order`,
    мртве гране не коштају ништа у времену; ово је чисто козметичко скраћивање генома).

    Језгро `{0,1,2}` се увек задржава експлицитно — crank-ова ивица (0,2) није „solving"
    зависност (crank/fixed никад нису мете у `parents_of`), па је обилазак предака преко
    tracer-a сам по себи не би нужно обухватио. Реиндексирање је преко `sorted(keep)`;
    tracer увек испадне последњи јер је већ имао највећи индекс у целом геному (исто
    запажање које чини проверу 5 сувишном, в. DECISIONS §16).
    """
    order = solving_order(topology)
    parents_of = {step.target: step.parents for step in order}
    t = tracer(topology.n_nodes)

    keep = {FIXED_A, FIXED_B, CRANK, t}
    stack = [t]
    while stack:
        node = stack.pop()
        for parent in parents_of.get(node, ()):
            if parent not in keep:
                keep.add(parent)
                stack.append(parent)

    kept_sorted = sorted(keep)
    old_to_new = {old: new for new, old in enumerate(kept_sorted)}

    new_edges = [
        (old_to_new[a], old_to_new[b])
        for a, b in topology.edges
        if a in keep and b in keep
    ]
    new_topology = Topology(n_nodes=len(kept_sorted), edges=new_edges)
    new_coords = coords[kept_sorted]
    return new_topology, new_coords
