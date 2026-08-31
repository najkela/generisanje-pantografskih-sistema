"""Baseline: чист ГА над монолитним геномом — истовремена оптимизација (README 2.2).

Baseline мора бити best-effort, не намерно ослабљен — поређење H1 губи смисао
ако baseline нема праведну шансу.

Генерацијска шема из оригиналног предлога: 20% елите пролази непромењено;
преосталих 80% настаје укрштањем — 30% деца два родитеља из горњих 20%,
30% деца из мешовитог пара (горњих 20% × средњих 20–80%), 20% деца из средњих 20–80%.
Мутацији су подложне само новонастале јединке, не и пренета елита.
"""

import numpy as np

from .config import Config, DEFAULT_CONFIG
from .curve import TargetCurve
from .genome import Genome, Sequence, from_sequence, to_sequence
from .operators import add_node, delete_node, mutate_coords
from .validation import InvalidTopology


def _copy_genome(genome: Genome) -> Genome:
    """Дубока копија — исти разлог као `Topology.copy` (родитељ се не сме мењати на месту)."""
    return Genome(topology=genome.topology.copy(), coords=genome.coords.copy())


def select(population: list[Genome], scores: np.ndarray) -> list[Genome]:
    """Рангирање по фитнесу — мање је боље, Chamfer растојање (README 1.6).

    Само рангира (сортира); издвајање елите (горњих 20%) и средњег слоја (20–80%) ради
    позивалац (`evolve_generation`) индексирањем сортиране листе (BASELINE_SPEC §6).
    """
    order = np.argsort(scores)
    return [population[i] for i in order]


def crossover(parent_a: Genome, parent_b: Genome, rng: np.random.Generator) -> Genome | None:
    """Укрштање два монолитна генома — рез секвенце склапања (README 2.2, BASELINE_SPEC §5).

    Дете = језгро од `parent_a` + гени `g_3..g_{c-1}` од `parent_a` + гени `g_c..g_{n_B-1}`
    од `parent_b`; `n_дете = n_B`. Гранични случај `min(n_A,n_B) <= 4` (нема легалног реза)
    се сам своди на `c=3` — дете постаје копија `parent_b` са `parent_a`-овим језгром,
    исто као што спецификација тражи, без посебног рачвања кода.

    Враћа `None` ако реконструкција падне на circuit defect (§5, „без поправке") — тада
    јединка остаје неважећа и добија казну при евалуацији фитнеса (README 1.4).
    """
    try:
        seq_a = to_sequence(parent_a.topology, parent_a.coords)
        seq_b = to_sequence(parent_b.topology, parent_b.coords)
    except InvalidTopology:
        # родитељ је сам по себи неважећи (преживео селекцију само ако је цела
        # популација лоша) — дете се не може извести, третирамо као circuit defect.
        return None

    n_b = parent_b.topology.n_nodes
    cut = int(rng.integers(3, min(parent_a.topology.n_nodes, n_b)))

    child_genes = seq_a.genes[: cut - 3] + seq_b.genes[cut - 3 : n_b - 3]
    child_seq = Sequence(core=seq_a.core.copy(), genes=child_genes)

    result = from_sequence(child_seq)
    if result is None:
        return None
    topology, coords = result
    return Genome(topology=topology, coords=coords)


def _choose_topology_operator(n_nodes: int, config: Config, rng: np.random.Generator) -> str:
    """Бира `'add_A' | 'add_B' | 'delete'` по табели 7 (BASELINE_SPEC §7), форсирано на границама."""
    if n_nodes >= config.n_max:
        return "delete"
    if n_nodes <= config.n_min:
        return "add_A"
    r = rng.uniform()
    if r < config.p_add_node_a:
        return "add_A"
    if r < config.p_add_node_a + config.p_add_node_b:
        return "add_B"
    return "delete"


def mutate(
    genome: Genome,
    p_topo: float,
    p_coord: float,
    k: float,
    rng: np.random.Generator,
    config: Config = DEFAULT_CONFIG,
) -> Genome | None:
    """Једна мутација монолитног генома: независни новчићи за топологију и координате
    (одлука 09.08., BASELINE_SPEC §6.1).

    Враћа `None` само кад сам оператор пријави дегенерисан случај (нпр. `add_node`
    начин Б са поклопљеним ослонцима) — то се третира као невалидна мутација, БЕЗ
    понављања (README 2.3, „Остало"); резултујућа топологија иначе може бити структурно
    неважећа (нпр. изолован чвор 1 после `delete_node`) и то се хвата тек при евалуацији
    фитнеса, не овде.
    """
    topology, coords = genome.topology, genome.coords

    if rng.uniform() < p_topo:
        operator = _choose_topology_operator(topology.n_nodes, config, rng)
        if operator == "add_A":
            result = add_node(topology, coords, rng, mode="A")
        elif operator == "add_B":
            result = add_node(topology, coords, rng, mode="B")
        else:
            result = delete_node(topology, coords, rng)
        if result is None:
            return None
        topology, coords = result

    if rng.uniform() < p_coord:
        coords = mutate_coords(topology, coords, k, config.gene_wise_prob, rng)

    return Genome(topology=topology, coords=coords)


def _crossover_child(
    pool_a: list[Genome],
    pool_b: list[Genome],
    rng_select: np.random.Generator,
    rng_cross: np.random.Generator,
    max_attempts: int = 5,
) -> Genome:
    """Дете из насумичног пара (pool_a, pool_b); ако рез падне на circuit defect, пробамо
    нов пар/рез (§5 „без поправке" забрањује крпљење ИСТОГ покушаја, не нов покушај) —
    крајњи излаз (изузетно редак) је копија последњег `parent_b`, да генерација задржи
    тачну величину.
    """
    parent_b = pool_b[int(rng_select.integers(len(pool_b)))]
    for _ in range(max_attempts):
        parent_a = pool_a[int(rng_select.integers(len(pool_a)))]
        parent_b = pool_b[int(rng_select.integers(len(pool_b)))]
        child = crossover(parent_a, parent_b, rng_cross)
        if child is not None:
            return child
    return _copy_genome(parent_b)


def evolve_generation(
    population: list[Genome],
    scores: np.ndarray,
    rng_select: np.random.Generator,
    rng_cross: np.random.Generator,
    rng_mut: np.random.Generator,
    k: float,
    config: Config = DEFAULT_CONFIG,
) -> list[Genome]:
    """Једна генерација: 20% елита непромењено + 30/30/20 укрштање, мутација само на
    новонасталим јединкама (README 2.2, BASELINE_SPEC §6). `scores[i]` мора одговарати
    `population[i]` (мање је боље — Chamfer растојање).

    Ово је НАМЕРНО ужа функција од старог потписа `evolve(target, budget, seed,
    population_size)` (застарео скелет, остаје нетакнут) — четири RNG тока, распоред N,
    плато-детекција и логовање су ван обима 3.3, припадају `experiment.py` (корак 4.1,
    в. BASELINE_SPEC §10 табела).
    """
    p = len(population)
    ranked = select(population, scores)

    n_elite = round(p * 0.20)
    n_upper_upper = round(p * 0.30)
    n_upper_middle = round(p * 0.30)
    n_middle_middle = p - n_elite - n_upper_upper - n_upper_middle  # остатак покрива заокруживање

    upper_pool = ranked[:n_elite]
    middle_pool = ranked[n_elite : round(p * 0.80)]

    next_generation = [_copy_genome(g) for g in ranked[:n_elite]]

    offspring_specs = (
        (n_upper_upper, upper_pool, upper_pool),
        (n_upper_middle, upper_pool, middle_pool),
        (n_middle_middle, middle_pool, middle_pool),
    )
    for count, pool_a, pool_b in offspring_specs:
        for _ in range(count):
            child = _crossover_child(pool_a, pool_b, rng_select, rng_cross)
            mutated = mutate(child, config.p_topo, config.p_coord, k, rng_mut, config)
            next_generation.append(mutated if mutated is not None else child)

    return next_generation


def evolve(target: TargetCurve, budget, seed: int, population_size: int):
    """Застарео потпис — четири RNG тока, распоред N, плато-детекција и логовање
    (BASELINE_SPEC §9, §10 корак 4.1) припадају `experiment.py`, не овом модулу. Замењено
    са `evolve_generation` (30.08.) + спољашња петља у `experiment.py`. Не користити.
    """
    raise NotImplementedError
