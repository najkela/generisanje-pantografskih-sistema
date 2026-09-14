"""Baseline: чист ГА над монолитним геномом — истовремена оптимизација (README 2.2).

Baseline мора бити best-effort, не намерно ослабљен — поређење H1 губи смисао
ако baseline нема праведну шансу.

Генерацијска шема из оригиналног предлога: 20% елите пролази непромењено;
преосталих 80% настаје укрштањем — 30% деца два родитеља из горњих 20%,
30% деца из мешовитог пара (горњих 20% × средњих 20–80%), 20% деца из средњих 20–80%.
Мутацији су подложне само новонастале јединке, не и пренета елита.
"""

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from .config import Config, DEFAULT_CONFIG
from .curve import TargetCurve
from .experiment import (
    Budget,
    RunLog,
    curve_file_hash,
    git_commit_hash,
    plateau_detected,
    resolution_schedule,
    spawn_rng_streams,
)
from .fitness import PENALTY, evaluate
from .genome import Genome, Sequence, from_sequence, link_lengths, to_sequence
from .operators import add_node, delete_node, mutate_coords, prune_dead_nodes, random_initial_genome
from .simulator import path_health, simulate_with_transmission_angle
from .validation import InvalidTopology, validate

if TYPE_CHECKING:
    # само за анотацију типа — `evolve` не сме да увезе `progress` на runtime нивоу
    # (козметика 31.08.): алгоритамска путања baseline → experiment не сме да увуче
    # matplotlib преко прогрес-репортера.
    from .progress import ProgressReporter


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
) -> tuple[list[Genome], list[float | None]]:
    """Једна генерација: 20% елита непромењено + 30/30/20 укрштање, мутација само на
    новонасталим јединкама (README 2.2, BASELINE_SPEC §6). `scores[i]` мора одговарати
    `population[i]` (мање је боље — Chamfer растојање).

    Ово је НАМЕРНО ужа функција од `evolve` — четири RNG тока, распоред N, плато-детекција
    и логовање су ван обима 3.3, припадају заједничкој инфраструктури у `experiment.py`
    (корак 4.1); `evolve` их користи и позива ову функцију сваку генерацију.

    Враћа и листу познатих оцена, паралелну повратној популацији (DECISIONS §17, кеш елите):
    оцена родитеља за елитне позиције (непромењене, оцена им остаје тачна), `None` за
    позиције насталe укрштањем и мутацијом. `evolve` ову листу користи да прескочи поновну
    евалуацију елите АКО се `N` није променило од претходне генерације.
    """
    p = len(population)
    ranked = select(population, scores)
    # Сортирана вредност на позицији i је иста без обзира на tie-break међу равним оценама
    # (казна 1e9 је честа) — `ranked_scores[i]` тачно одговара оцени `ranked[i]`.
    ranked_scores = np.sort(scores)

    n_elite = round(p * 0.20)
    n_upper_upper = round(p * 0.30)
    n_upper_middle = round(p * 0.30)
    n_middle_middle = p - n_elite - n_upper_upper - n_upper_middle  # остатак покрива заокруживање

    upper_pool = ranked[:n_elite]
    middle_pool = ranked[n_elite : round(p * 0.80)]

    next_generation = [_copy_genome(g) for g in ranked[:n_elite]]
    known_scores: list[float | None] = list(ranked_scores[:n_elite])

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
            known_scores.append(None)

    return next_generation, known_scores


def initialize_population(config: Config, rng_init: np.random.Generator) -> list[Genome]:
    """Почетна популација (README 4.1, BASELINE_SPEC §8) — `n` равномерно из
    `config.n_init_choices`, понавља `random_initial_genome` до валидне (§8: чвор 1 се
    бира случајно као ослонац па повремено остане изолован — то се третира као невалидна
    иницијализација, исто као невалидна мутација, БЕЗ посебне логике овде).
    """
    population = []
    for _ in range(config.population_size):
        result = None
        while result is None:
            n_nodes = int(rng_init.choice(config.n_init_choices))
            result = random_initial_genome(n_nodes, rng_init)
        topology, coords = result
        population.append(Genome(topology=topology, coords=coords))
    return population


def evolve(
    target: TargetCurve,
    budget: int,
    seed: int,
    population_size: int,
    config: Config = DEFAULT_CONFIG,
    reporter: "ProgressReporter | None" = None,
) -> RunLog:
    """Главна петља baseline ГА (README 3.3, BASELINE_SPEC §9, §10 корак 4.1).

    Троши буџет по 1 позиву симулатора по НОВОЈ јединки по генерацији — елита (горњих 20%)
    која се преноси непромењена не кошта поновну евалуацију АКО се `N` није променило од
    претходне генерације (DECISIONS §17): Chamfer рачунат на различитим `N` вредностима
    није упоредив, па се кеш поништава на свакој промени `N` (`resolution_schedule` подигне
    ниво) и цела популација, укључујући елиту, оцењује изнова. Ово је управо разлог који је
    раније (31.08.) био наведен ПРОТИВ кеширања — сада је решен инвалидацијом кеша на промену
    N, не занемарен. Мерено: 19.9% буџета уштеђено у покретању од 500 генерација
    (docs/NALAZ_01_09_poza.md §6).

    RNG токови (BASELINE_SPEC §9): `rng_init` само за `initialize_population`; `rng_select`,
    `rng_cross`, `rng_mut` се преносе НЕПРОМЕЊЕНИ (исти објекти, стање им расте) кроз све
    генерације у `evolve_generation`.

    `reporter` је опциони чист посматрач (`progress.ProgressReporter`, козметика 31.08.)
    — прима готов `GenerationRecord` и тренутно најбољи геном, ништа не мења у току
    петље. `start()`/`finish()` зове позивалац (`run.py`), не `evolve`.

    Критеријум заустављања на largest `N` (исправљено 01.09., docs/NALAZ_31_08.md
    НАЛАЗ 2): плато се мери ИСКЉУЧИВО над `history` од уласка у `n_schedule[-1]`
    (`max_n_start`), не над целом историјом — иначе прозор пуца одмах, пун старих уноса
    са нижих `N` где Chamfer није упоредива вредност (исти разлог због ког
    `progress.ProgressReporter` ресетује бројач стагнације на промени `N`).
    """
    run_config = dataclasses.replace(config, population_size=population_size, total_budget=budget)
    rng_init, rng_select, rng_cross, rng_mut = spawn_rng_streams(seed)
    budget_tracker = Budget(max_calls=budget)

    log = RunLog(
        method="baseline",
        curve=target.path or "",
        curve_hash=curve_file_hash(target.path) if target.path else "",
        seed=seed,
        git_commit=git_commit_hash(),
        config=dataclasses.asdict(run_config),
    )
    log.grid_step = run_config.grid_step

    population = initialize_population(run_config, rng_init)
    history: list[float] = []
    best_score_so_far = float("inf")
    generation = 0
    # Индекс у `history` од ког важи плато на највећем N (docs/NALAZ_31_08.md НАЛАЗ 2):
    # без овога критеријум заустављања мери плато преко уноса са нижих N, где је крива
    # (рачуната на другачијем N, неупоредива вредност) одавно легла — прозор пуца одмах.
    max_n_start: int | None = None
    # Кеш оцене елите унутар истог N (DECISIONS §17) — `known_scores[i]` је позната оцена
    # `population[i]` (елита пренета непромењена из претходне генерације) или `None`
    # (настало укрштањем/мутацијом, мора се оценити). Поништава се на сваку промену N.
    known_scores: list[float | None] | None = None
    previous_n_curve: int | None = None

    while True:
        n_curve = resolution_schedule(generation, history, run_config)
        if max_n_start is None and n_curve == run_config.n_schedule[-1]:
            max_n_start = len(history)
        if n_curve != previous_n_curve:
            known_scores = None  # промена N → Chamfer није упоредив, цео кеш пада
        scores = np.empty(len(population))
        for i, g in enumerate(population):
            if known_scores is not None and known_scores[i] is not None:
                scores[i] = known_scores[i]
                log.cached_calls += 1   # позив уштеђен кешом елите (§17) — иде у извештај
            else:
                scores[i] = evaluate(g, target, n_curve, counter=budget_tracker.counter, config=run_config)
        previous_n_curve = n_curve
        best_index = int(np.argmin(scores))
        best_score = float(scores[best_index])

        if best_score < best_score_so_far:
            best_score_so_far = best_score
            log.best_genome = _copy_genome(population[best_index])

        invalid_count = int((scores >= PENALTY).sum())
        best_genome = population[best_index]
        try:
            working_topology, _ = prune_dead_nodes(best_genome.topology, best_genome.coords)
            working_nodes = working_topology.n_nodes
        except InvalidTopology:
            working_nodes = 0  # одбрамбено — не треба да се деси за валидну јединку

        # Здравље путање најбоље јединке — ван буџета, `simulate_with_transmission_angle`
        # не троши `budget_tracker.counter` (docs/NALAZ_01_09_ugao_prenosa.md, задатак 3).
        try:
            best_order = validate(best_genome.topology)
            diag = simulate_with_transmission_angle(
                best_genome.topology, best_genome.coords, best_order, n_curve, run_config
            )
        except InvalidTopology:
            diag = None
        if diag is not None:
            best_path, min_angle_deg = diag
            jump_count, loop_closure = path_health(best_path)
            # Гломазност (DECISIONS §17) — највећа полуга / полупречник путање најбоље
            # јединке. Ван буџета, чист дијагностички рачун над већ израчунатим `best_path`;
            # НЕ улази у оцену (Chamfer остаје једина мера квалитета поклапања).
            largest_link = max(link_lengths(best_genome.topology, best_genome.coords).values())
            path_bbox_center = (best_path.min(axis=0) + best_path.max(axis=0)) / 2.0
            path_radius = np.linalg.norm(best_path - path_bbox_center, axis=1).max()
            link_to_radius_ratio = largest_link / path_radius if path_radius > 0 else float("nan")
        else:
            min_angle_deg, jump_count, loop_closure = float("nan"), 0, float("nan")
            link_to_radius_ratio = float("nan")

        record = log.record(
            generation=generation,
            calls_spent=budget_tracker.spent,
            best_fitness=best_score,
            mean_fitness=float(scores.mean()),
            n_curve=n_curve,
            best_n_nodes=best_genome.topology.n_nodes,
            best_so_far=best_score_so_far,
            invalid_count=invalid_count,
            working_nodes=working_nodes,
            min_transmission_angle_deg=min_angle_deg,
            path_jump_count=jump_count,
            path_loop_closure=loop_closure,
            link_to_radius_ratio=link_to_radius_ratio,
        )
        if reporter is not None:
            reporter.update(record, log.best_genome)
        history.append(best_score)
        log.update_grid(budget_tracker.spent, best_score_so_far)

        if budget_tracker.exhausted:
            break
        # `early_stop=False` (режим поређења, §24.2) гаси ПОЗИВАОЦА, не сам механизам —
        # `plateau_detected` остаје нетакнут и даље ради за динамички K и за распоред N.
        if run_config.early_stop and n_curve == run_config.n_schedule[-1] and plateau_detected(
            history[max_n_start:], run_config.plateau_window_stop, run_config.plateau_eps_stop
        ):
            break

        fraction = min(budget_tracker.spent / run_config.total_budget, 1.0)
        k = run_config.k_start * (run_config.k_end / run_config.k_start) ** fraction
        population, known_scores = evolve_generation(
            population, scores, rng_select, rng_cross, rng_mut, k, run_config
        )
        generation += 1

    log.flush_grid(run_config.total_budget)

    if log.best_genome is not None:
        # Скраћивање мртвог терета на самом крају, само једном — не по генерацији
        # (README 2.3, `prune_dead_nodes`). Чворови ван предачког стабла tracer-a не
        # утичу на путању (симулатор их не решава), па скраћивање не мења ни Chamfer
        # ни анимацију осим што уклања вишак ивица/чворова са цртежа.
        try:
            pruned_topology, pruned_coords = prune_dead_nodes(
                log.best_genome.topology, log.best_genome.coords
            )
            log.best_genome = Genome(topology=pruned_topology, coords=pruned_coords)
        except InvalidTopology:
            pass  # одбрамбено — не треба да се деси за већ валидну најбољу јединку

    return log
