"""Наш метод: спољашњи ГА над топологијом + унутрашњи ЦМА-ЕС над геометријом (README 2.3).

Фитнес јединке спољашњег ГА = фитнес најбоље јединке коју унутрашњи ЦМА-ЕС нађе за ту
топологију. Спољашњи ГА по default-у користи само само-мутацију, без crossover-а (H2,
DECISIONS §22 А7). Параметризација вектора геометрије, третман мртвих чворова и живо
ЦМА-ЕС стање по топологији су закључани у DECISIONS §22 (чет-сесија 04.09.) — овај модул
је извршни облик тих одлука.

Огледа облик `baseline.py`: `outer_ga` враћа исти `RunLog`, исту плато-логику и исти
распоред N, да поређење H1 буде фер и да `run.py`/`ProgressReporter`/`plot_error_curve`
раде без измена.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cma
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
from .fitness import PENALTY
from .genome import Genome, Sequence, TopologySkeleton, from_sequence, link_lengths, to_sequence
from .geometry_vector import active_positions, evaluate_vector, skeleton_of, to_sequence_from_x, to_x
from .operators import add_node, delete_node_with_target, prune_dead_nodes, random_initial_genome
from .simulator import path_health, simulate_with_transmission_angle
from .validation import InvalidTopology, validate

if TYPE_CHECKING:
    # исти разлог као у baseline.py: алгоритамска путања не сме да увуче matplotlib.
    from .progress import ProgressReporter


# --- стање записа спољашњег ГА (DECISIONS §22, В1) --------------------------------------


@dataclass
class TopologyRecord:
    """Стање једне топологије кроз генерације спољашњег ГА (DECISIONS §22, А3, В1).

    `skeleton is None` означава мртав запис (невалидна тополошка мутација, А6) — нема
    ЦМА-ЕС, нема геометрију, преживи само као казна до прве селекције. `es` се гради
    ЛЕЊО, при ПРВОМ позиву `inner_cmaes` (В2) — `init_x0`/`init_stds`/`init_sigma0` носе
    warm-start резултат (В3) до тог тренутка. `rng_state` изолује `cma`-ин ослонац на
    legacy `numpy.random` глобално стање (в. `_ask_isolated`) — без овога би резултат
    зависио од редоследа позива над ДРУГИМ записима, не само од сопственог seed-а
    (упозорење из PROMPT_BILEVEL целина Ђ.3).
    """

    skeleton: TopologySkeleton | None
    frozen_rho: dict[int, tuple[float, float]]
    active: list[int]
    init_x0: np.ndarray | None = None
    init_stds: np.ndarray | None = None
    init_sigma0: float = 1.0
    es: "cma.CMAEvolutionStrategy | None" = None
    rng_state: tuple | None = None
    best_x: np.ndarray | None = None
    best_fitness: float = float("inf")
    history: list[float] = field(default_factory=list)
    calls: int = 0
    age: int = 0


def _invalid_record() -> TopologyRecord:
    """Мртав запис — невалидно дете (DECISIONS §22, А6). Позивалац наплаћује тачно један
    позив (`budget.spend()`) пре него што ово врати — овде се бројач не дира."""
    return TopologyRecord(
        skeleton=None, frozen_rho={}, active=[], best_x=None, best_fitness=PENALTY,
    )


def _record_to_genome(record: TopologyRecord) -> Genome | None:
    """Реконструише `Genome` из `record.best_x` — `None` ако запис нема геометрију или
    ако реконструкција падне на circuit defect (изузетно редак: `best_x` већ потиче од
    вектора који је прошао кроз `evaluate_vector`)."""
    if record.skeleton is None or record.best_x is None:
        return None
    seq = to_sequence_from_x(record.best_x, record.skeleton, record.frozen_rho, record.active)
    result = from_sequence(seq)
    if result is None:
        return None
    topology, coords = result
    return Genome(topology=topology, coords=coords)


def _initial_stds(n_genes: int, config: Config) -> np.ndarray:
    """Почетне ширине по координати за СВЕЖ ЦМА-ЕС — језгро + `n_genes` парова `ρ` (В4)."""
    stds = list(config.cma_stds_q1) + list(config.cma_stds_q2) + list(config.cma_stds_gene) * n_genes
    return np.array(stds, dtype=float)


# --- унутрашњи ниво: ЦМА-ЕС над геометријом фиксне топологије (В2) ----------------------


def _construct_cma_es(
    x0: np.ndarray, stds: np.ndarray, sigma0: float, config: Config, seed: int
) -> tuple["cma.CMAEvolutionStrategy", tuple]:
    """Гради `cma.CMAEvolutionStrategy` са опцијама из В2 — `bounds` за `ρ` димензије
    `[config.cma_rho_lower, config.cma_rho_upper]`, језгро неограничено, `verbose: -9`
    (испис искључиво кроз `ProgressReporter`). Изолује `cma`-ин legacy `numpy.random`
    ослонац: конструкција привремено мења глобално стање (опција `seed`), па се чува и
    враћа стање позиваоца, а НОВО стање (после конструкције) се враћа као `rng_state` да
    `_ask_isolated` од њега настави.

    Напомена (нађено при писању тестова, целина Ђ): `ρ` реконструисан из стварне
    геометрије (нпр. после `delete_node` преспајања зависника) повремено испадне ВАН
    `[cma_rho_lower, cma_rho_upper]` — `cma` тада одбија саму конструкцију (`geno()` на
    почетној средини захтева тачку УНУТАР граница, диже `ValueError`). Промпт то не
    покрива изричито; најбезбедније решење без одлагања за одобрење је `clip` почетне
    тачке у декларисане границе пре конструкције — исте границе које смо већ увели, не
    нове, а почетна тачка је свеједно само warm-start процена, не резултат.
    """
    saved = np.random.get_state()
    try:
        n_gene_dims = len(x0) - 4
        lower = [None, None, None, None] + [config.cma_rho_lower] * n_gene_dims
        upper = [None, None, None, None] + [config.cma_rho_upper] * n_gene_dims
        x0_clipped = list(x0[:4]) + [
            float(np.clip(v, config.cma_rho_lower, config.cma_rho_upper)) for v in x0[4:]
        ]
        opts = {
            "CMA_stds": [float(v) for v in stds],
            "bounds": [lower, upper],
            "verbose": -9,
            "verb_log": 0,
            "verb_disp": 0,
            "seed": seed,
        }
        if config.cma_lambda is not None:
            opts["popsize"] = config.cma_lambda
        es = cma.CMAEvolutionStrategy(x0_clipped, sigma0, opts)
        rng_state = np.random.get_state()
    finally:
        np.random.set_state(saved)
    return es, rng_state


def _ask_isolated(record: TopologyRecord) -> list[np.ndarray]:
    """`record.es.ask()` изолован од legacy `numpy.random` глобалног стања — `cma` узорке
    вуче директно из њега (докстринг класе `TopologyRecord`), не из сопственог генератора."""
    saved = np.random.get_state()
    try:
        np.random.set_state(record.rng_state)
        candidates = record.es.ask()
        record.rng_state = np.random.get_state()
    finally:
        np.random.set_state(saved)
    return candidates


def _update_best(record: TopologyRecord, candidates: list[np.ndarray], scores: list[float]) -> None:
    """Ажурира `best_x`/`best_fitness` из СТВАРНО оцењених кандидата (не из `es.result`,
    који уме да заостане за једну итерацију, В2)."""
    best_idx = int(np.argmin(scores))
    if scores[best_idx] < record.best_fitness:
        record.best_fitness = float(scores[best_idx])
        record.best_x = np.asarray(candidates[best_idx], dtype=float).copy()


def inner_cmaes(
    record: TopologyRecord,
    target: TargetCurve,
    n: int,
    budget: Budget,
    config: Config = DEFAULT_CONFIG,
    rng_cma: np.random.Generator = None,
) -> tuple[np.ndarray | None, float]:
    """ЦМА-ЕС над вектором геометрије `x` за фиксну топологију записа (README 2.3, В2).

    Број итерација K је динамички — плато-детекција `experiment.plateau_detected` као
    early stopping, осим ако је `config.fixed_k` постављено (референтне 5/15/40, H3),
    тада плато игнорише и ради се тачно толико итерација. Буџет се проверава ПРЕ сваког
    `ask()`-а; ако попуни усред оцене популације пресуши, итерација се прекида БЕЗ позива
    `tell()`-а (cma не прима непотпуну листу) — већ оцењени кандидати ипак ажурирају
    `best_x`/`best_fitness`, ти позиви су стварно потрошени.

    Враћа `(record.best_x, record.best_fitness)` — исто стање остаје и на `record`.
    """
    if record.skeleton is None:
        return record.best_x, record.best_fitness  # мртав запис (А6) — нема шта да се ради

    if record.es is None:
        seed = int(rng_cma.integers(0, 2**31 - 1))
        record.es, record.rng_state = _construct_cma_es(
            record.init_x0, record.init_stds, record.init_sigma0, config, seed
        )
        if record.best_x is None:
            record.best_x = np.asarray(record.init_x0, dtype=float).copy()

    iterations = 0
    while True:
        if budget.exhausted:
            break
        if config.fixed_k is not None:
            if iterations >= config.fixed_k:
                break
        else:
            if iterations >= config.k_max:
                break
            if iterations > 0 and plateau_detected(record.history, config.plateau_window_k, config.plateau_eps_k):
                break

        candidates = _ask_isolated(record)
        scores: list[float] = []
        for x in candidates:
            if budget.exhausted:
                break
            score = evaluate_vector(
                np.asarray(x, dtype=float), record.skeleton, record.frozen_rho, record.active,
                target, n, budget.counter, config,
            )
            scores.append(score)

        if len(scores) < len(candidates):
            if scores:
                _update_best(record, candidates[: len(scores)], scores)
            break  # непотпуна популација — без tell(), боље прекинута итерација него прекорачен буџет

        record.es.tell(candidates, scores)
        _update_best(record, candidates, scores)
        record.history.append(record.best_fitness)
        record.calls += len(scores)
        iterations += 1

    record.age += iterations
    return record.best_x, record.best_fitness


# --- warm-start (В3) ---------------------------------------------------------------------


def warm_start(
    parent_record: TopologyRecord,
    operator_name: str,
    index_map: dict[int, int | None],
    child_seq: Sequence,
    config: Config = DEFAULT_CONFIG,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Почетна средина/ширина/скала ЦМА-ЕС расподеле детета (DECISIONS §22 А4, В3).

    `x0` је вектор детета изведен из `child_seq` (већ мутиране геометрије коју је
    оператор произвео) — ништа се не интерполира овде. `stds`: језгро (прве 4 димензије)
    се преноси увек 1:1 од родитеља; за сваку осталу димензију детета која има претка у
    родитељу (по `index_map`, в. `_abs_map_add_node_a/_b/_delete_node`) узима се
    родитељева ТРЕНУТНА ширина по тој координати (`es.sigma * es.result.stds`); нове
    димензије добијају `config.cma_sigma0_new`. `sigma0 = 1.0` — цела скала стоји у
    `stds` (В3). `operator_name` је информативан (сав потребан податак носи `index_map`).
    """
    del operator_name  # информативно, В3 — index_map већ носи сву потребну информацију
    child_skeleton = skeleton_of(child_seq)
    child_active = active_positions(child_skeleton)
    x0 = to_x(child_seq, child_active)

    if parent_record.es is not None:
        parent_widths = parent_record.es.sigma * parent_record.es.result.stds
    elif parent_record.init_stds is not None:
        parent_widths = parent_record.init_sigma0 * parent_record.init_stds
    else:
        parent_widths = None  # одбрамбено — не треба да се деси (родитељ увек има бар init_stds)

    stds = np.empty(len(x0))
    stds[0:4] = parent_widths[0:4] if parent_widths is not None else config.cma_sigma0_new

    parent_active_slot = {pos: i for i, pos in enumerate(parent_record.active)}
    for i, pos in enumerate(child_active):
        lo = 4 + 2 * i
        parent_pos = index_map.get(pos)
        if parent_pos is not None and parent_pos in parent_active_slot and parent_widths is not None:
            j = parent_active_slot[parent_pos]
            stds[lo : lo + 2] = parent_widths[4 + 2 * j : 4 + 2 * j + 2]
        else:
            stds[lo : lo + 2] = config.cma_sigma0_new

    return x0, stds, 1.0


def _abs_map_add_node_a(n_parent: int) -> dict[int, int | None]:
    """Пресликавање апсолутних позиција генā родитељ→дете за `add_node` начин А (В3
    табела): гени `0..m-1` се преносе 1:1, последњи (нови) ген нема претка."""
    mapping: dict[int, int | None] = {k: k for k in range(3, n_parent)}
    mapping[n_parent] = None
    return mapping


def _abs_map_add_node_b(n_parent: int) -> dict[int, int | None]:
    """Као горе, за начин Б: гени пре места убацивања 1:1, убачени нема претка, стари
    tracer (један ген, помера се за +1) наслеђује своју стару позицију (В3 табела)."""
    old_tracer = n_parent - 1
    mapping: dict[int, int | None] = {k: k for k in range(3, old_tracer)}
    mapping[old_tracer] = None
    mapping[old_tracer + 1] = old_tracer
    return mapping


def _abs_map_delete_node(n_parent: int, deleted_position: int) -> dict[int, int]:
    """За `delete_node` на позицији `deleted_position`: тај ген отпада, гени после ње
    померени за −1 (В3 табела) — зависници којима је пресрачунат `ρ` и даље задржавају
    родитељеву ширину, то је корак претраге, не вредност."""
    n_child = n_parent - 1
    mapping: dict[int, int] = {k: k for k in range(3, deleted_position)}
    for k in range(deleted_position, n_child):
        mapping[k] = k + 1
    return mapping


def _choose_topology_operator(n_nodes: int, config: Config, rng: np.random.Generator) -> str:
    """Иста расподела и форсирања на границама као у baseline-у (BASELINE_SPEC §7) —
    сопствена копија (не увоз приватне функције из `baseline.py`) да bilevel не зависи
    од унутрашњости другог модула. DECISIONS §22 А7: исти оператори, исте вероватноће."""
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


def _mutate_topology(genome: Genome, config: Config, rng: np.random.Generator):
    """Тополошка само-мутација (add_node А/Б или delete_node) + `index_map` за
    `warm_start` (DECISIONS §22 А7, В3). Враћа `(topology, coords, index_map,
    operator_name)` или `None` ако сам оператор пријави дегенерисан случај (нпр. add_node
    Б са поклопљеним ослонцима) — исти третман као `baseline.mutate` (README 2.3, „Остало")."""
    topology, coords = genome.topology, genome.coords
    n_parent = topology.n_nodes
    operator = _choose_topology_operator(n_parent, config, rng)

    if operator == "add_A":
        result = add_node(topology, coords, rng, mode="A")
        if result is None:
            return None
        new_topology, new_coords = result
        return new_topology, new_coords, _abs_map_add_node_a(n_parent), "add_A"

    if operator == "add_B":
        result = add_node(topology, coords, rng, mode="B")
        if result is None:
            return None
        new_topology, new_coords = result
        return new_topology, new_coords, _abs_map_add_node_b(n_parent), "add_B"

    new_topology, new_coords, u = delete_node_with_target(topology, coords, rng)
    return new_topology, new_coords, _abs_map_delete_node(n_parent, u), "delete"


# --- спољашњи ниво (В5) ------------------------------------------------------------------


def _initialize_population(config: Config, rng_init: np.random.Generator) -> list[TopologyRecord]:
    """Почетна популација топологија (README 4.1) — исти извор као baseline
    (`operators.random_initial_genome`), понавља до валидне И до реконструктибилне
    секвенце (`to_sequence` може пасти на нумерички дегенерисан случај и поред `validate`)."""
    population: list[TopologyRecord] = []
    for _ in range(config.outer_population):
        while True:
            n_nodes = int(rng_init.choice(config.n_init_choices))
            result = random_initial_genome(n_nodes, rng_init)
            if result is None:
                continue
            topology, coords = result
            try:
                seq = to_sequence(topology, coords)
            except InvalidTopology:
                continue
            break

        skeleton = skeleton_of(seq)
        active = active_positions(skeleton)
        active_set = set(active)
        frozen_rho = {
            3 + i: (gene.rho_a, gene.rho_b)
            for i, gene in enumerate(seq.genes)
            if (3 + i) not in active_set
        }
        x0 = to_x(seq, active)
        stds = _initial_stds(len(active), config)
        population.append(
            TopologyRecord(
                skeleton=skeleton, frozen_rho=frozen_rho, active=active,
                init_x0=x0, init_stds=stds, init_sigma0=1.0,
            )
        )
    return population


def _make_child(
    parent_record: TopologyRecord, config: Config, rng_mut: np.random.Generator, budget: Budget
) -> TopologyRecord:
    """Дете спољашњег ГА: тополошка само-мутација родитеља + warm-start (В5 корак 4).

    Невалидно дете (пад на `validation.validate` или circuit defect у `to_sequence`, или
    родитељ без реконструктибилне геометрије) наплаћује тачно један позив и не добија
    ЦМА-ЕС (DECISIONS §22 А6) — умире при следећој селекцији.
    """
    parent_genome = _record_to_genome(parent_record)
    if parent_genome is None:
        budget.spend()
        return _invalid_record()

    mutation = _mutate_topology(parent_genome, config, rng_mut)
    if mutation is None:
        budget.spend()
        return _invalid_record()
    child_topology, child_coords, index_map, operator_name = mutation

    try:
        validate(child_topology)
        child_seq = to_sequence(child_topology, child_coords)
    except InvalidTopology:
        budget.spend()
        return _invalid_record()

    child_skeleton = skeleton_of(child_seq)
    child_active = active_positions(child_skeleton)
    active_set = set(child_active)
    frozen_rho = {
        3 + i: (gene.rho_a, gene.rho_b)
        for i, gene in enumerate(child_seq.genes)
        if (3 + i) not in active_set
    }
    x0, stds, sigma0 = warm_start(parent_record, operator_name, index_map, child_seq, config)

    return TopologyRecord(
        skeleton=child_skeleton, frozen_rho=frozen_rho, active=child_active,
        init_x0=x0, init_stds=stds, init_sigma0=sigma0,
    )


def _next_generation(
    population: list[TopologyRecord],
    scores: np.ndarray,
    config: Config,
    rng_select: np.random.Generator,
    rng_mut: np.random.Generator,
    budget: Budget,
) -> list[TopologyRecord]:
    """Једна генерација спољашњег ГА (В5 кораци 2–5): елита `round(0.20·P)` преживљава СА
    својим живим ЦМА-ЕС објектом (А3 — наставља се, не рестартује); остатак су деца,
    родитељ равномерно из горње половине, дете тополошком само-мутацијом (А7)."""
    p = len(population)
    order = np.argsort(scores)
    ranked = [population[i] for i in order]

    n_elite = round(p * 0.20)
    elite = ranked[:n_elite]
    upper_half = ranked[: max(1, round(p * 0.50))]

    next_population = list(elite)
    for _ in range(p - n_elite):
        parent = upper_half[int(rng_select.integers(len(upper_half)))]
        next_population.append(_make_child(parent, config, rng_mut, budget))
    return next_population


def _generation_diagnostics(
    best_genome: Genome | None, n_curve: int, config: Config
) -> tuple[int, float, int, float, float, int]:
    """Дијагностика над најбољом јединком генерације — исти скуп поља као `baseline.evolve`
    (README 4.3), рачунато ван буџета над већ реконструисаним геномом. Враћа
    `(working_nodes, min_angle_deg, jump_count, loop_closure, link_to_radius_ratio, n_nodes)`."""
    if best_genome is None:
        return 0, float("nan"), 0, float("nan"), float("nan"), 0

    try:
        working_topology, _ = prune_dead_nodes(best_genome.topology, best_genome.coords)
        working_nodes = working_topology.n_nodes
    except InvalidTopology:
        working_nodes = 0

    try:
        best_order = validate(best_genome.topology)
        diag = simulate_with_transmission_angle(
            best_genome.topology, best_genome.coords, best_order, n_curve, config
        )
    except InvalidTopology:
        diag = None

    if diag is None:
        return working_nodes, float("nan"), 0, float("nan"), float("nan"), best_genome.topology.n_nodes

    best_path, min_angle_deg = diag
    jump_count, loop_closure = path_health(best_path)
    largest_link = max(link_lengths(best_genome.topology, best_genome.coords).values())
    path_bbox_center = (best_path.min(axis=0) + best_path.max(axis=0)) / 2.0
    path_radius = np.linalg.norm(best_path - path_bbox_center, axis=1).max()
    link_to_radius_ratio = largest_link / path_radius if path_radius > 0 else float("nan")
    return (
        working_nodes, min_angle_deg, jump_count, loop_closure, link_to_radius_ratio,
        best_genome.topology.n_nodes,
    )


def outer_ga(
    target: TargetCurve,
    budget: int,
    seed: int,
    config: Config = DEFAULT_CONFIG,
    reporter: "ProgressReporter | None" = None,
) -> RunLog:
    """Главна bilevel петља: спољашњи ГА над топологијама + унутрашњи ЦМА-ЕС по топологији
    (README 2.3, DECISIONS §22, В5). Троши стварно потрошен K по евалуацији (README 3.3).

    Огледа `baseline.evolve`: исти облик `RunLog`-а, иста плато-логика, исти распоред N.
    RNG токови: `experiment.spawn_rng_streams(seed, n_streams=5)` — пети ток је `rng_cma`,
    прва четири остају бит-идентична baseline-овим (В5).
    """
    run_config = dataclasses.replace(config, total_budget=budget)
    rng_init, rng_select, _rng_cross, rng_mut, rng_cma = spawn_rng_streams(seed, n_streams=5)
    budget_tracker = Budget(max_calls=budget)

    log = RunLog(
        method="bilevel",
        curve=target.path or "",
        curve_hash=curve_file_hash(target.path) if target.path else "",
        seed=seed,
        git_commit=git_commit_hash(),
        config=dataclasses.asdict(run_config),
    )

    population = _initialize_population(run_config, rng_init)
    history: list[float] = []
    best_score_so_far = float("inf")
    best_genome_overall: Genome | None = None
    generation = 0
    max_n_start: int | None = None
    previous_n_curve: int | None = None

    while True:
        n_curve = resolution_schedule(generation, history, run_config)
        if max_n_start is None and n_curve == run_config.n_schedule[-1]:
            max_n_start = len(history)

        if previous_n_curve is not None and n_curve != previous_n_curve:
            # Промена N — Chamfer није упоредив (§17): преоцени best_x сваког живог записа,
            # испразни му историју (плато-детекција динамичког K креће испочетка на новом N).
            for record in population:
                if record.skeleton is None or record.best_x is None:
                    continue
                record.best_fitness = evaluate_vector(
                    record.best_x, record.skeleton, record.frozen_rho, record.active,
                    target, n_curve, budget_tracker.counter, run_config,
                )
                record.history = []
        previous_n_curve = n_curve

        for record in population:
            if budget_tracker.exhausted:
                break
            if record.skeleton is None:
                continue  # мртав запис (А6) — нема шта да се оптимизује
            inner_cmaes(record, target, n_curve, budget_tracker, run_config, rng_cma)

        scores = np.array([r.best_fitness for r in population])
        best_index = int(np.argmin(scores))
        best_score = float(scores[best_index])

        if best_score < best_score_so_far:
            candidate_genome = _record_to_genome(population[best_index])
            if candidate_genome is not None:
                best_score_so_far = best_score
                best_genome_overall = candidate_genome

        invalid_count = int((scores >= PENALTY).sum())
        best_generation_genome = _record_to_genome(population[best_index])
        working_nodes, min_angle_deg, jump_count, loop_closure, link_to_radius_ratio, best_n_nodes = (
            _generation_diagnostics(best_generation_genome, n_curve, run_config)
        )

        record_entry = log.record(
            generation=generation,
            calls_spent=budget_tracker.spent,
            best_fitness=best_score,
            mean_fitness=float(scores.mean()),
            n_curve=n_curve,
            best_n_nodes=best_n_nodes,
            best_so_far=best_score_so_far,
            invalid_count=invalid_count,
            working_nodes=working_nodes,
            min_transmission_angle_deg=min_angle_deg,
            path_jump_count=jump_count,
            path_loop_closure=loop_closure,
            link_to_radius_ratio=link_to_radius_ratio,
        )
        if reporter is not None:
            reporter.update(record_entry, best_genome_overall)
        history.append(best_score)

        if budget_tracker.exhausted:
            break
        if n_curve == run_config.n_schedule[-1] and plateau_detected(
            history[max_n_start:], run_config.plateau_window_stop, run_config.plateau_eps_stop
        ):
            break

        population = _next_generation(population, scores, run_config, rng_select, rng_mut, budget_tracker)
        generation += 1

    if best_genome_overall is not None:
        try:
            pruned_topology, pruned_coords = prune_dead_nodes(
                best_genome_overall.topology, best_genome_overall.coords
            )
            best_genome_overall = Genome(topology=pruned_topology, coords=pruned_coords)
        except InvalidTopology:
            pass
    log.best_genome = best_genome_overall

    return log
