"""Спецификација bilevel метода: `inner_cmaes`, `warm_start`, `outer_ga` (DECISIONS §22,
PROMPT_BILEVEL целина Ђ). Тестови су спецификација, не накнадна провера (CLAUDE.md §7)."""

import dataclasses

import numpy as np
from scipy.spatial import cKDTree

from pantograph import bilevel
from pantograph.config import DEFAULT_CONFIG
from pantograph.curve import TargetCurve
from pantograph.experiment import Budget, RunLog
from pantograph.genome import to_sequence
from pantograph.geometry_vector import active_positions, evaluate_vector, skeleton_of, to_x
from pantograph.operators import random_initial_genome
from pantograph.simulator import CallCounter
from pantograph.validation import solving_order, validate


def _circle_target() -> TargetCurve:
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    return TargetCurve(points=points, tree=cKDTree(points))


def _random_genome_without_dead_weight(n_nodes: int, seed: int):
    """Исти помоћник као у `test_geometry_vector.py` — насумичан геном без мртвог терета,
    само преко јавног интерфејса `random_initial_genome`."""
    rng = np.random.default_rng(seed)
    while True:
        result = random_initial_genome(n_nodes, rng)
        if result is None:
            continue
        topology, coords = result
        seq = to_sequence(topology, coords)
        skeleton = skeleton_of(seq)
        active = active_positions(skeleton)
        if len(active) == n_nodes - 3:
            return topology, coords


def _make_record(topology, coords, config=DEFAULT_CONFIG) -> "bilevel.TopologyRecord":
    """Свеж запис (без ЦМА-ЕС-а — лења конструкција, В2) за дати геном."""
    seq = to_sequence(topology, coords)
    skeleton = skeleton_of(seq)
    active = active_positions(skeleton)
    active_set = set(active)
    frozen_rho = {
        3 + i: (gene.rho_a, gene.rho_b)
        for i, gene in enumerate(seq.genes)
        if (3 + i) not in active_set
    }
    x0 = to_x(seq, active)
    stds = bilevel._initial_stds(len(active), config)
    return bilevel.TopologyRecord(
        skeleton=skeleton, frozen_rho=frozen_rho, active=active,
        init_x0=x0, init_stds=stds, init_sigma0=1.0,
    )


def test_inner_cmaes_strictly_decreases_fitness():
    """`inner_cmaes` на фиксној, ручно склопљеној валидној топологији смањује фитнес после
    20 итерација — не тврди колико, само да је строго мање од почетног (Ђ.1)."""
    target = _circle_target()
    topology, coords = _random_genome_without_dead_weight(6, seed=4)
    record = _make_record(topology, coords)

    counter = CallCounter()
    initial_score = evaluate_vector(
        record.init_x0, record.skeleton, record.frozen_rho, record.active, target, 90, counter
    )

    budget = Budget(max_calls=100_000)
    record.rng = np.random.default_rng(0)
    config = dataclasses.replace(DEFAULT_CONFIG, fixed_k=20)

    bilevel.inner_cmaes(record, target, 90, budget, config)

    assert record.best_fitness < initial_score


def test_inner_cmaes_spent_budget_equals_evaluated_candidates():
    """Потрошени буџет је тачно једнак броју оцењених кандидата, укључујући и оне који
    падну на реконструкцији — овде форсирано малим `max_calls` да пресуши усред популације
    (В2: „прекини усред оцене популације ако пресуши", Ђ.2)."""
    target = _circle_target()
    topology, coords = _random_genome_without_dead_weight(6, seed=9)
    record = _make_record(topology, coords)

    budget = Budget(max_calls=7)  # мање од величине популације ЦМА-ЕС-а по default-у
    record.rng = np.random.default_rng(1)

    bilevel.inner_cmaes(record, target, 90, budget, DEFAULT_CONFIG)

    assert budget.spent == 7


def test_outer_ga_same_seed_gives_bit_identical_run_log():
    """Иста семена → бит-идентичан `RunLog` (два узастопна `outer_ga`-а са истим seed-ом,
    Ђ.3). `cma` вуче узорке директно из `record.rng`, позиционог тока
    `cma_rng(seed, generation, slot)` (DECISIONS §24, В2) — детерминизам не сме да зависи
    од редоследа позива над ДРУГИМ записима, само од сопствене позиције."""
    target = _circle_target()
    config = dataclasses.replace(
        DEFAULT_CONFIG, outer_population=6, total_budget=500, k_max=6, plateau_window_k=3,
    )

    log1 = bilevel.outer_ga(target, budget=500, seed=7, config=config)
    log2 = bilevel.outer_ga(target, budget=500, seed=7, config=config)

    assert log1.final_error == log2.final_error
    assert len(log1.records) == len(log2.records)
    for r1, r2 in zip(log1.records, log2.records):
        assert r1.calls_spent == r2.calls_spent
        assert r1.best_fitness == r2.best_fitness
    assert log1.best_genome is not None and log2.best_genome is not None
    assert np.array_equal(log1.best_genome.coords, log2.best_genome.coords)


def test_n_change_invalidates_cache_and_recharges_population():
    """Промена N поништава кеш: `best_fitness` записа се преоцењује — број позива у ГЕНЕРАЦИЈИ
    промене N није мањи од броја записа у популацији (Ђ.4)."""
    target = _circle_target()
    config = dataclasses.replace(
        DEFAULT_CONFIG,
        outer_population=6, total_budget=3000, k_max=3, plateau_window_k=2,
        n_schedule=(90, 180), plateau_window_grow=2, plateau_eps_grow=0.5,
    )

    log = bilevel.outer_ga(target, budget=3000, seed=2, config=config)

    n_values = [r.n_curve for r in log.records]
    transition = next(i for i in range(1, len(n_values)) if n_values[i] != n_values[i - 1])
    calls_this_generation = log.records[transition].calls_spent - log.records[transition - 1].calls_spent

    assert calls_this_generation >= config.outer_population


def test_operators_preserve_order_invariant_through_200_random_mutations():
    """Инваријанта редоследа (§16) важи за сваки скелет који оператори произведу кроз 200
    насумичних мутација — `solving_order` (језгро провере, DECISIONS §16) не сме да падне
    ни на једном путу. Пуни `validate()` МОЖЕ легитимно пасти из ДРУГИХ разлога (степен,
    ослонац chvora 1 — документовано у `operators.delete_node`), то овде НИЈЕ грешка."""
    rng = np.random.default_rng(123)
    topology, coords = _random_genome_without_dead_weight(6, seed=1)

    successes = 0
    attempts = 0
    while successes < 200:
        attempts += 1
        if attempts > 5000:
            raise AssertionError("превише узастопних дегенерисаних мутација — нешто није у реду")
        from pantograph.genome import Genome

        mutation = bilevel._mutate_topology(Genome(topology=topology, coords=coords), DEFAULT_CONFIG, rng)
        if mutation is None:
            continue
        new_topology, new_coords, _index_map, _operator_name = mutation

        solving_order(new_topology)  # не сме да подигне InvalidTopology (§16)

        if new_topology.n_nodes < 4:
            continue
        topology, coords = new_topology, new_coords
        successes += 1


def test_warm_start_after_add_node_a_has_two_more_dimensions():
    """После `add_node` начина А дете има тачно две димензије више, првих `d` вредности
    средине приближно једнако родитељевом `x*` (закон округљивања при round-trip-у
    геометрије, `atol=1e-9`), последње две ширине једнаке `config.cma_sigma0_new` (Ђ.6)."""
    from pantograph.genome import from_sequence
    from pantograph.geometry_vector import to_sequence_from_x
    from pantograph.operators import add_node

    target = _circle_target()
    topology, coords = _random_genome_without_dead_weight(6, seed=6)
    parent_record = _make_record(topology, coords)

    budget = Budget(max_calls=100_000)
    parent_record.rng = np.random.default_rng(5)
    config = dataclasses.replace(DEFAULT_CONFIG, fixed_k=3)
    bilevel.inner_cmaes(parent_record, target, 90, budget, config)

    parent_seq = to_sequence_from_x(
        parent_record.best_x, parent_record.skeleton, parent_record.frozen_rho, parent_record.active
    )
    parent_topology, parent_coords = from_sequence(parent_seq)

    rng_mut = np.random.default_rng(8)
    result = None
    while result is None:
        result = add_node(parent_topology, parent_coords, rng_mut, mode="A")
    child_topology, child_coords = result
    child_seq = to_sequence(child_topology, child_coords)

    index_map = bilevel._abs_map_add_node_a(parent_topology.n_nodes)
    x0, stds, sigma0 = bilevel.warm_start(parent_record, "add_A", index_map, child_seq, config)

    d_parent = len(parent_record.best_x)
    assert len(x0) == d_parent + 2
    assert sigma0 == 1.0
    assert np.allclose(x0[:d_parent], parent_record.best_x, atol=1e-9)
    assert np.allclose(stds[-2:], config.cma_sigma0_new)


def test_rho_out_of_bounds_becomes_invalid_record_without_clipping():
    """`ρ` намерно ван `[cma_rho_lower, cma_rho_upper]` → запис постаје невалидан:
    буџет +1, `log.rho_out_of_bounds` +1, `record.es is None` — БЕЗ тихог clip-a почетне
    тачке (DECISIONS §24, А1; PROMPT_RNG целина Б)."""
    topology, coords = _random_genome_without_dead_weight(6, seed=4)
    record = _make_record(topology, coords)

    original_x0 = record.init_x0.copy()
    record.init_x0[4] = DEFAULT_CONFIG.cma_rho_upper + 1.0  # намерно ван горње границе

    budget = Budget(max_calls=100_000)
    log = RunLog(method="bilevel", curve="", seed=0)
    target = _circle_target()

    bilevel.inner_cmaes(record, target, 90, budget, DEFAULT_CONFIG, log)

    assert budget.spent == 1
    assert log.rho_out_of_bounds == 1
    assert record.es is None
    assert record.skeleton is None
    assert record.best_fitness == bilevel.PENALTY
    # почетна тачка није тихо clip-ована — намерно постављена вредност остаје нетакнута
    assert record.init_x0[4] == DEFAULT_CONFIG.cma_rho_upper + 1.0
    assert record.init_x0[5:].tolist() == original_x0[5:].tolist()
