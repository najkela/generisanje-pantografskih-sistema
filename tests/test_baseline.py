"""Спецификација baseline ГА: select, crossover, mutate, evolve_generation, evolve
(README 2.2, BASELINE_SPEC §5, §6, §9)."""

import dataclasses

import numpy as np

from pantograph.baseline import crossover, evolve, evolve_generation, mutate, select
from pantograph.config import DEFAULT_CONFIG
from pantograph.curve import TargetCurve
from pantograph.fitness import evaluate
from pantograph.genome import Genome, Topology
from pantograph.operators import random_initial_genome
from pantograph.validation import degrees_of_freedom, validate


def _genome_from(topology: Topology, coords: np.ndarray) -> Genome:
    return Genome(topology=topology, coords=coords)


def _fourbar_genome() -> Genome:
    topology = Topology(n_nodes=4, edges=[(0, 2), (1, 3), (2, 3)])
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    return _genome_from(topology, coords)


def _random_genome(n_nodes: int, seed: int) -> Genome:
    rng = np.random.default_rng(seed)
    result = None
    while result is None:
        result = random_initial_genome(n_nodes, rng)
    topology, coords = result
    return _genome_from(topology, coords)


def test_select_sorts_ascending_by_score():
    g0, g1, g2 = _fourbar_genome(), _fourbar_genome(), _fourbar_genome()
    ranked = select([g0, g1, g2], np.array([5.0, 1.0, 3.0]))
    assert ranked == [g1, g2, g0]


def test_crossover_child_size_matches_parent_b():
    """n_дете = n_B (README 2.2, BASELINE_SPEC §5)."""
    parent_a = _fourbar_genome()
    parent_b = _random_genome(6, seed=1)
    rng = np.random.default_rng(0)

    child = crossover(parent_a, parent_b, rng)

    assert child is not None
    assert child.topology.n_nodes == parent_b.topology.n_nodes
    assert degrees_of_freedom(child.topology) == 1


def test_crossover_reciprocal_child_matches_parent_a():
    """Реципрочно дете (замена улога родитеља) има n_A (README 2.2, BASELINE_SPEC §5)."""
    parent_a = _fourbar_genome()
    parent_b = _random_genome(6, seed=1)
    rng = np.random.default_rng(0)

    child = crossover(parent_b, parent_a, rng)

    assert child is not None
    assert child.topology.n_nodes == parent_a.topology.n_nodes


def test_crossover_boundary_case_copies_parent_b_with_parent_a_core():
    """min(n_A, n_B) <= 4 → нема легалног реза, дете = копија B са A-овим језгром (§5)."""
    parent_a = _fourbar_genome()
    parent_b = _fourbar_genome()
    parent_b.coords = parent_b.coords + 5.0  # различито језгро да разликујемо A/B
    rng = np.random.default_rng(0)

    child = crossover(parent_a, parent_b, rng)

    assert child is not None
    assert child.topology.n_nodes == 4
    assert np.allclose(child.coords[:3], parent_a.coords[:3])


def test_mutate_with_zero_probabilities_is_identity():
    genome = _fourbar_genome()
    rng = np.random.default_rng(0)

    result = mutate(genome, p_topo=0.0, p_coord=0.0, k=0.1, rng=rng)

    assert result is not None
    assert result.topology.edges == genome.topology.edges
    assert np.allclose(result.coords, genome.coords)


def test_mutate_coord_only_keeps_topology():
    genome = _fourbar_genome()
    rng = np.random.default_rng(0)

    result = mutate(genome, p_topo=0.0, p_coord=1.0, k=0.2, rng=rng)

    assert result is not None
    assert result.topology.edges == genome.topology.edges
    assert np.allclose(result.coords[0], [0.0, 0.0])  # чвор 0 се никад не мења


def test_evolve_generation_preserves_population_size():
    population = [_random_genome(n, seed=n) for n in (5, 6, 7, 8, 5, 6, 7, 8, 5, 6)]
    scores = np.array([float(i) for i in range(len(population))])  # веће i = гори (мање место)
    rng_select = np.random.default_rng(1)
    rng_cross = np.random.default_rng(2)
    rng_mut = np.random.default_rng(3)

    next_generation, known_scores = evolve_generation(
        population, scores, rng_select, rng_cross, rng_mut, k=0.1, config=DEFAULT_CONFIG
    )

    assert len(next_generation) == len(population)
    assert len(known_scores) == len(population)


def test_evolve_generation_elite_survives_unchanged():
    population = [_random_genome(n, seed=n) for n in (5, 6, 7, 8, 5, 6, 7, 8, 5, 6)]
    scores = np.array([float(i) for i in range(len(population))])
    best = population[0]  # score 0.0, најмање — најбоље
    rng_select = np.random.default_rng(1)
    rng_cross = np.random.default_rng(2)
    rng_mut = np.random.default_rng(3)

    next_generation, known_scores = evolve_generation(
        population, scores, rng_select, rng_cross, rng_mut, k=0.1, config=DEFAULT_CONFIG
    )

    assert next_generation[0].topology.edges == best.topology.edges
    assert np.allclose(next_generation[0].coords, best.coords)
    validate(next_generation[0].topology)  # елита остаје валидна

    n_elite = round(len(population) * 0.20)
    # Елитне позиције носе познату оцену (DECISIONS §17, кеш елите) — идентичну сортираним
    # оценама родитеља; остале позиције (укрштање/мутација) су непознате.
    assert list(known_scores[:n_elite]) == list(np.sort(scores)[:n_elite])
    assert all(s is None for s in known_scores[n_elite:])


def test_evolve_caches_elite_scores_within_same_n(circle):
    """Кеш оцене елите унутар истог N (DECISIONS §17): друга генерација троши тачно мање
    позива за број кеширане елите."""
    from scipy.spatial import cKDTree

    target = TargetCurve(points=circle, tree=cKDTree(circle))
    population_size = 20
    # n_schedule са једним нивоом → N се никад не мења, кеш никад не пада (изолује ефекат).
    config = dataclasses.replace(DEFAULT_CONFIG, n_schedule=(90,), population_size=population_size)
    log = evolve(target, budget=200, seed=1, population_size=population_size, config=config)

    assert len(log.records) >= 2
    n_elite = round(population_size * 0.20)
    delta = log.records[1].calls_spent - log.records[0].calls_spent
    assert delta == population_size - n_elite


def test_evolve_generation_known_scores_match_reevaluation(circle):
    """Два узастопна позива `evolve_generation` при непромењеном N: елитне позиције из другог
    позива носе оцену која се бит-подудара са директним поновним `evaluate` над истим
    геномом/N-ом (DECISIONS §17: "оцене непромењене елите су бит-идентичне")."""
    from scipy.spatial import cKDTree

    target = TargetCurve(points=circle, tree=cKDTree(circle))
    population = [_random_genome(n, seed=n) for n in (5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6)]
    n_curve = 90
    rng_select = np.random.default_rng(1)
    rng_cross = np.random.default_rng(2)
    rng_mut = np.random.default_rng(3)

    scores1 = np.array([evaluate(g, target, n_curve) for g in population])
    generation2, known_scores2 = evolve_generation(
        population, scores1, rng_select, rng_cross, rng_mut, k=0.1, config=DEFAULT_CONFIG
    )

    n_elite = round(len(population) * 0.20)
    for i in range(n_elite):
        assert known_scores2[i] is not None
        assert known_scores2[i] == evaluate(generation2[i], target, n_curve)


def test_evolve_runs_to_budget_and_returns_run_log(circle):
    """Интеграциони тест малог обима: цела петља (README 3.3, BASELINE_SPEC §9) — заврши,
    достигне/премаши буџет (кеш елите значи да генерације не троше увек тачно
    `population_size` позива, DECISIONS §17), RunLog садржи записе, коначна грешка коначан
    број."""
    from scipy.spatial import cKDTree

    target = TargetCurve(points=circle, tree=cKDTree(circle))
    log = evolve(target, budget=200, seed=1, population_size=10)

    assert log.method == "baseline"
    assert log.seed == 1
    assert len(log.records) > 0
    assert log.records[-1].calls_spent >= 200  # буџет достигнут или премашен, никад мање
    assert np.isfinite(log.final_error)
    assert log.best_genome is not None
    assert log.error_curve == [(r.calls_spent, r.best_fitness) for r in log.records]
    # Гломазност (DECISIONS §17) — коначна за сваку генерацију кад постоји решива најбоља
    # јединка (circle fixture-у решиве јединке падају брзо преко пуне популације).
    assert all(np.isfinite(r.link_to_radius_ratio) for r in log.records)


def test_evolve_respects_config_max_link_to_radius_ratio(ellipse_curve_file):
    """config се мора стварно проследити до evaluate/simulate (регресија — раније се тихо
    игнорисао, DECISIONS §17): исти seed, различита граница гломазности → различит резултат."""
    config_5 = dataclasses.replace(DEFAULT_CONFIG, max_link_to_radius_ratio=5.0)
    config_10 = dataclasses.replace(DEFAULT_CONFIG, max_link_to_radius_ratio=10.0)

    log_5 = evolve(ellipse_curve_file, budget=600, seed=1, population_size=30, config=config_5)
    log_10 = evolve(ellipse_curve_file, budget=600, seed=1, population_size=30, config=config_10)

    assert log_5.final_error != log_10.final_error
    # Детерминистичка тврдња која гађа прослеђивање директно: број невалидних у НУЛТОЈ
    # генерацији (почетна популација, пре иједног еволутивног корака) зависи ИСКЉУЧИВО од
    # тога да ли је граница стигла до `evaluate` — мерено (budget=600, pop=30, seed=1):
    # 19/30 под границом 5, 15/30 под границом 10.
    assert log_5.records[0].invalid_count > log_10.records[0].invalid_count
