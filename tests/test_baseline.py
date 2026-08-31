"""Спецификација baseline ГА: select, crossover, mutate, evolve_generation (README 2.2,
BASELINE_SPEC §5, §6)."""

import numpy as np

from pantograph.baseline import crossover, evolve_generation, mutate, select
from pantograph.config import DEFAULT_CONFIG
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

    next_generation = evolve_generation(
        population, scores, rng_select, rng_cross, rng_mut, k=0.1, config=DEFAULT_CONFIG
    )

    assert len(next_generation) == len(population)


def test_evolve_generation_elite_survives_unchanged():
    population = [_random_genome(n, seed=n) for n in (5, 6, 7, 8, 5, 6, 7, 8, 5, 6)]
    scores = np.array([float(i) for i in range(len(population))])
    best = population[0]  # score 0.0, најмање — најбоље
    rng_select = np.random.default_rng(1)
    rng_cross = np.random.default_rng(2)
    rng_mut = np.random.default_rng(3)

    next_generation = evolve_generation(
        population, scores, rng_select, rng_cross, rng_mut, k=0.1, config=DEFAULT_CONFIG
    )

    assert next_generation[0].topology.edges == best.topology.edges
    assert np.allclose(next_generation[0].coords, best.coords)
    validate(next_generation[0].topology)  # елита остаје валидна
