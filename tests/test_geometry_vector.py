"""Спецификација пресликавања секвенца ↔ вектор геометрије за bilevel (DECISIONS §22)."""

import numpy as np
from scipy.spatial import cKDTree

from pantograph.curve import TargetCurve
from pantograph.fitness import evaluate as fitness_evaluate
from pantograph.genome import Genome, from_sequence, to_sequence
from pantograph.geometry_vector import active_positions, evaluate_vector, skeleton_of, to_sequence_from_x, to_x
from pantograph.operators import add_node, random_initial_genome
from pantograph.simulator import CallCounter


def _frozen_rho(seq, active):
    """Помоћна функција теста: `ρ` свих позиција ВАН `active`, за тачну реконструкцију."""
    active_set = set(active)
    frozen = {}
    for i, gene in enumerate(seq.genes):
        k = 3 + i
        if k not in active_set:
            frozen[k] = (gene.rho_a, gene.rho_b)
    return frozen


def _random_genome_without_dead_weight(n_nodes: int, seed: int):
    """Насумичан геном без мртвог терета — понавља до скелета код ког је свака позиција
    предак трагача (потребно тесту димензионалности, БЕЗ ослањања на унутрашњост
    `random_initial_genome`, само на јавни интерфејс)."""
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


def _circle_target() -> TargetCurve:
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    return TargetCurve(points=points, tree=cKDTree(points))


def test_round_trip_to_x_and_back():
    """to_x → to_sequence_from_x → from_sequence враћа исте координате на 1e-12."""
    rng = np.random.default_rng(3)
    result = None
    while result is None:
        result = random_initial_genome(8, rng)
    topology, coords = result

    seq = to_sequence(topology, coords)
    skeleton = skeleton_of(seq)
    active = active_positions(skeleton)
    frozen = _frozen_rho(seq, active)

    x = to_x(seq, active)
    seq2 = to_sequence_from_x(x, skeleton, frozen, active)
    result2 = from_sequence(seq2)

    assert result2 is not None
    _topology2, coords2 = result2
    assert np.allclose(coords2, coords, atol=1e-12)


def test_dimension_is_2n_minus_2_without_dead_weight():
    """Димензија вектора је `2n−2` кад нема мртвог терета (DECISIONS §22)."""
    n_nodes = 7
    topology, coords = _random_genome_without_dead_weight(n_nodes, seed=11)
    seq = to_sequence(topology, coords)
    skeleton = skeleton_of(seq)
    active = active_positions(skeleton)

    x = to_x(seq, active)

    assert len(active) == n_nodes - 3
    assert x.shape == (2 * n_nodes - 2,)


def test_dead_node_from_add_node_mode_b_is_excluded_and_path_unchanged():
    """Геном са мртвим чвором (add_node начин Б): димензија је за 2 мања, а `evaluate_vector`
    даје исти број као `fitness.evaluate` над оригиналним геномом — мртав чвор не мења
    путању (DECISIONS §22, А2)."""
    original_topology, original_coords = _random_genome_without_dead_weight(6, seed=5)
    target = _circle_target()

    original_genome = Genome(topology=original_topology, coords=original_coords)
    original_score = fitness_evaluate(original_genome, target, n=90)

    rng = np.random.default_rng(7)
    mutated = None
    while mutated is None:
        mutated = add_node(original_topology, original_coords, rng, mode="B")
    mutated_topology, mutated_coords = mutated

    seq = to_sequence(mutated_topology, mutated_coords)
    skeleton = skeleton_of(seq)
    active = active_positions(skeleton)
    frozen = _frozen_rho(seq, active)
    x = to_x(seq, active)

    assert len(x) == 2 * original_topology.n_nodes - 2  # за 2 мање него без мртвог терета

    counter = CallCounter()
    mutated_score = evaluate_vector(x, skeleton, frozen, active, target, n=90, counter=counter)

    assert mutated_score == original_score
    assert counter.calls == 1


def test_evaluate_vector_counts_exactly_one_call_per_invocation():
    """`k` позива `evaluate_vector` увек даје `counter.calls == k`, и за валидне и за
    намерно поломљене векторе (DECISIONS §22, А5)."""
    topology, coords = _random_genome_without_dead_weight(6, seed=2)
    seq = to_sequence(topology, coords)
    skeleton = skeleton_of(seq)
    active = active_positions(skeleton)
    frozen = _frozen_rho(seq, active)
    x_valid = to_x(seq, active)

    x_broken = x_valid.copy()
    x_broken[4:] = 1e-6  # rho занемарљиво мали — пресека нема, from_sequence враћа None

    target = _circle_target()
    counter = CallCounter()

    for i, x in enumerate([x_valid, x_broken, x_valid, x_broken, x_broken], start=1):
        evaluate_vector(x, skeleton, frozen, active, target, n=90, counter=counter)
        assert counter.calls == i
