"""Спецификација тополошких оператора: add_node (начин А, Б), delete_node (BASELINE_SPEC §4)."""

import numpy as np

from pantograph.genome import Topology, role
from pantograph.operators import add_node, delete_node, random_initial_genome
from pantograph.validation import degrees_of_freedom, validate


def test_add_node_mode_a_new_node_becomes_tracer(fourbar):
    """Начин А: нови чвор се дописује на крај и постаје tracer; обавезан ослонац је стари
    tracer (README 2.3, BASELINE_SPEC §4.1)."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    rng = np.random.default_rng(0)
    old_tracer = fourbar.n_nodes - 1

    new_topology, new_coords = add_node(fourbar, coords, rng, mode="A")

    assert new_topology.n_nodes == fourbar.n_nodes + 1
    new_tracer = new_topology.n_nodes - 1
    assert role(new_tracer, new_topology.n_nodes) == "трагач"
    assert old_tracer in new_topology.neighbors(new_tracer)
    # DOF остаје 1: j = 2n - 5 после сваке мутације (README 1.4).
    assert len(new_topology.edges) == 2 * new_topology.n_nodes - 5
    validate(new_topology)  # резултат мора остати структурно валидан механизам


def test_add_node_mode_b_does_not_change_fitness(fourbar):
    """Начин Б је неутрална мутација — стари tracer (иста улога, исте координате, исти
    предачки скуп) остаје нетакнут, па Chamfer грешка остаје идентична (BASELINE_SPEC §4.2)."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    rng = np.random.default_rng(0)
    old_tracer_coords = coords[fourbar.n_nodes - 1].copy()

    new_topology, new_coords = add_node(fourbar, coords, rng, mode="B")

    assert new_topology.n_nodes == fourbar.n_nodes + 1
    new_tracer = new_topology.n_nodes - 1
    assert role(new_tracer, new_topology.n_nodes) == "трагач"
    assert np.allclose(new_coords[new_tracer], old_tracer_coords)
    assert len(new_topology.edges) == 2 * new_topology.n_nodes - 5
    validate(new_topology)  # резултат мора остати структурно валидан механизам


def test_delete_node_keeps_dof_and_tracer_at_end():
    """DOF инваријанта после брисања: n' = n-1, j' = j-2 (README 2.3, BASELINE_SPEC §4.3)."""
    t = Topology(n_nodes=5, edges=[(0, 2), (1, 3), (2, 3), (1, 4), (3, 4)])
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    rng = np.random.default_rng(0)

    new_topology, new_coords = delete_node(t, coords, rng)

    assert new_topology.n_nodes == t.n_nodes - 1
    assert len(new_topology.edges) == 2 * new_topology.n_nodes - 5
    tracer = new_topology.n_nodes - 1
    assert role(tracer, new_topology.n_nodes) == "трагач"
    validate(new_topology)  # резултат мора остати структурно валидан механизам


def test_delete_node_replacement_support_always_exists():
    """Замена ослонца `r` увек постоји: `a_u != b_u`, највише један може бити `y_w`
    (BASELINE_SPEC §4.3) — ниједан зависник не сме остати са степеном < 2 или са
    два једнака ослонца после преспајања."""
    t = Topology(n_nodes=5, edges=[(0, 2), (1, 3), (2, 3), (1, 4), (3, 4)])
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    rng = np.random.default_rng(0)

    new_topology, _ = delete_node(t, coords, rng)

    for node in range(3, new_topology.n_nodes):
        neighbors = new_topology.neighbors(node)
        assert node not in neighbors
        assert len(neighbors) >= 2


def _random_initial_genome_retrying(n_nodes, rng, attempts=50):
    """Позивалац сам понавља до валидне (README 2.3, „Остало") — иницијализација није
    гарантовано валидна: чвор 1 понекад остане изолован (BASELINE_SPEC §8, degree провера
    није загарантована конструкцијом)."""
    for _ in range(attempts):
        result = random_initial_genome(n_nodes, rng)
        if result is not None:
            return result
    raise AssertionError(f"{attempts} покушаја без валидне јединке за n={n_nodes}.")


def test_random_initial_genome_is_valid_by_construction():
    """Конструктивна иницијализација је валидна по конструкцији за произвољно n (README 4.1,
    BASELINE_SPEC §8) — solving order се тривијално поклапа с редоследом индекса, до
    повременог `None` (чвор 1 изолован), што позивалац решава поновним позивом."""
    for n_nodes in (4, 5, 6, 8, 12):
        rng = np.random.default_rng(n_nodes)
        topology, coords = _random_initial_genome_retrying(n_nodes, rng)

        assert topology.n_nodes == n_nodes
        assert coords.shape == (n_nodes, 2)
        assert degrees_of_freedom(topology) == 1
        order = validate(topology)
        assert [step.target for step in order] == list(range(3, n_nodes))


def test_random_initial_genome_core_matches_spec_ranges():
    """Језгро мора поштовати опсеге из BASELINE_SPEC §8 (чвор 1, crank растојање)."""
    rng = np.random.default_rng(0)
    topology, coords = _random_initial_genome_retrying(6, rng)

    assert np.allclose(coords[0], [0.0, 0.0])
    assert np.all(np.abs(coords[1]) <= 1.5)
    crank_radius = np.linalg.norm(coords[2] - coords[0])
    assert 0.2 <= crank_radius <= 0.8


def test_random_initial_genome_none_on_isolated_fixed_node():
    """Документује стварну недоследност BASELINE_SPEC §8 (пронађено 30.08.): равномеран
    избор ослонаца не гарантује да ће чвор 1 икад бити изабран, па остаје изолован —
    третира се као невалидна иницијализација, не као бага у конструкцији."""
    rng = np.random.default_rng(6)  # ово конкретно семе даје изолован чвор 1 за n=6
    assert random_initial_genome(6, rng) is None
