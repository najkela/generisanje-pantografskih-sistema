"""Спецификација тополошких оператора: add_node (начин А, Б), delete_node (BASELINE_SPEC §4),
reconnect_node (режим фиксног n, DECISIONS §26.3)."""

import numpy as np
import pytest

from pantograph import operators
from pantograph.genome import Topology, role, to_sequence
from pantograph.operators import (
    _reconnect_change_anchor,
    _reconnect_flip_sign,
    add_node,
    delete_node,
    prune_dead_nodes,
    random_initial_genome,
    reconnect_node,
)
from pantograph.validation import InvalidTopology, degrees_of_freedom, solving_order, validate

# `sixbar` fixture (шест чворова, без мртвог терета) живи у tests/conftest.py — дели га и
# tests/test_fiksno_n.py (DECISIONS §26.3).


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


def test_prune_dead_nodes_removes_unused_branch(fourbar):
    """Мртав терет (начин Б) нестаје; предачко стабло tracer-a (§, „Мртав терет") остаје
    нетакнуто — исте координате, иста улога, DOF инваријанта и даље важи."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    rng = np.random.default_rng(0)
    with_dead_weight, coords_with_dead_weight = add_node(fourbar, coords, rng, mode="B")
    # начин Б помера стари tracer на нову последњу позицију (BASELINE_SPEC §4.2) —
    # координате читамо ОТУДА, не са старог индекса (сад заузетог мртвим чвором).
    old_tracer_coords = coords_with_dead_weight[with_dead_weight.n_nodes - 1].copy()

    pruned_topology, pruned_coords = prune_dead_nodes(with_dead_weight, coords_with_dead_weight)

    assert pruned_topology.n_nodes == fourbar.n_nodes  # мртви чвор нестао
    assert degrees_of_freedom(pruned_topology) == 1
    validate(pruned_topology)
    new_tracer = pruned_topology.n_nodes - 1
    assert role(new_tracer, pruned_topology.n_nodes) == "трагач"
    assert np.allclose(pruned_coords[new_tracer], old_tracer_coords)


def test_prune_dead_nodes_is_idempotent(fourbar):
    """Прунинг генома без мртвог терета не мења ништа."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    pruned_topology, pruned_coords = prune_dead_nodes(fourbar, coords)
    assert pruned_topology.edges == fourbar.edges
    assert pruned_topology.n_nodes == fourbar.n_nodes
    assert np.allclose(pruned_coords, coords)


# --- reconnect_node: подпотез „промена ослонца" (DECISIONS §26.3) ----------------


def test_reconnect_change_anchor_touches_exactly_one_edge_coords_untouched(sixbar):
    """Мења тачно једну ивицу инцидентну чвору k; координате бит-идентичне пре/после; n
    непромењено (DECISIONS §26.3) — за разлику од „обртања знака"."""
    topology, coords = sixbar
    for trial in range(200):
        rng = np.random.default_rng(trial)
        k = int(rng.integers(3, topology.n_nodes))
        result = _reconnect_change_anchor(topology, coords, rng, k)
        assert result is not None  # структурно недостижно за k≥3 на овом фикстуру
        new_topology, new_coords = result

        assert new_topology.n_nodes == topology.n_nodes
        assert np.array_equal(new_coords, coords)  # координате СЕ НЕ дирају

        old_edges = {frozenset(e) for e in topology.edges}
        new_edges = {frozenset(e) for e in new_topology.edges}
        removed = old_edges - new_edges
        added = new_edges - old_edges
        assert len(removed) == 1 and len(added) == 1
        (removed_edge,) = removed
        (added_edge,) = added
        assert k in removed_edge
        assert k in added_edge


def test_reconnect_change_anchor_new_anchor_below_k_and_distinct(sixbar):
    """Нови ослонац је увек `< k` (инваријанта редоследа §16) и никад не поклапа преостали
    ослонац — обоје по конструкцији, не по накнадној провери (DECISIONS §26.3)."""
    topology, coords = sixbar
    for trial in range(200):
        rng = np.random.default_rng(1000 + trial)
        k = int(rng.integers(3, topology.n_nodes))
        new_topology, _ = _reconnect_change_anchor(topology, coords, rng, k)

        order = solving_order(new_topology)  # мора проћи — инваријанта одржана по конструкцији
        new_a, new_b = {step.target: step.parents for step in order}[k]
        assert new_a < k
        assert new_b < k
        assert new_a != new_b


def test_reconnect_change_anchor_preserves_edge_count(sixbar):
    """DOF инваријанта: тачно једна ивица мање, тачно једна више — укупан број непромењен."""
    topology, coords = sixbar
    rng = np.random.default_rng(0)
    k = int(rng.integers(3, topology.n_nodes))
    new_topology, _ = _reconnect_change_anchor(topology, coords, rng, k)
    assert len(new_topology.edges) == len(topology.edges)


# --- reconnect_node: подпотез „обртање знака" (DECISIONS §26.3) ------------------


def test_reconnect_flip_sign_touches_only_target_gene_sign(sixbar):
    """Мења искључиво `s` циљаног гена — ослонци и `ρ` СВИХ гена (укључујући циљани) остају
    нетакнути, за разлику од подпотеза „промена ослонца" (DECISIONS §26.3). Координате низводно
    од `k` МОГУ се променити (физичко огледање), гени ипак остају исти скуп вредности."""
    topology, coords = sixbar
    parent_seq = to_sequence(topology, coords)
    seen_success = False
    for trial in range(200):
        rng = np.random.default_rng(2000 + trial)
        k = int(rng.integers(3, topology.n_nodes))
        result = _reconnect_flip_sign(topology, coords, rng, k)
        if result is None:
            continue  # геометријски пад дозвољен (circuit defect при round-trip-у)
        seen_success = True
        new_topology, new_coords = result
        assert new_topology.n_nodes == topology.n_nodes
        child_seq = to_sequence(new_topology, new_coords)
        idx = k - 3
        for i, (pg, cg) in enumerate(zip(parent_seq.genes, child_seq.genes)):
            assert cg.a == pg.a
            assert cg.b == pg.b
            assert cg.rho_a == pytest.approx(pg.rho_a)
            assert cg.rho_b == pytest.approx(pg.rho_b)
            if i == idx:
                assert cg.s == -pg.s
            else:
                assert cg.s == pg.s
    assert seen_success  # бар један покушај мора успети на овом фикстуру


# --- reconnect_node: оба подпотеза — валидност резултата -------------------------


def test_reconnect_result_validates_or_fails_cleanly(sixbar):
    """Резултат који прође реконструкцију пролази `validate()` или пада искључиво на
    `InvalidTopology` (нпр. чвор 1 престане да буде предак трагача) — никад друга врста
    изузетка (DECISIONS §26.3)."""
    topology, coords = sixbar
    for trial in range(200):
        rng = np.random.default_rng(3000 + trial)
        result = reconnect_node(topology, coords, rng, p_change_anchor=0.70)
        if result is None:
            continue
        new_topology, _ = result
        try:
            validate(new_topology)
        except InvalidTopology:
            pass  # прихватљив исход — circuit-defect типа провере, не пад програма


# --- reconnect_node: дистрибуција подпотеза (детерминистички, БЕЗ статистике) ----


class _FixedUniformRNG:
    """Минималан РНГ „патрљак" за детерминистичко тестирање гранања унутар `reconnect_node`
    (DECISIONS §26.3) — `uniform()` увек враћа задату вредност, `integers()`/`choice()`
    делегирају правом генератору. Statистички тест са толеранцијом би био повремено-црвен;
    ово уместо тога тврди тачну грану."""

    def __init__(self, real_rng: np.random.Generator, uniform_value: float) -> None:
        self._real = real_rng
        self._uniform_value = uniform_value

    def integers(self, *args, **kwargs):
        return self._real.integers(*args, **kwargs)

    def uniform(self, *args, **kwargs):
        return self._uniform_value

    def choice(self, *args, **kwargs):
        return self._real.choice(*args, **kwargs)


def test_reconnect_node_dispatches_change_anchor_below_threshold(monkeypatch, sixbar):
    """`rng.uniform() < p_change_anchor` → подпотез „промена ослонца" (DECISIONS §26.3).
    Монкипатч на саме подпотезе, не статистика — детерминистичка тврдња гране."""
    topology, coords = sixbar
    calls: list[str] = []
    monkeypatch.setattr(
        operators, "_reconnect_change_anchor",
        lambda t, c, r, k: calls.append("anchor") or (t, c),
    )
    monkeypatch.setattr(
        operators, "_reconnect_flip_sign",
        lambda t, c, r, k: calls.append("sign") or (t, c),
    )
    rng = _FixedUniformRNG(np.random.default_rng(0), uniform_value=0.0)

    reconnect_node(topology, coords, rng, p_change_anchor=0.70)

    assert calls == ["anchor"]


def test_reconnect_node_dispatches_flip_sign_above_threshold(monkeypatch, sixbar):
    """`rng.uniform() >= p_change_anchor` → подпотез „обртање знака" (DECISIONS §26.3)."""
    topology, coords = sixbar
    calls: list[str] = []
    monkeypatch.setattr(
        operators, "_reconnect_change_anchor",
        lambda t, c, r, k: calls.append("anchor") or (t, c),
    )
    monkeypatch.setattr(
        operators, "_reconnect_flip_sign",
        lambda t, c, r, k: calls.append("sign") or (t, c),
    )
    rng = _FixedUniformRNG(np.random.default_rng(0), uniform_value=0.99)

    reconnect_node(topology, coords, rng, p_change_anchor=0.70)

    assert calls == ["sign"]
