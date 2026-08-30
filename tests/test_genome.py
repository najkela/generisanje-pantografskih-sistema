"""Спецификација структура генома: топологија, вектор ↔ чворови (README 1.2, 1.3)."""

import numpy as np
import pytest

from pantograph.genome import Topology, from_vector, link_lengths, role, to_vector


def test_neighbors_and_degree(fourbar):
    # ивица (0,1) не постоји (BASELINE_SPEC §0) — чвор 0 је сусед само чвору 2.
    assert fourbar.neighbors(0) == {2}
    assert fourbar.neighbors(3) == {1, 2}
    assert fourbar.degree(0) == 1
    assert fourbar.degree(3) == 2


def test_copy_is_deep(fourbar):
    clone = fourbar.copy()
    clone.edges.append((0, 3))
    assert (0, 3) not in fourbar.edges
    assert clone is not fourbar


def test_role_by_index(fourbar):
    n = fourbar.n_nodes
    assert role(0, n) == "фиксни"
    assert role(1, n) == "фиксни"
    assert role(2, n) == "ручица"
    assert role(n - 1, n) == "трагач"


def test_role_floating_for_middle_nodes():
    # петочлани механизам: чвор 3 је ни fixed ни crank ни tracer (n-1=4)
    assert role(3, n_nodes=5) == "слободни"


def test_vector_dimensionality_is_2n_minus_2(fourbar):
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    vector = to_vector(coords)
    assert vector.shape == (2 * fourbar.n_nodes - 2,)


def test_round_trip_to_vector_from_vector(fourbar):
    coords = np.array([[0.0, 0.0], [1.0, 0.3], [0.2, 1.0], [1.1, 0.9]])
    vector = to_vector(coords)
    restored = from_vector(vector, fourbar.n_nodes)
    assert np.allclose(restored, coords)


def test_from_vector_pins_node_zero_at_origin(fourbar):
    vector = np.array([1.0, 0.3, 0.2, 1.0, 1.1, 0.9])
    coords = from_vector(vector, fourbar.n_nodes)
    assert np.allclose(coords[0], [0.0, 0.0])


def test_link_lengths_from_initial_geometry(fourbar):
    # ивица (0,1) не постоји (BASELINE_SPEC §0) — само три полуге имају дужину.
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    lengths = link_lengths(fourbar, coords)
    assert lengths[(0, 2)] == pytest.approx(1.0)
    assert lengths[(2, 3)] == pytest.approx(1.0)
    assert lengths[(1, 3)] == pytest.approx(1.0)
