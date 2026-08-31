"""Спецификација секвенце склапања: канонски запис јединке (README 1.2.1, BASELINE_SPEC §1)."""

import numpy as np

from pantograph.genome import from_sequence, to_sequence


def _genes_as_arrays(genes):
    """Помоћна функција: гени → низови по пољу, ради поређења без ослањања на float `==`."""
    return (
        np.array([g.a for g in genes]),
        np.array([g.b for g in genes]),
        np.array([g.rho_a for g in genes]),
        np.array([g.rho_b for g in genes]),
        np.array([g.s for g in genes]),
    )


def test_round_trip_returns_same_geometry(fourbar):
    """to_sequence → from_sequence мора вратити исту геометрију (README 1.2.1)."""
    coords = np.array([[0.0, 0.0], [1.0, 0.3], [0.2, 1.0], [1.1, 0.9]])
    seq = to_sequence(fourbar, coords)
    topology, restored = from_sequence(seq)

    assert np.allclose(restored, coords)
    # fourbar је већ у канонском поретку (solving order = [3], ослонци {1,2}) —
    # реконструисане ивице морају се поклопити тачно, без потребе за релабелингом.
    assert topology.n_nodes == fourbar.n_nodes
    assert topology.edges == fourbar.edges


def test_canonicalization_is_deterministic(fourbar):
    """Канонизација исте јединке је увек иста (README 1.2.1, једнозначност)."""
    coords = np.array([[0.0, 0.0], [1.0, 0.3], [0.2, 1.0], [1.1, 0.9]])
    seq_a = to_sequence(fourbar, coords)
    seq_b = to_sequence(fourbar, coords)

    assert np.allclose(seq_a.core, seq_b.core)
    a1, b1, rho_a1, rho_b1, s1 = _genes_as_arrays(seq_a.genes)
    a2, b2, rho_a2, rho_b2, s2 = _genes_as_arrays(seq_b.genes)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2) and np.array_equal(s1, s2)
    assert np.allclose(rho_a1, rho_a2) and np.allclose(rho_b1, rho_b2)


def test_tail_independent_of_core_scale(fourbar):
    """Реп секвенце (ρ, s) не зависи од скале језгра — само `core` се скалира (README 1.2.1)."""
    coords = np.array([[0.0, 0.0], [1.0, 0.3], [0.2, 1.0], [1.1, 0.9]])
    seq_small = to_sequence(fourbar, coords)
    seq_big = to_sequence(fourbar, coords * 3.0)

    assert np.allclose(seq_big.core, seq_small.core * 3.0)
    a1, b1, rho_a1, rho_b1, s1 = _genes_as_arrays(seq_small.genes)
    a2, b2, rho_a2, rho_b2, s2 = _genes_as_arrays(seq_big.genes)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2) and np.array_equal(s1, s2)
    assert np.allclose(rho_a1, rho_a2) and np.allclose(rho_b1, rho_b2)
