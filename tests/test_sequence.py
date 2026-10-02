"""Спецификација секвенце склапања: канонски запис јединке (README 1.2.1, BASELINE_SPEC §1)."""

import numpy as np
import pytest

from pantograph.genome import Sequence, SequenceGene, from_sequence, to_sequence
from pantograph.validation import InvalidTopology


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


def test_to_sequence_raises_on_collinear_node():
    """Регресија (01.09.): round-trip `rho = |u−p|/d` па назад `r = rho*d` губи последњи
    бит, па за чвор колинеаран са ослонцима у θ=0 `circle_intersect_pair` врати `None`
    иако је `from_sequence` управо успешно реконструисао ту исту геометрију. Пре
    поправке ово је пуцало на `TypeError: cannot unpack non-iterable NoneType object`
    при распакивању на линији где се бира грана пресека — сада мора дати `InvalidTopology`.

    Конкретне вредности нису насумичне — пронађене претрагом (seed=0, погодак на 7. од
    200000 покушаја облика `core=[[0,0],p1,p2]`, `rho_a+rho_b=1.0`), закуцане овде да
    тест буде детерминистичан.
    """
    p1 = np.array([-1.13854875, -0.08498785])
    p2 = np.array([2.33692701, 2.6042611])
    core = np.array([[0.0, 0.0], p1, p2])
    gene = SequenceGene(a=1, b=2, rho_a=0.3720156770381632, rho_b=0.6279843229618368, s=1)
    seq = Sequence(core=core, genes=[gene])

    result = from_sequence(seq)
    assert result is not None  # реконструкција мора успети — бug је у ДРУГОМ, повратном кораку
    topology, coords = result

    with pytest.raises(InvalidTopology):
        to_sequence(topology, coords)


def test_to_sequence_raises_on_coincident_supports(fourbar):
    """Поклопљени ослонци (`d == 0` у θ=0) морају дати `InvalidTopology`, не тихо дељење
    нулом (`inf`/`nan` у `rho_a`/`rho_b`)."""
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])

    with pytest.raises(InvalidTopology):
        to_sequence(fourbar, coords)
