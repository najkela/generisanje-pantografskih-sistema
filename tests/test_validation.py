"""Спецификација валидације топологије (README 1.4)."""

import pytest

from pantograph.genome import Topology
from pantograph.validation import (
    InvalidTopology,
    degrees_of_freedom,
    solving_order,
    validate,
)

skeleton = pytest.mark.xfail(raises=NotImplementedError, reason="скелет — још није имплементирано")


@skeleton
def test_valid_mechanism_passes(fourbar):
    order = validate(fourbar)
    assert [s.target for s in order] == [3]


@skeleton
def test_isolated_node_fails():
    # чвор 4 није ни са чим повезан
    t = Topology(n_nodes=5, edges=[(0, 2), (0, 1), (1, 3), (2, 3)])
    with pytest.raises(InvalidTopology):
        validate(t)


@skeleton
def test_crank_must_not_attach_to_node_one():
    t = Topology(n_nodes=4, edges=[(0, 2), (1, 2), (0, 1), (1, 3), (2, 3)])
    with pytest.raises(InvalidTopology):
        validate(t)


@skeleton
def test_cyclic_dependency_fails():
    """Ниједан floating чвор нема тачно 2 позната суседа → BFS не напредује (README 1.4)."""
    t = Topology(n_nodes=5, edges=[(0, 2), (0, 1), (3, 4), (1, 3), (2, 4)])
    with pytest.raises(InvalidTopology):
        solving_order(t)


@skeleton
def test_valid_mechanism_has_one_dof(fourbar):
    assert degrees_of_freedom(fourbar) == 1
