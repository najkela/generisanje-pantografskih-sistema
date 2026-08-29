"""Спецификација симулатора (README 1.2)."""

import numpy as np
import pytest

from pantograph.genome import link_lengths
from pantograph.simulator import simulate

skeleton = pytest.mark.xfail(raises=NotImplementedError, reason="скелет — још није имплементирано")


@skeleton
def test_link_lengths_are_positive(fourbar):
    """Круте полуге: дужина сваког крака је константа изведена из почетне геометрије."""
    coords = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 0.0], [3.0, 2.0]])
    lengths = link_lengths(fourbar, coords)
    assert all(d > 0 for d in lengths.values())


@skeleton
def test_path_has_requested_resolution(fourbar):
    coords = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 0.0], [3.0, 2.0]])
    path = simulate(fourbar, coords, [], n=90)
    assert path.shape == (90, 2)


@skeleton
def test_unsolvable_geometry_returns_none(fourbar):
    """Крак предугачак да пресек постоји → None, што фитнес претвара у казну."""
    coords = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 0.0], [300.0, 200.0]])
    assert simulate(fourbar, coords, [], n=90) is None
