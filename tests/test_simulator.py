"""Спецификација симулатора (README 1.2)."""

import numpy as np

from pantograph.genome import link_lengths
from pantograph.simulator import simulate
from pantograph.validation import solving_order


def test_link_lengths_are_positive(fourbar):
    """Круте полуге: дужина сваког крака је константа изведена из почетне геометрије."""
    coords = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 0.0], [3.0, 2.0]])
    lengths = link_lengths(fourbar, coords)
    assert all(d > 0 for d in lengths.values())


def test_path_has_requested_resolution(fourbar):
    coords = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 0.0], [3.0, 2.0]])
    path = simulate(fourbar, coords, solving_order(fourbar), n=90)
    assert path.shape == (90, 2)


def test_unsolvable_geometry_returns_none(fourbar):
    """Крак предугачак да пресек постоји → None, што фитнес претвара у казну.

    Чвор 3 је колинеаран са чворовима 1 (4,0) и 2 (1,0) — l(1,3)+l(2,3) = 1.5+1.5 = 3.0,
    тачно једнако растојању |1−2| у θ=0 (тангентно, једва решиво). Чим crank напусти
    почетни угао, растојање ослонаца расте изнад 3.0 (до 5.0 на супротној страни круга),
    премашује збир полуга → пресек не постоји (README 1.4, circuit defect).

    Претходна верзија (чвор 3 у (300,200)) није заправо неразрешива — неједнакост
    троугла ограничава |l(1,3)−l(2,3)| на највише l(1,2)=3, па тако удаљена, скоро
    једнака два крака увек имају и превише простора да се секу за сваки crank угао.
    """
    coords = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 0.0], [2.5, 0.0]])
    assert simulate(fourbar, coords, solving_order(fourbar), n=90) is None
