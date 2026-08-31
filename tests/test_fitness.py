"""Спецификација Chamfer растојања (README 1.6)."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pantograph.curve import TargetCurve
from pantograph.fitness import PENALTY, chamfer


def test_identical_curve_scores_zero(circle):
    target = TargetCurve(points=circle, tree=cKDTree(circle))
    assert chamfer(circle, target) == pytest.approx(0.0, abs=1e-12)


def test_degenerate_solution_is_penalized(circle):
    """Све тачке сведене на једну — симетричност мора то да казни (README 1.6)."""
    target = TargetCurve(points=circle, tree=cKDTree(circle))
    collapsed = np.tile(circle[0], (720, 1))
    assert chamfer(collapsed, target) > 0.1


def test_invariant_to_traversal_direction(circle):
    target = TargetCurve(points=circle, tree=cKDTree(circle))
    assert chamfer(circle, target) == pytest.approx(chamfer(circle[::-1], target))


def test_penalty_is_finite():
    """Никад inf и никад nan — ЦМА-ЕС мора остати стабилан (README 1.6)."""
    assert np.isfinite(PENALTY)
