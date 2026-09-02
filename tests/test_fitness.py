"""Спецификација Chamfer растојања (README 1.6)."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pantograph.curve import TargetCurve, center_and_radius, normalize
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


def test_pose_invariance_translation_and_scale(circle):
    """Круг радијуса 1 на правом месту добија исту Chamfer вредност под транслацијом и
    скалом (DECISIONS §17, docs/NALAZ_01_09_poza.md §4) — фитнес мери искључиво облик."""
    target = TargetCurve(points=circle, tree=cKDTree(circle))

    plain = chamfer(normalize(circle), target)
    shifted = chamfer(normalize(circle + np.array([5.0, -2.0])), target)
    scaled = chamfer(normalize(circle * 3.3), target)
    shifted_and_scaled = chamfer(normalize(circle * 3.3 + np.array([5.0, -2.0])), target)

    assert plain == pytest.approx(shifted, abs=1e-9)
    assert plain == pytest.approx(scaled, abs=1e-9)
    assert plain == pytest.approx(shifted_and_scaled, abs=1e-9)


def test_wrong_shape_is_not_improved_by_normalization(circle, ellipse):
    """Погрешан облик (круг наспрам циљне елипсе) НЕ добија бољу оцену него пре — скална
    инваријантност сужава фитнес, не попушта га (DECISIONS §17)."""
    target = TargetCurve(points=normalize(ellipse), tree=cKDTree(normalize(ellipse)))

    same_shape = chamfer(normalize(ellipse * 2.5 + np.array([1.0, -1.0])), target)
    wrong_shape = chamfer(normalize(circle), target)

    assert same_shape == pytest.approx(0.0, abs=1e-9)
    assert wrong_shape > 0.01


def test_degenerate_path_center_and_radius_signals_zero_radius():
    """Дегенерисана путања (сви чворови исте вредности) → радијус ~0, штити `evaluate` од
    дељења нулом (DECISIONS §17) — `evaluate` враћа `PENALTY` пре позива `normalize`."""
    collapsed = np.tile(np.array([1.23, -4.56]), (720, 1))
    _center, radius = center_and_radius(collapsed)
    assert radius < 1e-9
