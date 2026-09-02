"""Спецификација Chamfer растојања (README 1.6)."""

import dataclasses

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pantograph.config import DEFAULT_CONFIG
from pantograph.curve import TargetCurve, apply_similarity_transform, center_and_radius, normalize, place_on_target
from pantograph.fitness import PENALTY, chamfer, evaluate
from pantograph.genome import Genome, Topology
from pantograph.simulator import simulate
from pantograph.validation import solving_order


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


def _gromazan_genome() -> Genome:
    """Фербар геном чији однос largest_link/path_radius ≈ 7058 — тиња crank (0.01), огромни
    coupler/rocker (~95–100). Решив под подразумеваним прагом угла преноса (10°): `simulate`
    враћа путању, не `None` — тестира се искључиво граница гломазности, не праг угла."""
    topology = Topology(n_nodes=4, edges=[(0, 2), (1, 3), (2, 3)])
    coords = np.array([[0.0, 0.0], [100.0, 0.0], [0.01, 0.0], [95.0, 5.0]])
    return Genome(topology=topology, coords=coords)


def test_gromazan_mehanizam_dobija_penalty(circle):
    """Механизам преко границе гломазности (DECISIONS §17, ревизија) добија `PENALTY` — тврда
    граница уведена ПОСЛЕ пуне скалне инваријантности, јер без ње претрага одлази у гломазна
    решења (мерено 6.73× и 26.78× на буџету од 12 000 позива)."""
    target = TargetCurve(points=circle, tree=cKDTree(circle))
    genome = _gromazan_genome()
    assert evaluate(genome, target, 90) == PENALTY


def test_povecana_granica_gromaznosti_dozvoljava_isti_genom(circle):
    """Исти геном, граница подигнута преко његовог односа → коначан Chamfer, не казна —
    потврђује да казну изазива баш граница гломазности, не нешто друго (нпр. праг угла
    преноса или дегенерисан радијус)."""
    target = TargetCurve(points=circle, tree=cKDTree(circle))
    genome = _gromazan_genome()
    config = dataclasses.replace(DEFAULT_CONFIG, max_link_to_radius_ratio=1e9)
    score = evaluate(genome, target, 90, config=config)
    assert np.isfinite(score)
    assert score < PENALTY


def test_place_on_target_matches_evaluate_exactly(ellipse_curve_file):
    """Затворено поравнање (DECISIONS §17, ревизија) даје сирову Chamfer вредност ТАЧНО
    једнаку `fitness.evaluate` — циљ мора бити `ellipse.txt`, НЕ круг: `resample`
    (узорковање по дужини лука, не по параметру) НИЈЕ идентитет за елипсу (помера тачке до
    `3.4e-06`), за разлику од круга (идентитет до `2.6e-14`) — тест на кругу не би имао
    снагу. Обе стране поређења морају бити над ИСТИМ скупом тачака: `evaluate` (одлука 2б)
    пореди против `target.at_resolution(n)`, не против пуног циља — поређење сирове Chamfer
    против пуног `target` мери нешто друго (мерено: разлика `8.2e-07` против пуног циља,
    наспрам `1.7e-16` против `at_resolution(720)`)."""
    topology = Topology(n_nodes=4, edges=[(0, 2), (1, 3), (2, 3)])
    coords = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 0.0], [3.0, 2.0]])
    genome = Genome(topology=topology, coords=coords)
    order = solving_order(topology)
    target = ellipse_curve_file

    expected = evaluate(genome, target, 720)

    path = simulate(topology, coords, order, 720)
    tx, ty, angle, scale = place_on_target(path)
    new_coords = apply_similarity_transform(coords, tx, ty, angle, scale)
    new_path = simulate(topology, new_coords, order, 720)
    actual = chamfer(new_path, target.at_resolution(720))  # НЕ против пуног target-а

    assert abs(actual - expected) < 1e-12
