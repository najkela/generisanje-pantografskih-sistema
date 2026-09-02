"""Спецификација учитавања и нормализације циљне криве (README 1.5)."""

import numpy as np

from pantograph.curve import TargetCurve, apply_similarity_transform, center_and_radius, normalize, resample
from pantograph.simulator import simulate
from pantograph.validation import solving_order


def test_bounding_box_center_goes_to_origin(ellipse):
    n = normalize(ellipse + np.array([100.0, -50.0]))
    center = (n.min(axis=0) + n.max(axis=0)) / 2
    assert np.allclose(center, 0.0, atol=1e-12)


def test_max_distance_from_center_is_one(ellipse):
    n = normalize(ellipse)
    assert np.isclose(np.linalg.norm(n, axis=1).max(), 1.0)


def test_normalization_is_scale_invariant(circle):
    assert np.allclose(normalize(circle), normalize(circle * 17.3))


def test_resample_keeps_shape(circle):
    assert resample(circle, 90).shape == (90, 2)


def test_center_and_radius_matches_normalize(ellipse):
    """`center_and_radius` + ручна подела мора бити бит-идентично `normalize` (DECISIONS §17)
    — потврђује да инлајновани рачун у `fitness.evaluate` (уместо позива `normalize`, да се
    bbox/норма не рачунају двапут у најврелијој петљи) није апроксимација него иста формула."""
    points = ellipse + np.array([3.0, -1.0])
    center, radius = center_and_radius(points)
    manual = (points - center) / radius
    assert np.array_equal(manual, normalize(points))


def test_apply_similarity_transform_matches_transformed_simulation(fourbar):
    """Сличносна трансформација координата даје идентично трансформисану путању
    (README 1.5, DECISIONS §17) — провера, одступање испод `1e-9` (мерена вредност у
    NALAZ-у: `1.3e-15`)."""
    coords = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 0.0], [3.0, 2.0]])
    order = solving_order(fourbar)
    path1 = simulate(fourbar, coords, order, n=90)
    assert path1 is not None

    tx, ty, angle, scale = 2.5, -1.5, 0.7, 3.1
    transformed_coords = apply_similarity_transform(coords, tx, ty, angle, scale)
    path2 = simulate(fourbar, transformed_coords, order, n=90)
    assert path2 is not None

    expected = apply_similarity_transform(path1, tx, ty, angle, scale)
    assert np.max(np.abs(path2 - expected)) < 1e-9


def test_target_curve_at_resolution_caches_by_n(circle):
    """`TargetCurve.at_resolution(n)` гради KD-дрво само при првом позиву за то `n`
    (DECISIONS §17) — два узастопна позива враћају ИСТИ објекат, не само једнак."""
    from scipy.spatial import cKDTree

    target = TargetCurve(points=circle, tree=cKDTree(circle))
    first = target.at_resolution(90)
    second = target.at_resolution(90)
    assert first is second
