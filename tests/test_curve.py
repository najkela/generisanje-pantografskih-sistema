"""Спецификација учитавања и нормализације циљне криве (README 1.5)."""

import numpy as np

from pantograph.curve import normalize, resample


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
