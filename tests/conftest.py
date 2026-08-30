"""Заједнички fixtures. Сваки тест овде се пише као спецификација README-а."""

import numpy as np
import pytest

from pantograph.genome import Topology


@pytest.fixture
def fourbar() -> Topology:
    """Најмањи валидан механизам: 2 fixed, crank, tracer (README 1.3).

    Ивица (0,1) не постоји — чворови 0 и 1 су оба закуцана за подлогу, па је веза
    међу њима сама подлога, не полуга (README 1.4, BASELINE_SPEC §0). Три ивице,
    `j = 2n - 5 = 3` за `n = 4`.
    """
    return Topology(n_nodes=4, edges=[(0, 2), (1, 3), (2, 3)])


@pytest.fixture
def circle() -> np.ndarray:
    """Аналитичка тест крива нивоа 1 (README 3.4)."""
    θ = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    return np.column_stack([np.cos(θ), np.sin(θ)])


@pytest.fixture
def ellipse() -> np.ndarray:
    """Аналитичка тест крива нивоа 1 (README 3.4)."""
    θ = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    return np.column_stack([3.0 * np.cos(θ), 1.5 * np.sin(θ)])
