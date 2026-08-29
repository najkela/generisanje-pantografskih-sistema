"""Заједнички fixtures. Сваки тест овде се пише као спецификација README-а."""

import numpy as np
import pytest

from pantograph.genome import Topology


@pytest.fixture
def fourbar() -> Topology:
    """Најмањи валидан механизам: 2 fixed, crank, tracer (README 1.3)."""
    return Topology(n_nodes=4, edges=[(0, 2), (0, 1), (1, 3), (2, 3)])


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
