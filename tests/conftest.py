"""Заједнички fixtures. Сваки тест овде се пише као спецификација README-а."""

from pathlib import Path

import numpy as np
import pytest

from pantograph.curve import TargetCurve
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


@pytest.fixture
def sixbar() -> tuple[Topology, np.ndarray]:
    """Механизам са шест чворова, БЕЗ мртвог терета — свака ивица је на предачком стаблу
    trace-a (3→4→5, редом), два „унутрашња" гена (k=3, k=4), да оператор преповезивања
    (DECISIONS §26.3) има шта да бира. Координате намерно нису колинеарне ни по једном
    трojцу (родитељ, родитељ, дете) — `to_sequence` мора успети на овом фикстуру."""
    topology = Topology(
        n_nodes=6,
        edges=[(0, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)],
    )
    coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 2.0], [1.5, 1.5]]
    )
    return topology, coords


@pytest.fixture
def ellipse_curve_file() -> TargetCurve:
    """`TargetCurve` из `data/curves/ellipse.txt` (720 тачака из фајла, не аналитичка).

    Другачије име од `ellipse` (аналитички генерисана) намерно — да се не помешају.
    Треба тестовима којима је важна конкретна `resample`-ова густина/фаза узорковања
    (DECISIONS §17): `resample` (узорковање по дужини лука) НИЈЕ идентитет за ову криву,
    за разлику од `circle`-а, па само она даје таквим тестовима стварну снагу.
    """
    path = Path(__file__).resolve().parent.parent / "data" / "curves" / "ellipse.txt"
    return TargetCurve.from_file(str(path))
