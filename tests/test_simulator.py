"""Спецификација симулатора (README 1.2)."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from pantograph.config import DEFAULT_CONFIG
from pantograph.curve import TargetCurve
from pantograph.fitness import chamfer
from pantograph.genome import Topology, link_lengths, to_sequence
from pantograph.operators import random_initial_genome
from pantograph.simulator import branch_signs, simulate
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


def test_branch_signs_matches_to_sequence_signs():
    """`branch_signs` мора дати исте знакове као `genome.to_sequence` (DECISIONS §17).

    Solving order је увек `[3, ..., n-1]` (инваријанта редоследа, README 1.2.1), па
    `to_sequence`-ово `position_of[k] == k` — ген на позицији `i` у секвенци тачно одговара
    чвору `i+3`. Насумична валидна геометрија, три различите топологије/семена.
    """
    rng = np.random.default_rng(7)
    for n_nodes in (5, 7, 9):
        result = None
        while result is None:
            result = random_initial_genome(n_nodes, rng)
        topology, coords = result
        order = solving_order(topology)

        signs = branch_signs(topology, coords, order)
        seq = to_sequence(topology, coords)

        for i, gene in enumerate(seq.genes):
            assert signs[i + 3] == gene.s


ELLIPSE_CURVE = Path(__file__).resolve().parent.parent / "data" / "curves" / "ellipse.txt"


def _ellipse_seed3_genome() -> tuple[Topology, np.ndarray]:
    """Геном из results/2026-09-01_183518_baseline_ellipse_seed3/best_genome.json — раније
    прескакао грану (docs/NALAZ_01_09_poza.md §1–3); `results/` није под гитом, координате
    су литерал овде."""
    topology = Topology(
        n_nodes=8,
        edges=[(0, 2), (1, 3), (2, 3), (0, 4), (3, 4), (3, 5), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7)],
    )
    coords = np.array([
        [0.0, 0.0],
        [0.2979040715926017, 0.18140621442943147],
        [0.3680180011636773, 0.31650543586219027],
        [-0.480921242866441, -0.05266511613388175],
        [0.17226451298533316, -0.3933055252019902],
        [0.010924460233994238, -0.7576663704431363],
        [0.07218478542040241, -0.7196671368621385],
        [0.23147779762733287, -0.7893385994762464],
    ])
    return topology, coords


def test_ellipse_seed3_chamfer_is_monotone_and_matches_measurement():
    """Дрифт-тест НИЈЕ написан (разматрано и одбачено, DECISIONS §17): са закуцаном граном
    и без зависности од историје, `positions_at(θ₀)` и `positions_at(θ₀+2π)` дају идентичан
    резултат ПО КОНСТРУКЦИЈИ (cos/sin периодични, конфигурација зависи искључиво од угла и
    вектора знакова) — дрифт би био машинска нула чак и уз потпуно погрешну имплементацију
    гране, дакле не проверава ништа. Дрифт је имао смисла само док је постојала хеуристика
    која носи историју кроз углове; чим она нестане, нестаје и мерна вредност теста.

    Уместо тога, конкретан геном (results/2026-09-01_183518_baseline_ellipse_seed3), сирова
    Chamfer (без нормализације путање, независно од одлуке 2) наспрам циљне елипсе —
    измерене вредности из docs/NALAZ_01_09_poza.md §3.

    Праг угла преноса је ИЗРИЧИТО 5.0°, не подразумевани 10° (одлука §17/5): овај геном има
    минимални угао преноса 5.12° кроз пун обртај, па би под подразумеваним прагом `simulate`
    враћао `None` на сваком N — тест би пао из погрешног разлога (праг из друге, независне
    одлуке), не зато што грана није закуцана. Референтне вредности испод су мерене под 5°.
    Будућа ревизија прага (нпр. на 20°, в. коментар уз `min_transmission_angle_deg`) не сме
    оборити овај тест — он с тим нема везе.

    Циљна крива је ПУНА резолуција (свих 720 тачака из фајла), НЕ `target.at_resolution(n)`
    — референтне вредности су мерене против пуног циља (без подузорковања из одлуке 2б). Ако
    се овде икад дода подузорковање, бројеви се померају и толеранција `1e-3` више не важи.
    """
    topology, coords = _ellipse_seed3_genome()
    order = solving_order(topology)
    config = dataclasses.replace(DEFAULT_CONFIG, min_transmission_angle_deg=5.0)
    target = TargetCurve.from_file(str(ELLIPSE_CURVE))

    scores = {}
    for n in (90, 180, 360, 720):
        path = simulate(topology, coords, order, n, config)
        assert path is not None
        scores[n] = chamfer(path, target)

    # а) N=720 мора бити ≈ 3.81e-02, НЕ 2.27e-02 (вредност коју је давала хеуристика
    # "ближа претходном положају" — ако тест добије то, грана није стварно закуцана).
    assert scores[720] == pytest.approx(3.81e-02, abs=1e-3)

    # б) монотоно опадајући и конвергентан низ — са старом хеуристиком је скакао
    # (7.3e-02 → 8.3e-02 → 3.9e-02 → 2.3e-02, docs/NALAZ_01_09_poza.md §3).
    expected = {90: 4.10e-02, 180: 3.88e-02, 360: 3.83e-02, 720: 3.82e-02}
    for n, exp in expected.items():
        assert scores[n] == pytest.approx(exp, abs=1e-3)
    values = [scores[n] for n in (90, 180, 360, 720)]
    assert all(a >= b for a, b in zip(values, values[1:]))  # монотоно опадајуће


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
