"""Спецификација валидације топологије (README 1.4, BASELINE_SPEC §0, §2, §1.2 измењено 31.08.)."""

import pytest

from pantograph.genome import Topology
from pantograph.validation import (
    InvalidTopology,
    degrees_of_freedom,
    solving_order,
    validate,
)


def test_valid_mechanism_passes(fourbar):
    order = validate(fourbar)
    assert [s.target for s in order] == [3]


def test_isolated_node_fails():
    # ивица (0,1) не постоји (BASELINE_SPEC §0) — чвор 4 остаје изолован, ни са чим повезан.
    t = Topology(n_nodes=5, edges=[(0, 2), (1, 3), (2, 3)])
    with pytest.raises(InvalidTopology):
        validate(t)


def test_crank_must_not_attach_to_node_one():
    # ивица (1,2) је права забрана коју проверавамо; ивица (0,1) уклоњена (не постоји).
    t = Topology(n_nodes=4, edges=[(0, 2), (1, 2), (1, 3), (2, 3)])
    with pytest.raises(InvalidTopology):
        validate(t)


def test_cyclic_dependency_fails():
    """Чвор 3 нема тачно 2 суседа мањег индекса (README 1.4, инваријанта редоследа,
    измењено 31.08.) — суседи су му {1,4}, а 4 није мањи од 3.

    Чворови 3 и 4 узајамно зависе (ивица (3,4)): по старом BFS моделу овде ниједан не би
    имао 2 позната суседа; по новом моделу инваријанта пада директно на чвору 3. Исти
    резултат (InvalidTopology), другачији разлог. Ивица (0,1) уклоњена — не постоји.
    """
    t = Topology(n_nodes=5, edges=[(0, 2), (3, 4), (1, 3), (2, 4)])
    with pytest.raises(InvalidTopology):
        solving_order(t)


def test_valid_mechanism_has_one_dof(fourbar):
    assert degrees_of_freedom(fourbar) == 1


def test_solving_order_is_trivial_and_reproducible():
    """Инваријанта редоследа (README 1.2.1, измењено 31.08.): кад важи за све чворове,
    solving order је увек `[3, ..., n-1]` — нема претраге, нема tie-break-а (повучен
    30.08. одлуком, беспредметан уз инваријанту). Раније је овде стајао тест за
    детерминистички tie-break у BFS-у; сад демонстрира исто (поновљивост), без BFS-a.
    """
    t = Topology(
        n_nodes=6,
        edges=[(0, 2), (1, 3), (2, 3), (0, 4), (1, 4), (3, 5), (4, 5)],
    )
    order_first = solving_order(t)
    order_again = solving_order(t)
    assert [s.target for s in order_first] == [3, 4, 5]
    assert [s.target for s in order_again] == [3, 4, 5]


# НАПОМЕНА (31.08.): тест „tracer резолван прерано пада на провери 5" је уклоњен.
# Под новом инваријантом редоследа, tracer (индекс n-1, увек НАЈВЕЋИ у топологији) никад
# не може бити нечији "мањи-индекс" ослонац — ниједан чвор k не може имати n-1 < k. Дакле
# провера 4 сама искључује сваку могућност да нешто виси на tracer-у; не постоји топологија
# која пролази проверу 4 а пада на провери 5. Провера 5 остаје у `validate()` као
# документована, безопасна редундантна провера (BASELINE_SPEC §2), али је више није могуће
# демонстрирати посебним тестом.
