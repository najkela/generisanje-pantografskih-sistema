"""Спецификација валидације топологије (README 1.4, BASELINE_SPEC §0, §2)."""

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
    """Ниједан floating чвор нема тачно 2 позната суседа → BFS не напредује (README 1.4).

    Чворови 3 и 4 узајамно зависе (ивица (3,4)): сваки има само по једног познатог
    суседа ({1} односно {2}) док год други није решен. Ивица (0,1) уклоњена — не постоји.
    """
    t = Topology(n_nodes=5, edges=[(0, 2), (3, 4), (1, 3), (2, 4)])
    with pytest.raises(InvalidTopology):
        solving_order(t)


def test_valid_mechanism_has_one_dof(fourbar):
    assert degrees_of_freedom(fourbar) == 1


def test_tracer_resolved_early_fails():
    """Провера 5 (README 1.4, BASELINE_SPEC §2): tracer мора бити ПОСЛЕДЊИ решен чвор.

    n=5, tracer=4. Чвор 4 (tracer) има оба ослонца међу почетно познатим чворовима
    ({1,2}) па се решава у првом BFS пролазу; чвор 3 виси на tracer-у (ослонци {1,4})
    и решава се тек после њега. Solving order завршава чвором 3, не чвором 4 → неважеће,
    иако BFS обиђе све чворове (j = 5 = 2·5−5, DOF формула је задовољена).
    """
    t = Topology(n_nodes=5, edges=[(0, 2), (1, 4), (2, 4), (3, 4), (1, 3)])
    with pytest.raises(InvalidTopology):
        validate(t)


def test_tie_break_picks_smallest_index():
    """Детерминистички tie-break (README 1.2.1, BASELINE_SPEC §1.2): при више кандидата са
    тачно 2 позната суседа у истом BFS кораку, бира се увек најмањи индекс чвора.

    n=6, tracer=5. Чворови 3 и 4 постају решиви истовремено у првом пролазу (оба имају
    тачно 2 ослонца међу {0,1,2}); solving order мора кренути чвором 3, никад чвором 4,
    и то поновљиво при сваком позиву (j = 7 = 2·6−5).
    """
    t = Topology(
        n_nodes=6,
        edges=[(0, 2), (1, 3), (2, 3), (0, 4), (1, 4), (3, 5), (4, 5)],
    )
    order_first = solving_order(t)
    order_again = solving_order(t)
    assert [s.target for s in order_first] == [3, 4, 5]
    assert [s.target for s in order_again] == [3, 4, 5]
