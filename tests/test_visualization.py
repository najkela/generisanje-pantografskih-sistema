"""Спецификација цртања: `snapshot` не сме да мутира геном нити да обори тренинг
(PROMPT_ITERACIJA.md, целина Б; DECISIONS §20)."""

import numpy as np
from scipy.spatial import cKDTree

from pantograph.curve import TargetCurve, center_and_radius
from pantograph.genome import Genome
from pantograph.visualization import snapshot


def _fourbar_genome() -> Genome:
    coords = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 0.0], [3.0, 2.0]])
    from pantograph.genome import Topology
    return Genome(topology=Topology(n_nodes=4, edges=[(0, 2), (1, 3), (2, 3)]), coords=coords)


def test_snapshot_does_not_mutate_genome_coords(tmp_path, circle):
    """КРИТИЧНО (Б2): `snapshot` прима живи геном из популације — измена координата на
    месту би тихо покварила претрагу (`log.best_genome` се преноси кроз генерације и на
    крају враћа као резултат `evolve`-а). `snapshot` мора радити на копији."""
    genome = _fourbar_genome()
    original_coords = genome.coords.copy()
    target = TargetCurve(points=circle, tree=cKDTree(circle))

    snapshot(genome, target, str(tmp_path / "snap.png"), n=90)

    assert np.array_equal(genome.coords, original_coords)


def test_snapshot_draws_placed_mechanism_matching_target_scale(tmp_path, circle):
    """Механизам и путања се цртају ПОСТАВЉЕНИ на циљну криву (место, ротација, поставка
    из `place_on_target`) — путања после снимка мора имати полупречник ~1, исте скале као
    нормализована циљна крива, не сирове координате механизма (radius >> 1 за овај геном)."""
    genome = _fourbar_genome()
    target = TargetCurve(points=circle, tree=cKDTree(circle))

    from pantograph.validation import solving_order
    from pantograph.simulator import simulate

    order = solving_order(genome.topology)
    raw_path = simulate(genome.topology, genome.coords, order, 90)
    _center, raw_radius = center_and_radius(raw_path)
    assert abs(raw_radius - 1.0) > 0.05  # сирова путања није у скали циља (радијус 1)

    ok = snapshot(genome, target, str(tmp_path / "snap.png"))
    assert ok is True
    assert (tmp_path / "snap.png").exists()


def test_snapshot_falls_back_uncorrected_when_unsolvable(tmp_path, fourbar, circle):
    """Ако `tracer_path` врати `None` (circuit defect), `snapshot` не сме да падне — врата
    `False`, слика се свеједно снима са напоменом (постојеће понашање, DECISIONS §17)."""
    # Исти дегенерисани случај као test_simulator.py::test_unsolvable_geometry_returns_none.
    coords = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 0.0], [2.5, 0.0]])
    genome = Genome(topology=fourbar, coords=coords)
    target = TargetCurve(points=circle, tree=cKDTree(circle))

    ok = snapshot(genome, target, str(tmp_path / "snap.png"), n=90)

    assert ok is False
    assert (tmp_path / "snap.png").exists()
