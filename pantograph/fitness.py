"""Chamfer растојање и казне за невалидна решења (README 1.6)."""

import numpy as np
from scipy.spatial import cKDTree

from .curve import TargetCurve
from .genome import Genome
from .simulator import CallCounter, simulate
from .validation import InvalidTopology, validate

# Никад inf и никад nan, ради стабилности ЦМА-ЕС-a (README 1.6, BASELINE_SPEC §7, 30.08.).
PENALTY = 1e9          # казна за невалидну јединку (топологија/геометрија не решива)
FITNESS_CAP = 1e8      # одсецање валидног фитнеса — валидна јединка увек строго боља од казне


def chamfer(generated: np.ndarray, target: TargetCurve) -> float:
    """Симетрично Chamfer растојање d(G, T) (README 1.6).

    d = (1/N) Σ_i min_j ||g_i − t_j||²  +  (1/m) Σ_j min_i ||t_j − g_i||²

    Симетрично да спречи дегенерисана решења; инваријантно на смер и брзину обиласка.
    Први члан користи KD-дрво циљне криве, други KD-дрво генерисане путање.
    """
    generated = np.asarray(generated)
    dist_to_target, _ = target.tree.query(generated)
    generated_tree = cKDTree(generated)
    dist_to_generated, _ = generated_tree.query(target.points)
    return float(np.mean(dist_to_target**2) + np.mean(dist_to_generated**2))


def evaluate(
    genome: Genome,
    target: TargetCurve,
    n: int,
    counter: CallCounter | None = None,
) -> float:
    """Фитнес једне јединке: валидација → симулација → Chamfer, уз казну на сваком паду.

    Јединица трошка буџета (README 3.3, DECISIONS §8, план 31.08. питање 1) — један позив
    `evaluate` троши тачно **један** позив буџета, без обзира да ли је геном валидан
    („1 јединка = 1 позив", не „1 позив simulate()"): невалидна јединка ухваћена пре
    `simulate()`-а и даље троши буџет. `counter` је опциони (нпр. `experiment.Budget.counter`)
    — инкрементира се тачно једном, на свакој грани. Валидан фитнес се одсеца на
    `FITNESS_CAP` да невалидна јединка увек остане строго гора од валидне (§7).
    """
    if counter is not None:
        counter.increment()
    try:
        order = validate(genome.topology)
    except InvalidTopology:
        return PENALTY
    path = simulate(genome.topology, genome.coords, order, n)
    if path is None:
        return PENALTY
    return min(chamfer(path, target), FITNESS_CAP)
