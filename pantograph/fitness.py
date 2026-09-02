"""Chamfer растојање и казне за невалидна решења (README 1.6)."""

import numpy as np
from scipy.spatial import cKDTree

from .config import Config, DEFAULT_CONFIG
from .curve import TargetCurve, center_and_radius
from .genome import Genome, link_lengths
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
    config: Config = DEFAULT_CONFIG,
) -> float:
    """Фитнес једне јединке: валидација → симулација → Chamfer, уз казну на сваком паду.

    Јединица трошка буџета (README 3.3, DECISIONS §8, план 31.08. питање 1) — један позив
    `evaluate` троши тачно **један** позив буџета, без обзира да ли је геном валидан
    („1 јединка = 1 позив", не „1 позив simulate()"): невалидна јединка ухваћена пре
    `simulate()`-а и даље троши буџет. `counter` је опциони (нпр. `experiment.Budget.counter`)
    — инкрементира се тачно једном, на свакој грани. Валидан фитнес се одсеца на
    `FITNESS_CAP` да невалидна јединка увек остане строго гора од валидне (§7).

    Пуна скална инваријантност (DECISIONS §17, README 1.5): генерисана путања `G` се пре
    Chamfer-a нормализује истом формулом као циљна крива `T` — фитнес мери искључиво облик,
    поза (транслација/ротација/скала) се решава ван претраге (в. `run.py` на крају
    покретања). Центар/радијус се рачунају ЈЕДНОМ преко `curve.center_and_radius` (најврелија
    петља, 50 000 позива по покретању — не сме два пролаза кроз низ): ако је радијус
    дегенерисане путање (сведена на тачку) ~0, враћа се `PENALTY` пре дељења нулом.
    Циљна крива се подузоркује на текуће `n` (`TargetCurve.at_resolution`) да оба скупа буду
    исте густине.

    Гломазност (DECISIONS §17, ревизија) — ТВРДА ГРАНИЦА уведена ПОСЛЕ пуне скалне
    инваријантности: без ње претрага одлази у гломазна решења (мерено 6.73× и 26.78× на
    буџету од 12 000 позива, механизам 27× већи од своје криве). Однос највеће полуге и
    полупречника путање изнад `config.max_link_to_radius_ratio` → `PENALTY`. Ово НИЈЕ члан у
    фитнесу — Chamfer остаје једина мера квалитета поклапања, гломазност само одсеца
    неупотребљив део простора решења пре него што се уопште мери облик.
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

    center, radius = center_and_radius(path)
    if radius < 1e-9:
        return PENALTY  # дегенерисана путања (сведена на тачку) — дељење нулом у normalize

    largest_link = max(link_lengths(genome.topology, genome.coords).values())
    if largest_link / radius > config.max_link_to_radius_ratio:
        return PENALTY  # гломазан механизам — тврда граница, DECISIONS §17 (ревизија)

    normalized_path = (path - center) / radius  # нумерички идентично curve.normalize(path)
    return min(chamfer(normalized_path, target.at_resolution(n)), FITNESS_CAP)
