"""Chamfer растојање и казне за невалидна решења (README 1.6)."""

import numpy as np

from .curve import TargetCurve

# Велика КОНАЧНА казна — никад inf и никад nan, ради стабилности ЦМА-ЕС-a (README 1.6).
PENALTY = 1e6


def chamfer(generated: np.ndarray, target: TargetCurve) -> float:
    """Симетрично Chamfer растојање d(G, T) (README 1.6).

    d = (1/N) Σ_i min_j ||g_i − t_j||²  +  (1/m) Σ_j min_i ||t_j − g_i||²

    Симетрично да спречи дегенерисана решења; инваријантно на смер и брзину обиласка.
    Први члан користи KD-дрво циљне криве, други KD-дрво генерисане путање.
    """
    raise NotImplementedError


def evaluate(genome, target: TargetCurve, n: int) -> float:
    """Фитнес једне јединке: валидација → симулација → Chamfer, уз казну на сваком паду."""
    raise NotImplementedError
