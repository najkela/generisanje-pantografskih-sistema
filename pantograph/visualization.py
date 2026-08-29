"""Анимација механизма и цртање путање — дијагностика, не улази у мерења.

Сви наслови, легенде и осе се исписују ћирилицом.
"""

import numpy as np

from .curve import TargetCurve
from .genome import Genome


def animate(genome: Genome, n: int = 200):
    """Анимација пуне ротације crank-а; наследник прототипа `simulator.py` из корена."""
    raise NotImplementedError


def plot_comparison(path: np.ndarray, target: TargetCurve, title: str = ""):
    """Генерисана путања преко циљне криве — визуелна провера фитнеса."""
    raise NotImplementedError


def plot_error_curve(logs: list, title: str = ""):
    """Грешка у функцији броја позива симулатора, не генерација (README 3.2)."""
    raise NotImplementedError
