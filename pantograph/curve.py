"""Учитавање, нормализација и подузорковање циљне криве (README 1.1, 1.5)."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def load(path: str) -> np.ndarray:
    """Фајл са по једном тачком по реду, координате одвојене зарезом → низ (m, 2)."""
    raise NotImplementedError


def normalize(points: np.ndarray) -> np.ndarray:
    """Центар bounding box-a → (0,0); скалирање тако да највеће растојање од центра буде 1.

    Нормализује се *само циљна крива*, једном при учитавању (README 1.5).
    Центар bounding box-a, не центроид — независан од густине узорковања.
    Највеће растојање, не површина bounding box-a — површина лажно изједначава
    криве различитог облика.
    """
    raise NotImplementedError


def resample(points: np.ndarray, n: int) -> np.ndarray:
    """Равномерно подузорковање на `n` тачака (динамички N, README 2.4)."""
    raise NotImplementedError


@dataclass
class TargetCurve:
    """Нормализована циљна крива са унапред изграђеним KD-дрветом (README 1.6)."""

    points: np.ndarray
    tree: cKDTree

    @classmethod
    def from_file(cls, path: str) -> "TargetCurve":
        """Учитај → нормализуј → изгради KD-дрво (једном, за цео ток оптимизације)."""
        raise NotImplementedError
