"""Учитавање, нормализација и подузорковање циљне криве (README 1.1, 1.5)."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def load(path: str) -> np.ndarray:
    """Фајл са по једном тачком по реду, координате одвојене зарезом → низ (m, 2)."""
    return np.loadtxt(path, delimiter=",", dtype=float).reshape(-1, 2)


def normalize(points: np.ndarray) -> np.ndarray:
    """Центар bounding box-a → (0,0); скалирање тако да највеће растојање од центра буде 1.

    Нормализује се *само циљна крива*, једном при учитавању (README 1.5).
    Центар bounding box-a, не центроид — независан од густине узорковања.
    Највеће растојање, не површина bounding box-a — површина лажно изједначава
    криве различитог облика.
    """
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    centered = points - center
    scale = np.linalg.norm(centered, axis=1).max()
    return centered / scale


def resample(points: np.ndarray, n: int) -> np.ndarray:
    """Равномерно подузорковање на `n` тачака (динамички N, README 2.4).

    Крива се третира као затворена изломљена линија (README 1.1) — последња тачка
    се спаја с првом. Узорковање је равномерно по дужини лука, не по индексу.
    """
    m = len(points)
    closed = np.vstack([points, points[0]])
    segments = np.diff(closed, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    perimeter = cumulative[-1]

    targets = np.linspace(0.0, perimeter, n, endpoint=False)
    segment_index = np.clip(np.searchsorted(cumulative, targets, side="right") - 1, 0, m - 1)
    local_fraction = (targets - cumulative[segment_index]) / segment_lengths[segment_index]
    return closed[segment_index] + local_fraction[:, None] * segments[segment_index]


@dataclass
class TargetCurve:
    """Нормализована циљна крива са унапред изграђеним KD-дрветом (README 1.6).

    `path` је опционо (30.08.) — попуњава га само `from_file`, потребно логовању
    (`experiment.curve_file_hash`, `RunLog.curve`); ручна конструкција (тестови) га
    оставља на `None`.
    """

    points: np.ndarray
    tree: cKDTree
    path: str | None = None

    @classmethod
    def from_file(cls, path: str) -> "TargetCurve":
        """Учитај → нормализуј → изгради KD-дрво (једном, за цео ток оптимизације)."""
        points = normalize(load(path))
        return cls(points=points, tree=cKDTree(points), path=path)
