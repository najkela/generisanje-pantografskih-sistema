"""Учитавање, нормализација и подузорковање циљне криве (README 1.1, 1.5)."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


def load(path: str) -> np.ndarray:
    """Фајл са по једном тачком по реду, координате одвојене зарезом → низ (m, 2)."""
    return np.loadtxt(path, delimiter=",", dtype=float).reshape(-1, 2)


def center_and_radius(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Центар bounding box-a и највеће растојање тачке од центра (README 1.5).

    Дељено између `normalize` и `fitness.evaluate` (DECISIONS §17), да се bbox и норма не
    рачунају двапут у најврелијој петљи (50 000 позива симулатора по покретању): `evaluate`
    треба радијус ПРЕ дељења (заштита од дегенерисане путање), `normalize` исти радијус КАО
    делилац — иста формула, рачуната једном.
    """
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    radius = np.linalg.norm(points - center, axis=1).max()
    return center, radius


def normalize(points: np.ndarray) -> np.ndarray:
    """Центар bounding box-a → (0,0); скалирање тако да највеће растојање од центра буде 1.

    Нормализује се циљна крива једном при учитавању, и генерисана путања при свакој
    евалуацији истом функцијом (README 1.5, DECISIONS §17). Центар bounding box-a, не
    центроид — независан од густине узорковања. Највеће растојање, не површина bounding
    box-a — површина лажно изједначава криве различитог облика.
    """
    center, scale = center_and_radius(points)
    return (points - center) / scale


def apply_similarity_transform(
    points: np.ndarray, tx: float, ty: float, angle: float, scale: float
) -> np.ndarray:
    """Транслација + ротација + униформна скала над скупом тачака (README 1.5, DECISIONS §17).

    Користи се и за путању и за координате генома — сличносна трансформација координата даје
    идентично трансформисану путању (провера: `tests/test_curve.py`), па се примењује на
    `Genome.coords` на крају покретања да снимљени механизам стварно исцртава циљну криву на
    њеном месту и у њеној величини.
    """
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return (points @ rotation.T) * scale + np.array([tx, ty])


def place_on_target(path: np.ndarray) -> tuple[float, float, float, float]:
    """Поравнање генерисане путање са циљном кривом — сличносна трансформација у затвореном
    облику (DECISIONS §17, ревизија).

    Фитнес пореди `normalize(path)` са већ нормализованом циљном кривом, а `normalize` не
    ротира — ГА је дакле оријентацију већ погодио сам, па је ротација овде нула по
    конструкцији. Остају транслација и скала, одређене центром и радијусом путање
    (`center_and_radius`, иста формула коју користи `normalize` и `fitness.evaluate`).

    Резултат: сирова Chamfer постављеног механизма (примени трансформацију на `Genome.coords`,
    поново симулирај, упореди без нормализације против `TargetCurve.at_resolution(n)`) је
    ТАЧНО једнака вредности коју враћа `fitness.evaluate` за исту јединку/N (тест:
    `tests/test_fitness.py`, поклапање до `1e-12`) — поравнање нема режим у ком може да
    омане, за разлику од итеративне (Nelder-Mead) варијанте коју замењује.
    """
    center, radius = center_and_radius(path)
    if radius < 1e-9:            # дегенерисана путања — иста граница као у `evaluate`
        return 0.0, 0.0, 0.0, 1.0
    return -center[0] / radius, -center[1] / radius, 0.0, 1.0 / radius


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
    _resolution_cache: dict[int, "TargetCurve"] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_file(cls, path: str) -> "TargetCurve":
        """Учитај → нормализуј → изгради KD-дрво (једном, за цео ток оптимизације)."""
        points = normalize(load(path))
        return cls(points=points, tree=cKDTree(points), path=path)

    def at_resolution(self, n: int) -> "TargetCurve":
        """Циљна крива подузоркована на `n` тачака, исте густине као путања за дато N
        (README 1.6, DECISIONS §17). KD-дрво се гради само при ПРВОМ позиву за то `n`,
        кеширано по `n` — иначе би се градило по свакој евалуацији (најврелија петља).
        """
        if n not in self._resolution_cache:
            resampled = resample(self.points, n)
            self._resolution_cache[n] = TargetCurve(points=resampled, tree=cKDTree(resampled), path=self.path)
        return self._resolution_cache[n]
