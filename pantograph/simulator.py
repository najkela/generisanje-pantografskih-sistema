"""Forward kinematics: од генома и угла crank-а до путање tracer чвора (README 1.2).

Ово је најтоплија петља пројекта и уједно јединица трошка буџета (README 3.3):
један позив `simulate` = један позив симулатора.
"""

import numpy as np

from .config import Config, DEFAULT_CONFIG
from .genome import CRANK, FIXED_A, Topology, link_lengths, tracer
from .validation import Step


def _link_length(lengths: dict[tuple[int, int], float], i: int, j: int) -> float:
    """Дужина крака (i,j) без обзира на редослед у кључу речника."""
    return lengths[(i, j)] if (i, j) in lengths else lengths[(j, i)]


class CallCounter:
    """Експлицитан бројач позива симулатора — једина дозвољена глобална статистика."""

    def __init__(self) -> None:
        self.calls = 0

    def increment(self, by: int = 1) -> None:
        self.calls += by

    def reset(self) -> None:
        self.calls = 0


def circle_intersect_pair(
    p1: np.ndarray,
    p2: np.ndarray,
    r1: float,
    r2: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Обе тачке пресека кругова (p1,r1) и (p2,r2), плус полутетива `h`; прва тачка
    одговара грани s=+1, друга s=−1.

    Пресек не постоји ако је `d > r1+r2` или `d < |r1−r2|` (README 1.4) — тада враћа `None`.
    Конвенција гране (README 1.2.1): нека је `u = (p2−p1)/d`; `s=+1` је тачка помакнута
    од средишње тачке дужи p1p2 у смеру ротације `u` за +90°, `s=−1` у супротном смеру.
    Ово је основна геометрија коју користе и `circle_intersect` (симулација) и
    `genome.to_sequence`/`from_sequence` (канонизација, README 1.2.1) — `h` се враћа само
    за потребе провере угла преноса у `circle_intersect`; канонизација га игнорише и остаје
    бихевиорално непромењена (docs/NALAZ_01_09_ugao_prenosa.md).
    """
    d = float(np.linalg.norm(p2 - p1))
    if d == 0.0:
        return None
    if d > r1 + r2 or d < abs(r1 - r2):
        return None
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = float(np.sqrt(max(r1**2 - a**2, 0.0)))
    u = (p2 - p1) / d
    midpoint = p1 + a * u
    perp = np.array([-u[1], u[0]])
    return midpoint + h * perp, midpoint - h * perp, h


def branch_signs(topology: Topology, coords: np.ndarray, order: list[Step]) -> dict[int, int]:
    """Знак гране circle-circle пресека сваког чвора, рачунат једном из θ=0 геометрије
    (README 4.4, DECISIONS §17) — својство склопа, не историје кретања.

    За чвор `u` са ослонцима `(a,b)`: `s=+1` ако `u` лежи са стране усмерене праве `a→b`
    одређене ротацијом вектора `(b-a)` за +90° (иста конвенција као `circle_intersect_pair`:
    `perp = [-u[1], u[0]]`, `p_plus = midpoint + h*perp`), иначе `s=-1`. Ово даје исте
    знакове као `genome.to_sequence` (тест: test_branch_signs_matches_to_sequence_signs) —
    обе функције кодирају исту грану, само различитим формулама (поређење растојања наспрам
    знака векторског производа).
    """
    signs: dict[int, int] = {}
    for step in order:
        u = step.target
        a, b = step.parents
        v = coords[b] - coords[a]
        cross = v[0] * (coords[u][1] - coords[a][1]) - v[1] * (coords[u][0] - coords[a][0])
        signs[u] = 1 if cross >= 0 else -1
    return signs


def circle_intersect(
    p1: np.ndarray,
    p2: np.ndarray,
    r1: float,
    r2: float,
    sign: int,
    min_sin_angle: float,
) -> np.ndarray | None:
    """Пресек два круга; бира грану закуцану знаком `sign` (README 4.4, DECISIONS §17).

    Враћа `None` ако пресека нема (d > r1+r2 или d < |r1−r2|) — то је невалидна
    геометрија у том тренутку, коју фитнес претвара у коначну казну (README 1.6).

    Враћа `None` и ако је угао преноса ∠(a–u–b) исувише близу 0°/180° (мртва тачка):
    `sin(∠aub) = d·h / (r1·r2)` из површине троугла (docs/NALAZ_01_09_ugao_prenosa.md) —
    два множења и једно дељење, без арккосинуса у петљи; `sin` покрива оба колинеарна
    случаја без грањања. Провера иде ПРЕ бирања гране (`sin_angle` не зависи од ње) —
    јефтино прво, скупо после, исти принцип као валидација топологије. `min_sin_angle` је
    `sin(radians(праг))` већ израчунат једном у `simulate`, не по позиву.

    Раније коришћена хеуристика "грана ближа претходном положају" (`_closer_branch`,
    уклоњена 02.09.) је била извор circuit defect проблема (README 4.4, DECISIONS §17) —
    повремено скаче на другу грану без детекције (docs/NALAZ_01_09_poza.md §2). `sign` је
    закуцан из θ=0 геометрије (`branch_signs`) и исти важи на сваком углу, па је прескок
    структурно немогућ.
    """
    pair = circle_intersect_pair(p1, p2, r1, r2)
    if pair is None:
        return None
    p_plus, p_minus, h = pair
    d = float(np.linalg.norm(p2 - p1))
    sin_angle = (d * h) / (r1 * r2)
    if sin_angle < min_sin_angle:
        return None
    return p_plus if sign > 0 else p_minus


def positions_at(
    topology: Topology,
    lengths: dict[tuple[int, int], float],
    base: np.ndarray,
    order: list[Step],
    angle: float,
    signs: dict[int, int],
    min_sin_angle: float,
) -> np.ndarray | None:
    """Позиције свих чворова за дати угао crank-а; `None` ако пресек не постоји или је
    угао преноса неког чвора испод прага (README 1.4, docs/NALAZ_01_09_ugao_prenosa.md).

    Конфигурација на углу θ зависи ИСКЉУЧИВО од θ и вектора знакова `signs` — без икакве
    зависности од историје (претходног угла или претходног позива), DECISIONS §17. `base`
    даје θ=0 геометрију фиксних чворова (0,1) — увек иста, никад резултат претходног позива.
    Crank(2) се поставља апсолутно на растојању `l(0,2)` под углом `angle` (нема drift-а).
    Остали чворови се решавају редом из `order` (README 1.2), свако бирајући своју грану из
    `signs[u]`, никад из близине претходне позиције — то је поента ове измене: раније
    коришћена хеуристика "грана ближа претходном положају" повремено прескаче на другу грану
    без детекције (docs/NALAZ_01_09_poza.md §2), што закуцана грана структурно искључује.
    """
    positions = base.copy()
    r02 = _link_length(lengths, FIXED_A, CRANK)
    positions[CRANK] = positions[FIXED_A] + r02 * np.array([np.cos(angle), np.sin(angle)])

    for step in order:
        u = step.target
        a, b = step.parents
        ra = _link_length(lengths, a, u)
        rb = _link_length(lengths, b, u)
        point = circle_intersect(positions[a], positions[b], ra, rb, signs[u], min_sin_angle)
        if point is None:
            return None
        positions[u] = point

    return positions


def simulate(
    topology: Topology,
    coords: np.ndarray,
    order: list[Step],
    n: int,
    config: Config = DEFAULT_CONFIG,
) -> np.ndarray | None:
    """Путања tracer чвора при пуној ротацији crank-а, `n` равномерних корака по θ ∈ [0, 2π).

    Дужине кракова се рачунају једном из `coords` (θ=0 геометрија) и остају константне
    (README 1.2.1). Апсолутни угао crank-а у θ=0 (`theta0`) се чита из `coords` — прва
    тачка путање тако тачно репродукује улазну геометрију, без вештачког скока на почетку.

    Праг угла преноса (`config.min_transmission_angle_deg`) се претвара у `sin` ЈЕДНОМ,
    ван петље по `n` и по solving order кораку (docs/NALAZ_01_09_ugao_prenosa.md) —
    `fitness.evaluate` позива ову функцију без `config`, добија подразумевани праг из
    `DEFAULT_CONFIG` без икакве измене у `fitness.py`.

    Враћа низ облика (n, 2) или `None` ако геометрија у неком тренутку није решива.
    """
    lengths = link_lengths(topology, coords)
    tracer_index = tracer(topology.n_nodes)
    theta0 = float(np.arctan2(*(coords[CRANK] - coords[FIXED_A])[::-1]))
    min_sin_angle = float(np.sin(np.radians(config.min_transmission_angle_deg)))
    signs = branch_signs(topology, coords, order)

    path = np.zeros((n, 2))
    for i, phi in enumerate(np.linspace(0.0, 2 * np.pi, n, endpoint=False)):
        positions = positions_at(topology, lengths, coords, order, theta0 + phi, signs, min_sin_angle)
        if positions is None:
            return None
        path[i] = positions[tracer_index]

    return path


def positions_and_angle_at(
    topology: Topology,
    lengths: dict[tuple[int, int], float],
    base: np.ndarray,
    order: list[Step],
    angle: float,
    signs: dict[int, int],
) -> tuple[np.ndarray, float] | None:
    """Као `positions_at`, БЕЗ прага, уз минимални `sin(угла преноса)` овог тренутка —
    чиста дијагностика (docs/NALAZ_01_09_ugao_prenosa.md, задатак 3): не позива се из
    `simulate`/`fitness.evaluate`, само из health-извештавања о најбољој јединки, да
    `positions_at` остане неоптерећена додатним рачунањем у hot петљи. Мора мерити исти
    механизам као `simulate` (DECISIONS §17), па узима исти закуцани `signs` уместо
    хеуристике по близини.
    """
    positions = base.copy()
    r02 = _link_length(lengths, FIXED_A, CRANK)
    positions[CRANK] = positions[FIXED_A] + r02 * np.array([np.cos(angle), np.sin(angle)])

    min_sin_angle = 1.0  # sin(90°) — најбезбеднија почетна вредност
    for step in order:
        u = step.target
        a, b = step.parents
        ra = _link_length(lengths, a, u)
        rb = _link_length(lengths, b, u)
        pair = circle_intersect_pair(positions[a], positions[b], ra, rb)
        if pair is None:
            return None
        p_plus, p_minus, h = pair
        d = float(np.linalg.norm(positions[b] - positions[a]))
        sin_angle = (d * h) / (ra * rb)
        min_sin_angle = min(min_sin_angle, sin_angle)
        positions[u] = p_plus if signs[u] > 0 else p_minus

    return positions, min_sin_angle


def simulate_with_transmission_angle(
    topology: Topology,
    coords: np.ndarray,
    order: list[Step],
    n: int,
    config: Config = DEFAULT_CONFIG,
) -> tuple[np.ndarray, float] | None:
    """Као `simulate`, али уз минимални угао преноса (степени) кроз цео обртај — чиста
    дијагностика ван буџета (docs/NALAZ_01_09_ugao_prenosa.md, задатак 3): не троши
    `CallCounter`, зове се једном по генерацији за најбољу јединку, не по кораку.
    """
    lengths = link_lengths(topology, coords)
    tracer_index = tracer(topology.n_nodes)
    theta0 = float(np.arctan2(*(coords[CRANK] - coords[FIXED_A])[::-1]))
    signs = branch_signs(topology, coords, order)

    path = np.zeros((n, 2))
    min_sin_angle = 1.0
    for i, phi in enumerate(np.linspace(0.0, 2 * np.pi, n, endpoint=False)):
        result = positions_and_angle_at(topology, lengths, coords, order, theta0 + phi, signs)
        if result is None:
            return None
        positions, step_min_sin = result
        min_sin_angle = min(min_sin_angle, step_min_sin)
        path[i] = positions[tracer_index]

    min_angle_deg = float(np.degrees(np.arcsin(np.clip(min_sin_angle, -1.0, 1.0))))
    return path, min_angle_deg


def path_health(path: np.ndarray) -> tuple[int, float]:
    """Скокови (корак > 8× медијане) и затварање петље (последња↔прва тачка, у односу
    на медијану корака) — чист post-processing над већ израчунатом путањом, без иједног
    додатног позива симулатора (docs/NALAZ_01_09_ugao_prenosa.md, задатак 3).
    """
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    median_step = float(np.median(steps))
    if median_step <= 0:
        return 0, 0.0
    jump_count = int(np.sum(steps > 8 * median_step))
    loop_closure = float(np.linalg.norm(path[-1] - path[0]) / median_step)
    return jump_count, loop_closure
