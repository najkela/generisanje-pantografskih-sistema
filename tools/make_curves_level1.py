"""Генератор аналитичких кривих нивоа 1 (README 3.4, docs/OPEN_QUESTIONS.md 6.5).

Круг и елипса су прве две криве нивоа 1. Под пуном скалном инваријантношћу фитнеса
(DECISIONS §17) круг је постао дегенерисан тест — кружна путања БИЛО КОГ полупречника
савршено га погађа — па је елипса остала једини прави тест нивоа 1. Ове четири криве
попуњавају распон између елипсе и нивоа 2 (слова).

Свака крива додаје ДРУГУ врсту тежине, не само „више исто":

  1. `egg`          — асиметрична овална: руши двоструку симетрију елипсе (елипса има две
                      осе симетрије, јаје једну). Најлакша од четири.
  2. `limacon`      — улубљени лимасон: уводи промену знака кривине (једна конкавна област)
                      на иначе округлом облику.
  3. `cassini`      — Касинијев овал (кикирики): изражен струк, два режња, јака
                      неконвексност уз задржану симетрију.
  4. `superellipse` — суперелипса n=4: равне странице и заобљени углови, кривина
                      сконцентрисана у четири области — најближе словима без корнера.

`figure8` (Лисажу 1:2, самопресецајућа осмица) је ИЗБАЧЕНА 04.09. (docs/DECISIONS.md §19,
ревизија): самопресек тражи да трагач двапут прође исту тачку равни у различитим фазама
обртаја, што је квалитативно другачији захтев од затворене криве без самопресека — четворо-
полужје са спрежном тачком то не мора да поседује. Мерено: `figure8` је на сва три seed-а
стала на 2.4e-02 до 3.0e-02, ред величине изнад следеће најтеже криве (`superellipse`,
5.6e-03 до 7.0e-03). Критеријум за ниво 1 (README 3.4): циљна крива мора бити затворена и
без самопресека — конвексност се НЕ тражи (потврђено на Јансеновој кривој, README 3.4).

Ниједна крива нема шпиц (cusp): шпиц захтева да трагач тренутно стане, што је сингуларна
конфигурација коју праг угла преноса (`Config.min_transmission_angle_deg`) ионако одбија.
Зато кардиоида и слични облици НИСУ у скупу — не би били тешки него неизводљиви.

Узорковање је равномерно по ДУЖИНИ ЛУКА, не по параметру: `fitness.evaluate` циљну криву
ионако подузоркује преко `curve.resample` (по дужини лука), па су фајлови тако саморадни.
Формат излаза је исти као `circle.txt`/`ellipse.txt`: 720 редова, `x,y`, шест децимала.

Покретање:  python tools/make_curves_level1.py [излазни_фолдер]
"""

import numpy as np

N_POINTS = 720
DENSE = 200_000  # густо узорковање пре преузорковања по дужини лука


def _arc_length_resample(points: np.ndarray, n: int) -> np.ndarray:
    """Равномерно по дужини лука, третирајући криву као затворену изломљену линију."""
    closed = np.vstack([points, points[0]])
    steps = np.diff(closed, axis=0)
    seg = np.linalg.norm(steps, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, cum[-1], n, endpoint=False)
    idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(points) - 1)
    frac = (targets - cum[idx]) / seg[idx]
    return closed[idx] + frac[:, None] * steps[idx]


def egg() -> np.ndarray:
    """Асиметрична овална крива — једна оса симетрије уместо елипсиних двеју."""
    t = np.linspace(0.0, 2 * np.pi, DENSE, endpoint=False)
    return np.column_stack([np.cos(t), 0.62 * np.sin(t) * (1.0 + 0.30 * np.cos(t))])


def limacon() -> np.ndarray:
    """Улубљени лимасон Паскалов, r = b + a·cos θ уз 1 < b/a < 2 (без унутрашње петље)."""
    t = np.linspace(0.0, 2 * np.pi, DENSE, endpoint=False)
    r = 1.5 + 1.0 * np.cos(t)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def cassini() -> np.ndarray:
    """Касинијев овал, c < a < c·√2 → повезан „кикирики" са струком."""
    c, a = 1.0, 1.08
    t = np.linspace(0.0, 2 * np.pi, DENSE, endpoint=False)
    r2 = c ** 2 * np.cos(2 * t) + np.sqrt(a ** 4 - c ** 4 * np.sin(2 * t) ** 2)
    r = np.sqrt(np.maximum(r2, 0.0))
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def superellipse() -> np.ndarray:
    """Ламеова крива |x/A|⁴ + |y/B|⁴ = 1 — равне странице, заобљени углови."""
    t = np.linspace(0.0, 2 * np.pi, DENSE, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    x = 1.00 * np.sign(ct) * np.abs(ct) ** 0.5
    y = 0.68 * np.sign(st) * np.abs(st) ** 0.5
    return np.column_stack([x, y])


CURVES = {
    "egg": egg,
    "limacon": limacon,
    "cassini": cassini,
    "superellipse": superellipse,
}


def write(name: str, path: str) -> np.ndarray:
    """Испиши криву у формату README 1.1 и врати тачке."""
    points = _arc_length_resample(CURVES[name](), N_POINTS)
    if points.shape != (N_POINTS, 2) or not np.isfinite(points).all():
        raise ValueError(f"Крива {name} није дала {N_POINTS} коначних тачака.")
    with open(path, "w", encoding="utf-8") as f:
        for x, y in points:
            f.write(f"{x:.6f},{y:.6f}\n")
    return points


if __name__ == "__main__":
    import os
    import sys

    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/curves"
    os.makedirs(out_dir, exist_ok=True)
    for curve_name in CURVES:
        target = os.path.join(out_dir, f"{curve_name}.txt")
        written = write(curve_name, target)
        print(f"{curve_name:14s} → {target}  ({len(written)} тачака)")
