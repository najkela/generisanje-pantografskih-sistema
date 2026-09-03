"""Анимација механизма и цртање путање — дијагностика, не улази у мерења.

Сви наслови, легенде и осе се исписују ћирилицом.

Две путање цртања (проширено 31.08.):

* **интерактивна** — `save=None`: `pyplot` + `show()`, за `run.py demo` и `run.py replay
  --live`, где је блокирање и пожељно;
* **у фајл** — `save="...png"`: `Figure` + `FigureCanvasAgg` директно, БЕЗ `pyplot`-а.
  То је путања коју користи `progress.ProgressReporter` током тренинга: заобилази и
  глобално стање `pyplot`-а и избор backend-а, не отвара прозор, не блокира петљу и не
  оставља фигуре у регистру (нема цурења меморије кроз стотине снимака).

Ниједна функција овде не додирује ни бројач буџета ни RNG токове — цртање зове `simulate`
директно, никад `fitness.evaluate` (в. докстринг `progress`-а).
"""

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .config import Config, DEFAULT_CONFIG
from .curve import TargetCurve
from .genome import CRANK, FIXED_A, FIXED_B, Genome, link_lengths, tracer
from .simulator import branch_signs, positions_at, simulate
from .validation import InvalidTopology, solving_order, validate

TARGET_STYLE = dict(color="tab:gray", lw=1.5, label="циљна крива")
PATH_STYLE = dict(color="tab:red", lw=1.5, label="путања трагача")
LINK_STYLE = dict(color="tab:blue", lw=2.0)


def _render(draw, save: str | None, figsize=(6, 6)):
    """Заједнички оквир: исти `draw(ax)` иде или у фајл (Agg) или у прозор (pyplot).

    Раздвојено овако да се логика цртања не пише двапут и да се `pyplot` УОПШТЕ не увози
    кад се снима у фајл — увоз `pyplot`-а бира backend, што на macOS значи GUI backend.
    """
    if save is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        draw(ax)
        plt.show()
        return fig

    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    draw(fig.add_subplot(111))
    fig.savefig(save, dpi=140, bbox_inches="tight")
    return fig


def draw_mechanism(ax, genome: Genome, positions: np.ndarray | None = None) -> None:
    """Полуге и чворови механизма у затеченој геометрији (подразумевано θ = 0).

    Улоге се виде на слици јер се изводе из позиције у секвенци (DECISIONS §15): чворови
    0 и 1 су fixed (квадрати), 2 је crank, последњи је tracer.
    """
    coords = genome.coords if positions is None else positions
    topology = genome.topology
    tracer_index = tracer(topology.n_nodes)

    for a, b in topology.edges:
        ax.plot([coords[a, 0], coords[b, 0]], [coords[a, 1], coords[b, 1]], "-", **LINK_STYLE)

    floating = [
        i for i in range(topology.n_nodes)
        if i not in (FIXED_A, FIXED_B, CRANK, tracer_index)
    ]
    if floating:
        ax.plot(coords[floating, 0], coords[floating, 1], "o", color="tab:blue", ms=5)
    ax.plot(coords[[FIXED_A, FIXED_B], 0], coords[[FIXED_A, FIXED_B], 1],
            "s", color="black", ms=8, label="fixed (0, 1)")
    ax.plot(coords[CRANK, 0], coords[CRANK, 1], "o", color="tab:green", ms=8, label="crank (2)")
    ax.plot(coords[tracer_index, 0], coords[tracer_index, 1],
            "o", color="tab:red", ms=8, label=f"tracer ({tracer_index})")


def _fit_axes(ax, *point_sets: np.ndarray) -> None:
    """Једнаке осе и оквир који обухвата све дате скупове тачака, са 10% ваздуха."""
    points = np.vstack([p for p in point_sets if p is not None and len(p)])
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    span = float(np.max(high - low)) * 0.55 + 1e-9
    ax.set_aspect("equal")
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)


def tracer_path(genome: Genome, n: int, config: Config = DEFAULT_CONFIG) -> np.ndarray | None:
    """Путања трагача за цртање — `validate` + `simulate`, НИКАД `fitness.evaluate`.

    То је оно што снимке чини бесплатним: бројач буџета инкрементира само `evaluate`
    (README 3.3), па колико год се слика нацртало, потрошња позива остаје нетакнута.
    Враћа `None` ако је топологија неважећа или геометрија падне на circuit defect.

    `config` се прослеђује до `simulate` (праг угла преноса) — иначе би слика/поза увек
    користила `DEFAULT_CONFIG`, тихо мимо конфига под којим је покретање стварно радило
    (DECISIONS §17; в. `run.py::run_experiment`, које прослеђује своје `config`).
    """
    try:
        order = validate(genome.topology)
    except InvalidTopology:
        return None
    return simulate(genome.topology, genome.coords, order, n, config)


def snapshot(
    genome: Genome,
    target: TargetCurve,
    save: str,
    title: str = "",
    n: int = 720,
) -> bool:
    """Један PNG: механизам у θ = 0 + путања трагача преко циљне криве.

    Црта се на фиксном `n` (подразумевано највећи из распореда), без обзира на тренутну
    резолуцију претраге — слика тако изгледа глатко и рано у тренингу, а не кошта ништа
    јер није позив који се броји.

    Враћа `True` ако је слика снимљена. `False` значи да јединка у том тренутку није
    решива (неважећа топологија или circuit defect) — слика се свеједно снима, са
    механизмом и напоменом уместо путање, јер је и то информација вредна гледања.
    """
    path = tracer_path(genome, n)

    def draw(ax):
        ax.plot(target.points[:, 0], target.points[:, 1], "-", **TARGET_STYLE)
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], "-", **PATH_STYLE)
        draw_mechanism(ax, genome)
        _fit_axes(ax, target.points, path, genome.coords)
        ax.set_title(title or "Најбоља јединка", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        if path is None:
            ax.text(0.02, 0.02, "јединка није решива (circuit defect)",
                    transform=ax.transAxes, color="tab:red", fontsize=9)

    _render(draw, save)
    return path is not None


def plot_mechanism(genome: Genome, title: str = "", save: str | None = None):
    """Само механизам у θ = 0, без путање — за `run.py show` и `mechanism.png` (README 4.3)."""

    def draw(ax):
        draw_mechanism(ax, genome)
        _fit_axes(ax, genome.coords)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(title or "Механизам")

    return _render(draw, save)


def plot_comparison(path: np.ndarray, target: TargetCurve, title: str = "", save: str | None = None):
    """Генерисана путања преко циљне криве — визуелна провера фитнеса."""

    def draw(ax):
        ax.plot(target.points[:, 0], target.points[:, 1], "-", **TARGET_STYLE)
        ax.plot(path[:, 0], path[:, 1], "-", **PATH_STYLE)
        _fit_axes(ax, target.points, path)
        ax.legend(loc="upper right")
        ax.set_title(title or "Путања трагача наспрам циљне криве")

    return _render(draw, save)


def plot_error_curve(logs: list, title: str = "", save: str | None = None):
    """Грешка у функцији броја позива симулатора, не генерација (README 3.2).

    `logs: list[experiment.RunLog]` — свака се црта као једна линија (x = `calls_spent`,
    y = `best_fitness`, из `log.error_curve`), означена `log.method` + `log.seed` ако их
    има више истог метода — управо оно што омогућава директно H1 поређење baseline/bilevel
    на истом графику. По ПОЗИВИМА симулатора, не по индексу генерације — генерације нису
    фер јединица трошка (BASELINE_SPEC §9, DECISIONS §8), а `calls_spent` јесте.

    Уз криву се црта и распоред N: вертикалне испрекидане линије тамо где је `n_curve`
    порастао (31.08.). Без њих је скок грешке нагоре на прелазу необјашњив — а он није
    погоршање него промена мерила: Chamfer на различитом N није иста величина.
    """
    methods_seen = [log.method for log in logs]

    def draw(ax):
        for log in logs:
            curve = log.error_curve
            if not curve:
                continue
            calls, best = zip(*curve)
            label = log.method
            if methods_seen.count(log.method) > 1:
                label = f"{log.method} (seed={log.seed})"
            ax.plot(calls, best, "-", label=label)

        if len(logs) == 1:
            previous = None
            for record in logs[0].records:
                if previous is not None and record.n_curve != previous:
                    ax.axvline(record.calls_spent, color="tab:gray", ls=":", lw=1)
                    ax.annotate(f"N={record.n_curve}", (record.calls_spent, 1.0),
                                xycoords=("data", "axes fraction"), fontsize=8,
                                color="tab:gray", ha="left", va="top")
                previous = record.n_curve

        ax.set_xlabel("број позива симулатора")
        ax.set_ylabel("најбољи Chamfer")
        ax.set_yscale("log")
        ax.set_title(title or "Конвергенција по броју позива симулатора")
        ax.legend(loc="upper right")

    return _render(draw, save, figsize=(7, 5))


def animate(genome: Genome, n: int = 200, save: str | None = None, target: TargetCurve | None = None):
    """Анимација пуне ротације crank-а; наследник прототипа `simulator.py` из корена.

    Црта тренутне полуге механизма и траг који tracer чвор оставља за собом
    (BASELINE_SPEC корак Д). `save="...gif"` снима уместо да отвара прозор — GIF иде
    право у слајд и не зависи од GUI backend-а на туђој машини.
    """
    topology = genome.topology
    coords = genome.coords
    order = solving_order(topology)
    lengths = link_lengths(topology, coords)
    tracer_index = tracer(topology.n_nodes)
    theta0 = float(np.arctan2(*(coords[CRANK] - coords[FIXED_A])[::-1]))
    angles = theta0 + np.linspace(0.0, 2 * np.pi, n, endpoint=False)

    full_path = simulate(topology, coords, order, max(n, 360))

    if save is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = Figure(figsize=(6, 6))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

    _fit_axes(ax, coords, full_path, target.points if target is not None else None)
    ax.set_title("Анимација пантографског механизма")
    if target is not None:
        ax.plot(target.points[:, 0], target.points[:, 1], "-", **TARGET_STYLE)

    edge_lines = [ax.plot([], [], "o-", **LINK_STYLE)[0] for _ in topology.edges]
    trace_line, = ax.plot([], [], "-", **PATH_STYLE)
    ax.legend(loc="upper right", fontsize=8)

    # Знак гране је закуцан из θ=0 геометрије једном, пре анимације (DECISIONS §17) —
    # конфигурација на сваком углу зависи искључиво од угла и `signs`, не од претходног
    # кадра, па база остаје увек `coords`.
    signs = branch_signs(topology, coords, order)
    state = {"trace": []}
    min_sin_angle = float(np.sin(np.radians(DEFAULT_CONFIG.min_transmission_angle_deg)))

    def update(angle):
        positions = positions_at(topology, lengths, coords, order, angle, signs, min_sin_angle)
        if positions is None:
            return edge_lines + [trace_line]
        state["trace"].append(positions[tracer_index].copy())

        for line, (a, b) in zip(edge_lines, topology.edges):
            line.set_data([positions[a, 0], positions[b, 0]], [positions[a, 1], positions[b, 1]])
        trace = np.array(state["trace"])
        trace_line.set_data(trace[:, 0], trace[:, 1])
        return edge_lines + [trace_line]

    animation = FuncAnimation(fig, update, frames=angles, blit=True, interval=20)

    if save is None:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        animation.save(save, writer=PillowWriter(fps=25))
    return animation
