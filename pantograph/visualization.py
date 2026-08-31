"""Анимација механизма и цртање путање — дијагностика, не улази у мерења.

Сви наслови, легенде и осе се исписују ћирилицом.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .curve import TargetCurve
from .genome import CRANK, FIXED_A, Genome, link_lengths, tracer
from .simulator import positions_at
from .validation import solving_order


def animate(genome: Genome, n: int = 200):
    """Анимација пуне ротације crank-а; наследник прототипа `simulator.py` из корена.

    Црта тренутне полуге механизма и траг који tracer чвор оставља за собом
    (BASELINE_SPEC корак Д).
    """
    topology = genome.topology
    coords = genome.coords
    order = solving_order(topology)
    lengths = link_lengths(topology, coords)
    tracer_index = tracer(topology.n_nodes)
    theta0 = float(np.arctan2(*(coords[CRANK] - coords[FIXED_A])[::-1]))
    angles = theta0 + np.linspace(0.0, 2 * np.pi, n, endpoint=False)

    span = max(float(np.abs(coords).max()) * 2.5, 1.0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_title("Анимација пантографског механизма")

    edge_lines = [ax.plot([], [], "o-", lw=2, color="tab:blue")[0] for _ in topology.edges]
    trace_line, = ax.plot([], [], "-", lw=1, color="tab:red", label="путања трагача")
    ax.legend(loc="upper right")

    state = {"positions": coords.copy(), "trace": []}

    def update(angle):
        positions = positions_at(topology, lengths, state["positions"], order, angle)
        if positions is None:
            return edge_lines + [trace_line]
        state["positions"] = positions
        state["trace"].append(positions[tracer_index].copy())

        for line, (a, b) in zip(edge_lines, topology.edges):
            line.set_data([positions[a, 0], positions[b, 0]], [positions[a, 1], positions[b, 1]])
        trace = np.array(state["trace"])
        trace_line.set_data(trace[:, 0], trace[:, 1])
        return edge_lines + [trace_line]

    animation = FuncAnimation(fig, update, frames=angles, blit=True, interval=20)
    plt.show()
    return animation


def plot_comparison(path: np.ndarray, target: TargetCurve, title: str = ""):
    """Генерисана путања преко циљне криве — визуелна провера фитнеса."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.plot(target.points[:, 0], target.points[:, 1], "-", color="tab:gray", label="циљна крива")
    ax.plot(path[:, 0], path[:, 1], "-", color="tab:red", label="путања трагача")
    ax.legend(loc="upper right")
    ax.set_title(title or "Путања трагача наспрам циљне криве")
    plt.show()
    return fig


def plot_error_curve(logs: list, title: str = ""):
    """Грешка у функцији броја позива симулатора, не генерација (README 3.2)."""
    raise NotImplementedError
