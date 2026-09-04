"""Преглед свих кривих нивоа 1 у једној слици (README 3.4, docs/DECISIONS.md §19).

Свака крива у сопственом панелу, `equal` размера, без нормализације на заједничку скалу —
циљ је визуелна провера облика, не поређење величина (величине се решавају сличносном
трансформацијом при постављању механизма, README 1.5).

Покретање:  python tools/plot_curves_level1.py [излаз.png]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # корен репоа

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pantograph.curve import load

CURVE_NAMES = ["circle", "ellipse", "egg", "limacon", "cassini", "superellipse"]
CURVES_DIR = "data/curves"


def main(out_path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, name in zip(axes.flat, CURVE_NAMES):
        points = load(os.path.join(CURVES_DIR, f"{name}.txt"))
        closed = list(points) + [points[0]]
        xs = [p[0] for p in closed]
        ys = [p[1] for p in closed]
        ax.plot(xs, ys, "-", color="tab:blue", lw=1.5)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=12)
        ax.tick_params(labelsize=8)

    fig.suptitle("Циљне криве нивоа 1 (README 3.4, DECISIONS §19)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"сачувано: {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/krive_nivo1.png")
