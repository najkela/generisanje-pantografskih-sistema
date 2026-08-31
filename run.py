"""CLI улазна тачка: покретање једног експеримента (README Део III) и demo провера
(BASELINE_SPEC корак Д).

Примери:
    python run.py demo --curve data/curves/circle.txt
    python run.py run --method baseline --curve data/curves/circle.txt --seed 1
    python run.py run --method bilevel --curve data/curves/jansen.txt --budget 150 --seed 1
"""

import argparse

import numpy as np

from pantograph.curve import TargetCurve
from pantograph.fitness import chamfer
from pantograph.genome import Genome, Topology
from pantograph.simulator import simulate
from pantograph.validation import solving_order, validate
from pantograph.visualization import animate, plot_comparison


def _demo_genome() -> Genome:
    """Конкретан четворочлани механизам са задатим (не насумичним) координатама —
    circle-crank-rocker, пуна ротација crank-а изводљива (BASELINE_SPEC корак Д)."""
    topology = Topology(n_nodes=4, edges=[(0, 2), (1, 3), (2, 3)])
    coords = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 0.0], [3.0, 2.0]])
    return Genome(topology=topology, coords=coords)


def run_demo(args: argparse.Namespace) -> None:
    """Склопи конкретан механизам, одврти crank, нацртај анимацију и путању преко циљне
    криве, испиши Chamfer растојање (BASELINE_SPEC корак Д) — провера разумевања, не мерење.
    """
    genome = _demo_genome()
    order = validate(genome.topology)

    path = simulate(genome.topology, genome.coords, order, args.n)
    if path is None:
        print("Механизам није решив у пуном обртају crank-а (circuit defect).")
        return

    target = TargetCurve.from_file(args.curve)
    error = chamfer(path, target)
    print(f"Chamfer растојање (демо механизам наспрам {args.curve}): {error:.6f}")

    animate(genome, n=args.n)
    plot_comparison(path, target, title="Демо: путања трагача наспрам циљне криве")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Генерисање пантографских механизама")
    sub = p.add_subparsers(dest="command", required=True)

    demo_p = sub.add_parser(
        "demo", help="визуелна провера: један механизам, анимација, Chamfer (SPEC корак Д)"
    )
    demo_p.add_argument("--curve", default="data/curves/circle.txt",
                         help="фајл са тачкама циљне криве (ниво 1, README 3.4)")
    demo_p.add_argument("--n", type=int, default=360,
                         help="број равномерних корака по θ ∈ [0, 2π)")

    run_p = sub.add_parser("run", help="покрени експеримент (baseline/bilevel)")
    run_p.add_argument("--method", choices=["baseline", "bilevel"], required=True,
                        help="baseline = чист ГА (README 2.2); bilevel = ГА + ЦМА-ЕС (README 2.3)")
    run_p.add_argument("--curve", required=True, help="фајл са тачкама циљне криве")
    run_p.add_argument("--budget", type=int, default=150,
                        help="максималан број позива симулатора (README 3.3)")
    run_p.add_argument("--seed", type=int, default=0, help="семе случајности (README 4.3)")
    run_p.add_argument("--out", default="results", help="директоријум за логове")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "demo":
        run_demo(args)
        return
    raise NotImplementedError(f"метод „{args.method}” још није имплементиран")


if __name__ == "__main__":
    main()
