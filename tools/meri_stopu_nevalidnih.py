"""Мерна скрипта за режим фиксног n (DECISIONS §26.3) — не мења подразумеване вредности,
само мери и исписује. Резултати иду у docs/IZVESTAJ_FIKSNO_N.md.

Мери, за оператор преповезивања (`reconnect_node`), раздвојено по подпотезу и по k:

  (а) валидна и здрава — пролази validate(), simulate() враћа путању, путања није
      дегенерисана
  (б) одбијена од validate() (нпр. чвор 1 више није предак трагача)
  (в) прошла validate(), али путања дегенерисана или simulate() врати None
  (г) не може да се канонизује — to_sequence подигне InvalidTopology (важи за ОБА
      подпотеза — „промена ослонца" никад не зове to_sequence сама, али производи
      резултате које to_sequence одбија, в. DECISIONS §26.3)

Приоритет класификације: to_sequence → validate() → simulate() + провера полупречника
путање. Ако сам подпотез директно врати None (структурно недостижно за „промена ослонца";
могуће за „обртање знака" при round-trip-у) — рачуна се такође у (г).

Иста мера се ради и за затечену мешавину оператора при променљивом n (add_node А/Б,
delete_node) као референца, плус однос ρ_ново/ρ_старо за преповезани ген подпотеза
„промена ослонца" (за warm-start идентитет-мапу у bilevel-у).

Употреба:
    .venv/bin/python tools/meri_stopu_nevalidnih.py
    .venv/bin/python tools/meri_stopu_nevalidnih.py --trials 500 --n 6 8
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pantograph.config import DEFAULT_CONFIG  # noqa: E402
from pantograph.genome import to_sequence
from pantograph.operators import (
    _reconnect_change_anchor,
    _reconnect_flip_sign,
    add_node,
    delete_node,
    random_initial_genome,
)
from pantograph.simulator import simulate
from pantograph.validation import InvalidTopology, validate

N_CURVE = 90  # мала резолуција — мери се исход потеза, не тачност механизма
RADIUS_EPS = 1e-9  # праг за „дегенерисана путања" (нулти полупречник)


def _make_valid_genome(n_nodes: int, rng: np.random.Generator, attempts: int = 200):
    """Понавља `random_initial_genome` до валидне (README 2.3, „Остало")."""
    for _ in range(attempts):
        result = random_initial_genome(n_nodes, rng)
        if result is not None:
            return result
    raise RuntimeError(f"{attempts} покушаја без валидне почетне јединке за n={n_nodes}.")


def classify(topology, coords) -> str:
    """Врати "а"|"б"|"в"|"г" по приоритету to_sequence → validate() → simulate()."""
    try:
        to_sequence(topology, coords)
    except InvalidTopology:
        return "г"
    try:
        order = validate(topology)
    except InvalidTopology:
        return "б"
    path = simulate(topology, coords, order, N_CURVE)
    if path is None:
        return "в"
    center = (path.min(axis=0) + path.max(axis=0)) / 2.0
    radius = float(np.linalg.norm(path - center, axis=1).max())
    if radius < RADIUS_EPS:
        return "в"
    return "а"


def measure_reconnect(n_nodes: int, trials_per_k: int, seed: int):
    """Врати (по_k_и_подпотезу: Counter, ρ_однос_лист) за дато n."""
    rng = np.random.default_rng(seed)
    per_bucket: dict[tuple[str, int], Counter] = {}
    rho_ratios: list[tuple[float, float]] = []

    for k in range(3, n_nodes):
        for submove in ("anchor", "sign"):
            counts: Counter = Counter()
            for _ in range(trials_per_k):
                topology, coords = _make_valid_genome(n_nodes, rng)
                if submove == "anchor":
                    result = _reconnect_change_anchor(topology, coords, rng, k)
                else:
                    result = _reconnect_flip_sign(topology, coords, rng, k)

                if result is None:
                    counts["г"] += 1
                    continue

                new_topology, new_coords = result
                bucket = classify(new_topology, new_coords)
                counts[bucket] += 1

                if submove == "anchor" and bucket != "г":
                    try:
                        parent_seq = to_sequence(topology, coords)
                        child_seq = to_sequence(new_topology, new_coords)
                    except InvalidTopology:
                        continue  # не би требало — већ класификовано као не-г
                    idx = k - 3
                    pg, cg = parent_seq.genes[idx], child_seq.genes[idx]
                    rho_ratios.append((cg.rho_a / pg.rho_a, cg.rho_b / pg.rho_b))

            per_bucket[(submove, k)] = counts

    return per_bucket, rho_ratios


def measure_reference(trials: int, seed: int):
    """Иста мера за затечену мешавину оператора при променљивом n, као референца."""
    rng = np.random.default_rng(seed)
    per_operator: dict[str, Counter] = {"add_A": Counter(), "add_B": Counter(), "delete": Counter()}

    for _ in range(trials):
        n_nodes = int(rng.choice(DEFAULT_CONFIG.n_init_choices))
        topology, coords = _make_valid_genome(n_nodes, rng)

        result = add_node(topology, coords, rng, mode="A")
        per_operator["add_A"][classify(*result) if result is not None else "г"] += 1

        result = add_node(topology, coords, rng, mode="B")
        per_operator["add_B"][classify(*result) if result is not None else "г"] += 1

        if topology.n_nodes >= 5:
            result = delete_node(topology, coords, rng)
            per_operator["delete"][classify(*result) if result is not None else "г"] += 1

    return per_operator


def _print_bucket_table(title: str, rows: dict, trials_per_row: int) -> None:
    print(f"\n{title}")
    print(f"{'':20} {'а':>8} {'б':>8} {'в':>8} {'г':>8}   (n={trials_per_row})")
    for label, counts in rows.items():
        pct = {b: 100.0 * counts.get(b, 0) / trials_per_row for b in "абвг"}
        print(
            f"{str(label):20} "
            f"{pct['а']:7.1f}% {pct['б']:7.1f}% {pct['в']:7.1f}% {pct['г']:7.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--trials", type=int, default=300, help="покушаја по (k, подпотез)")
    parser.add_argument("--ref-trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for n_nodes in args.n:
        per_bucket, rho_ratios = measure_reconnect(n_nodes, args.trials, args.seed)
        labeled = {f"{submove}, k={k}": counts for (submove, k), counts in per_bucket.items()}
        _print_bucket_table(f"=== reconnect_node, n={n_nodes} ===", labeled, args.trials)

        if rho_ratios:
            ratios = np.array(rho_ratios).reshape(-1)  # ρ_a и ρ_b заједно
            print(
                f"\nρ_ново/ρ_старо (преповезани ген, подпотез промена ослонца, n={n_nodes}, "
                f"{len(ratios)} мерења): "
                f"мин={ratios.min():.4g}  медијана={np.median(ratios):.4g}  макс={ratios.max():.4g}"
            )

    reference = measure_reference(args.ref_trials, args.seed)
    _print_bucket_table("=== референца: мешавина при променљивом n ===", reference, args.ref_trials)


if __name__ == "__main__":
    main()
