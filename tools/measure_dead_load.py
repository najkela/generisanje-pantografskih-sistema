#!/usr/bin/env python3
"""Мери удео координатних мутација који је пре исправке од 02.10. одлазио у мртве чворове.

Пре исправке `operators.mutate_coords` је мутирао све чворове `1..n−1`, укључујући оне ван
предачког стабла трагача; двонивовски метод их изоставља из ЦМА-ЕС вектора (`DECISIONS §22`
А2). `§27.6` процењује да је у Е2 око 40% координатних мутација одлазило у чворове који по
конструкцији не могу да промене фитнес. Овај алат ту процену замењује мерењем.

Мери се НА ИСПРАВЉЕНОМ коду: сваки позив `mutate_coords` пријави колико је чворова било у
опсегу `1..n−1` и колико их је од тога живо. Удео мртвих је тачно оно што је стари кôд
трошио у празно. Профил је Е2 — режим поређења, без `--fiksno-n`.

Употреба:
    .venv/bin/python tools/measure_dead_load.py
    .venv/bin/python tools/measure_dead_load.py --budget 12000 --seeds 1 2 3
    .venv/bin/python tools/measure_dead_load.py --curves egg superellipse
"""

import argparse
import dataclasses
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pantograph import operators
from pantograph.baseline import evolve
from pantograph.config import POREDJENJE_CONFIG
from pantograph.curve import TargetCurve
from pantograph.validation import InvalidTopology

DEFAULT_CURVES = ("ellipse", "egg", "limacon", "cassini", "superellipse")


class DeadLoadTally:
    """Сабира слотове координатне мутације: укупно наспрам мртвих, по позиву `mutate_coords`.

    „Слот" је један чвор у опсегу `1..n−1` при једном позиву — тачно оно за шта је стари кôд
    бацао новчић. Удео мртвих слотова је удео узорака који су се трошили у празно.
    """

    def __init__(self) -> None:
        self.total_slots = 0
        self.dead_slots = 0
        self.calls = 0
        self.unsolvable_calls = 0
        self.live_fractions: list[float] = []
        # Допуњава `measure_one` после покретања — иде у исти ред извештаја.
        self.seconds = 0.0
        self.final_error = float("nan")
        self.unique_skeletons = 0
        self.calls_spent = 0

    def observe(self, topology) -> None:
        slots = topology.n_nodes - 1
        try:
            live = len(operators.live_coordinate_nodes(topology))
        except InvalidTopology:
            # Структурно неважећа топологија — исправка тада мутира све (в. `mutate_coords`),
            # дакле нема мртвих слотова за пријаву, али се позив броји засебно.
            self.unsolvable_calls += 1
            live = slots
        self.calls += 1
        self.total_slots += slots
        self.dead_slots += slots - live
        self.live_fractions.append(live / slots if slots else 1.0)

    @property
    def dead_share(self) -> float:
        return self.dead_slots / self.total_slots if self.total_slots else 0.0


def measure_one(curve_path: str, seed: int, budget: int) -> DeadLoadTally:
    """Једно покретање монолитног метода под профилом Е2; враћа сабирач."""
    target = TargetCurve.from_file(curve_path)
    config = dataclasses.replace(POREDJENJE_CONFIG, total_budget=budget)
    tally = DeadLoadTally()

    original = operators.mutate_coords

    def observed_mutate_coords(topology, coords, k, per_node_prob, rng):
        tally.observe(topology)
        return original(topology, coords, k, per_node_prob, rng)

    # `baseline` увози `mutate_coords` по имену, па се замена мора урадити ТАМО — замена у
    # `operators` не би ништа променила за већ увезено име.
    from pantograph import baseline as baseline_module

    original_bound = baseline_module.mutate_coords
    baseline_module.mutate_coords = observed_mutate_coords
    try:
        start = time.perf_counter()
        log = evolve(
            target, budget=budget, seed=seed,
            population_size=config.population_size, config=config,
        )
        tally.seconds = time.perf_counter() - start
    finally:
        baseline_module.mutate_coords = original_bound

    tally.final_error = log.final_error
    tally.unique_skeletons = log.unique_skeletons
    tally.calls_spent = log.records[-1].calls_spent if log.records else 0
    return tally


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Удео координатних мутација који је одлазио у мртве чворове (§27.6)."
    )
    parser.add_argument("--curves", nargs="+", default=list(DEFAULT_CURVES),
                        help="имена кривих из data/curves (без .txt)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1],
                        help="семена покретања")
    parser.add_argument("--budget", type=int, default=12_000,
                        help="буџет позива симулатора по покретању")
    args = parser.parse_args()

    print(f"Профил Е2 (режим поређења, променљиво n), буџет {args.budget} по покретању.")
    print(f"Криве: {', '.join(args.curves)} | семена: {args.seeds}")
    print()
    print(f"  {'крива':<16}{'seed':>5}{'позива':>9}{'слотова':>10}{'мртвих':>9}{'удео':>9}"
          f"{'скелета':>10}{'грешка':>12}{'трајање':>10}")
    print("  " + "─" * 90)

    all_shares: list[float] = []
    grand_total = grand_dead = 0
    unsolvable = 0
    for curve in args.curves:
        path = os.path.join("data", "curves", f"{curve}.txt")
        for seed in args.seeds:
            tally = measure_one(path, seed, args.budget)
            all_shares.append(tally.dead_share)
            grand_total += tally.total_slots
            grand_dead += tally.dead_slots
            unsolvable += tally.unsolvable_calls
            print(f"  {curve:<16}{seed:>5}{tally.calls:>9}{tally.total_slots:>10}"
                  f"{tally.dead_slots:>9}{100 * tally.dead_share:>8.1f}%"
                  f"{tally.unique_skeletons:>10}{tally.final_error:>12.3e}"
                  f"{tally.seconds / 60:>9.1f}m", flush=True)

    print("  " + "─" * 90)
    overall = grand_dead / grand_total if grand_total else 0.0
    print(f"  {'СВЕ ЗАЈЕДНО':<16}{'':>5}{'':>9}{grand_total:>10}{grand_dead:>9}"
          f"{100 * overall:>8.1f}%")
    print()
    print(f"  медијана по покретању   {100 * statistics.median(all_shares):.1f}%")
    print(f"  распон                  {100 * min(all_shares):.1f}% – {100 * max(all_shares):.1f}%")
    print(f"  нерешивих топологија    {unsolvable} позива (мутирају се сви чворови, в. `mutate_coords`)")
    print()
    print("  Толики удео координатних мутација је пре исправке од 02.10. одлазио у чворове")
    print("  који по конструкцији не могу да промене фитнес (DECISIONS §27.6).")


if __name__ == "__main__":
    main()
