#!/usr/bin/env python3
"""Мери накнаду у времену коју уводи бројач јединствених скелета (OPEN_QUESTIONS 9.2).

`baseline.evolve` од 02.10. хешира скелет сваке стварно оцењене јединке, преко
`skeleton_key(to_sequence(...))`. Налог `PROMPT_MRTAV_TERET.md` целина А тражи да се накнада
измери и, ако прелази ~2% трајања, да се предложи јефтинија варијанта.

Мери се ДИРЕКТНО — акумулирано време у самом хеширању наспрам укупног трајања покретања — а
не разликом два покретања: два пуна прогона се разликују неколико процената само од шума
машине, па разлика не би могла да разреши ефекат реда 1%.

Употреба:
    .venv/bin/python tools/measure_skeleton_overhead.py
    .venv/bin/python tools/measure_skeleton_overhead.py --curve data/curves/egg.txt --budget 12000
"""

import argparse
import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pantograph import baseline
from pantograph.config import POREDJENJE_CONFIG
from pantograph.curve import TargetCurve


class Tally:
    """Сабирач: укупно потрошено време и број позива. Држи га омотач око мерене функције."""

    def __init__(self) -> None:
        self.seconds = 0.0
        self.calls = 0

    def add(self, seconds: float) -> None:
        self.seconds += seconds
        self.calls += 1


def measure(curve_path: str, seed: int, budget: int) -> dict[str, float]:
    """Једно покретање монолитног метода, са мереним хеширањем. Враћа измерене бројеве.

    `to_sequence` се мери заједно са `skeleton_key` зато што је то целина коју је бројач
    додао — `skeleton_key` сам је само преписивање торке.
    """
    target = TargetCurve.from_file(curve_path)
    config = dataclasses.replace(POREDJENJE_CONFIG, total_budget=budget)

    original_key = baseline.skeleton_key
    original_to_sequence = baseline.to_sequence
    key_tally, sequence_tally = Tally(), Tally()

    def timed_key(seq):
        start = time.perf_counter()
        try:
            return original_key(seq)
        finally:
            key_tally.add(time.perf_counter() - start)

    def timed_to_sequence(topology, coords):
        # `to_sequence` се у `baseline` зове и из `crossover` (двапут по детету), не само из
        # бројача — мери се свако позивање, а бројачу се ниже припише само његов удео.
        start = time.perf_counter()
        try:
            return original_to_sequence(topology, coords)
        finally:
            sequence_tally.add(time.perf_counter() - start)

    baseline.skeleton_key = timed_key
    baseline.to_sequence = timed_to_sequence
    try:
        start = time.perf_counter()
        log = baseline.evolve(
            target, budget=budget, seed=seed,
            population_size=config.population_size, config=config,
        )
        total = time.perf_counter() - start
    finally:
        baseline.skeleton_key = original_key
        baseline.to_sequence = original_to_sequence

    # Од укупног времена `to_sequence`-а бројачу припада онолико позива колико их је само
    # хеширање направило — остатак би се потрошио и без бројача (укрштање).
    per_call = sequence_tally.seconds / max(sequence_tally.calls, 1)
    overhead = key_tally.seconds + per_call * key_tally.calls
    return {
        "total_seconds": total,
        "overhead_seconds": overhead,
        "overhead_percent": 100.0 * overhead / total if total else 0.0,
        "hash_calls": float(key_tally.calls),
        "unique_skeletons": float(log.unique_skeletons),
        "calls_spent": float(log.records[-1].calls_spent) if log.records else 0.0,
        "generations": float(len(log.records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Накнада бројача јединствених скелета.")
    parser.add_argument("--curve", default="data/curves/egg.txt", help="циљна крива")
    parser.add_argument("--seed", type=int, default=1, help="семе покретања")
    parser.add_argument("--budget", type=int, default=12_000, help="буџет позива симулатора")
    args = parser.parse_args()

    print(f"Мерим накнаду хеширања скелета: {args.curve}, seed={args.seed}, "
          f"буџет={args.budget}, режим поређења.")
    result = measure(args.curve, args.seed, args.budget)

    print()
    print(f"  трајање покретања       {result['total_seconds']:.1f} s")
    print(f"  потрошено позива        {int(result['calls_spent'])}")
    print(f"  генерација              {int(result['generations'])}")
    print(f"  хеширања                {int(result['hash_calls'])}")
    print(f"  јединствених скелета    {int(result['unique_skeletons'])}")
    print(f"  време у хеширању        {result['overhead_seconds']:.2f} s")
    print(f"  НАКНАДА                 {result['overhead_percent']:.2f}% трајања")
    print()
    if result["overhead_percent"] > 2.0:
        print("  Накнада прелази 2% — види налог целина А, тачка 2 (јефтинија варијанта).")
    else:
        print("  Накнада је испод 2% — бројач остаје у пуном облику.")


if __name__ == "__main__":
    main()
