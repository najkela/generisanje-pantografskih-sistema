"""CLI улазна тачка: покретање једног експеримента (README Део III).

Примери:
    python run.py --method baseline --curve data/curves/circle.txt --seed 1
    python run.py --method bilevel --curve data/curves/jansen.txt --budget 150 --seed 1
"""

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Генерисање пантографских механизама")
    p.add_argument("--method", choices=["baseline", "bilevel"], required=True,
                   help="baseline = чист ГА (README 2.2); bilevel = ГА + ЦМА-ЕС (README 2.3)")
    p.add_argument("--curve", required=True, help="фајл са тачкама циљне криве")
    p.add_argument("--budget", type=int, default=150,
                   help="максималан број позива симулатора (README 3.3)")
    p.add_argument("--seed", type=int, default=0, help="семе случајности (README 4.3)")
    p.add_argument("--out", default="results", help="директоријум за логове")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(f"метод „{args.method}” још није имплементиран")


if __name__ == "__main__":
    main()
