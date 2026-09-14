"""Крива грешка-по-позивима: оба метода на истој оси (DECISIONS §24.2).

Режим поређења тражи да се извештава ЦЕЛА крива за оба метода, не једна бројка — јер је
мерено (docs/NALAZ_06_09..., дијагностика 05.09.) да редослед може да се обрне у току
покретања: на `cassini` је bilevel на baseline-овом стварно потрошеном буџету био лошији,
па га претекао тек у другој половини. Једна бројка на крају то сакрива.

Употреба:
    python tools/kriva_greske.py results/*_seed*          # све што нађе, групише по кривој
    python tools/kriva_greske.py results/run1 results/run2 --out results/poredjenje

Излаз по кривој: `<крива>.png` (медијана преко seed-ова + опсег мин–макс, лог оса по y)
и `<крива>.csv` (иста решетка у бројевима, да иду у рад без поновног читања логова).
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pantograph.experiment import RunLog  # noqa: E402

BOJA = {"baseline": "#c0392b", "bilevel": "#2471a3"}
NAZIV = {"baseline": "baseline (истовремено)", "bilevel": "bilevel (двонивовски)"}


def _ucitaj(run_dirs: list[str]) -> dict[str, dict[str, list[RunLog]]]:
    """Групише логове: крива → метод → листа покретања (по seed-овима)."""
    grupe: dict[str, dict[str, list[RunLog]]] = defaultdict(lambda: defaultdict(list))
    for d in run_dirs:
        putanja = os.path.join(d, "log.json")
        if not os.path.exists(putanja):
            print(f"  прескачем {d} — нема log.json", file=sys.stderr)
            continue
        log = RunLog.load(putanja)
        if not log.calls_curve:
            print(f"  прескачем {d} — нема calls_curve (покретање пре режима поређења)",
                  file=sys.stderr)
            continue
        kriva = os.path.splitext(os.path.basename(log.curve))[0] or "непозната"
        grupe[kriva][log.method].append(log)
    return grupe


def _slozi(logovi: list[RunLog]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Решетка + медијана, мин и макс преко seed-ова. Решетке се секу на најкраћу."""
    duzina = min(len(l.calls_curve) for l in logovi)
    pozivi = np.array([c for c, _ in logovi[0].calls_curve[:duzina]])
    matrica = np.array([[v for _, v in l.calls_curve[:duzina]] for l in logovi])
    return pozivi, np.median(matrica, axis=0), matrica.min(axis=0), matrica.max(axis=0)


def _nacrtaj(kriva: str, po_metodu: dict[str, list[RunLog]], out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    redovi: dict[str, np.ndarray] = {}
    pozivi_ref = None

    for metod in ("baseline", "bilevel"):
        logovi = po_metodu.get(metod, [])
        if not logovi:
            continue
        pozivi, medijana, donja, gornja = _slozi(logovi)
        pozivi_ref = pozivi if pozivi_ref is None else pozivi_ref
        ax.plot(pozivi, medijana, color=BOJA[metod], lw=2,
                label=f"{NAZIV[metod]} — медијана, {len(logovi)} seed-а")
        if len(logovi) > 1:
            ax.fill_between(pozivi, donja, gornja, color=BOJA[metod], alpha=0.18, lw=0)
        redovi[f"{metod}_medijana"] = medijana
        redovi[f"{metod}_min"] = donja
        redovi[f"{metod}_max"] = gornja

    ax.set_yscale("log")
    ax.set_xlabel("потрошено позива симулатора")
    ax.set_ylabel("Chamfer растојање (најбоље до тог тренутка)")
    ax.set_title(f"{kriva} — режим поређења: фиксно N=720, без раног заустављања")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{kriva}.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)

    csv_put = os.path.join(out_dir, f"{kriva}.csv")
    with open(csv_put, "w", newline="", encoding="utf-8") as f:
        pisac = csv.writer(f)
        pisac.writerow(["pozivi"] + list(redovi))
        for i, p in enumerate(pozivi_ref):
            pisac.writerow([int(p)] + [f"{redovi[k][i]:.12g}" for k in redovi])
    print(f"  {kriva}: {png}, {csv_put}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dirs", nargs="+", help="фолдери покретања (results/…)")
    p.add_argument("--out", default="results/poredjenje", help="где иду слике и CSV")
    args = p.parse_args()

    grupe = _ucitaj(args.run_dirs)
    if not grupe:
        raise SystemExit("Ниједно покретање са кривом грешка-по-позивима.")
    for kriva, po_metodu in sorted(grupe.items()):
        nedostaje = {"baseline", "bilevel"} - set(po_metodu)
        if nedostaje:
            print(f"  ПАЖЊА: {kriva} нема {', '.join(sorted(nedostaje))} — цртам шта има")
        _nacrtaj(kriva, po_metodu, args.out)


if __name__ == "__main__":
    main()
