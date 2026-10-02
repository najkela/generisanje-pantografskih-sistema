"""Дијагностика: где дуж циљне криве настаје Chamfer грешка (А наспрам Б, 05.09.).

Питање: да ли решења промашују само на местима велике кривине (углови суперелипсе,
струк касинија) или је грешка размазана — и да ли се оба метода стварно заустављају
на „најближој елипси".

Не мења ништа у пакету; чита само већ снимљене резултате из results/runs_n720/.
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from pantograph.curve import TargetCurve, normalize, center_and_radius
from pantograph.fitness import chamfer
from pantograph.genome import Topology
from pantograph.simulator import simulate
from pantograph.validation import validate

N = 720
KRIVE = ["superellipse", "cassini", "egg", "ellipse", "limacon"]
METODI = ["baseline", "bilevel"]


def putanja_iz_loga(putanja_do_loga):
    d = json.load(open(putanja_do_loga))
    g = d["best_genome"]
    topo = Topology(n_nodes=g["n_nodes"], edges={tuple(e) for e in g["edges"]})
    coords = np.asarray(g["coords"], dtype=float)
    order = validate(topo)
    path = simulate(topo, coords, order, N)
    if path is None:
        return None, d["final_error"]
    center, radius = center_and_radius(path)
    return (path - center) / radius, d["final_error"]


def najbolja_elipsa(target_pts):
    """Најнижи Chamfer који постиже ЧИСТА елипса (однос страна + ротација),
    груба претрага — нормализација уклања скалу, остају два параметра."""
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    naj = (np.inf, None, None)
    cilj = TargetCurve(points=target_pts, tree=cKDTree(target_pts))
    for odnos in np.linspace(0.10, 1.0, 46):
        baza = np.column_stack([np.cos(t), odnos * np.sin(t)])
        for ugao in np.linspace(0, np.pi / 2, 19):
            c, s = np.cos(ugao), np.sin(ugao)
            rot = baza @ np.array([[c, -s], [s, c]])
            v = chamfer(normalize(rot), cilj)
            if v < naj[0]:
                naj = (v, odnos, ugao)
    return naj


red = []
fig, axes = plt.subplots(len(KRIVE), 2, figsize=(13, 3.0 * len(KRIVE)))
for i, kriva in enumerate(KRIVE):
    cilj = TargetCurve.from_file(f"data/curves/{kriva}.txt").at_resolution(N)
    naj_el, odnos, ugao = najbolja_elipsa(cilj.points)

    osa_profil, osa_oblik = axes[i]
    osa_oblik.plot(*np.vstack([cilj.points, cilj.points[:1]]).T, "k-", lw=2.4, label="циљ", zorder=1)

    for metod, boja in zip(METODI, ["#c44", "#36c"]):
        put, greska = putanja_iz_loga(f"results/runs_n720/{metod}_{kriva}_seed1/log.json")
        if put is None:
            continue
        d, _ = cKDTree(put).query(cilj.points)   # за сваку тачку ЦИЉА — колико је далеко најближа тачка путање
        kv = d ** 2
        udeo10 = float(np.sort(kv)[-N // 10:].sum() / kv.sum())
        red.append(dict(kriva=kriva, metod=metod, greska=greska, naj_elipsa=naj_el,
                        odnos_prema_elipsi=greska / naj_el, maks_d=float(d.max()),
                        med_d=float(np.median(d)), udeo_najgorih_10=udeo10))
        osa_profil.plot(np.arange(N) / N, d, color=boja, lw=1.2, label=metod)
        osa_oblik.plot(*np.vstack([put, put[:1]]).T, color=boja, lw=1.0, alpha=0.85, label=metod)

    osa_profil.set_title(f"{kriva} — растојање циљне тачке до путање, по обиму")
    osa_profil.set_xlabel("положај дуж циљне криве (удео обима)")
    osa_profil.legend(fontsize=8)
    osa_oblik.set_aspect("equal")
    osa_oblik.set_title(f"{kriva} — облик")
    osa_oblik.legend(fontsize=8)

plt.tight_layout()
plt.savefig("results/dijagnostika_greske.png", dpi=110)

print(f"{'крива':<14}{'метод':<10}{'Chamfer':>12}{'најбоља елипса':>16}{'однос':>9}"
      f"{'медијана d':>12}{'макс d':>9}{'удео најгорих 10%':>20}")
for r in red:
    print(f"{r['kriva']:<14}{r['metod']:<10}{r['greska']:>12.3e}{r['naj_elipsa']:>16.3e}"
          f"{r['odnos_prema_elipsi']:>9.2f}{r['med_d']:>12.4f}{r['maks_d']:>9.4f}"
          f"{r['udeo_najgorih_10']*100:>19.1f}%")
json.dump(red, open("results/dijagnostika_greske.json", "w"), indent=1)
