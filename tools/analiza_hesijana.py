"""Мери условљеност Chamfer пејзажа по геометријском вектору `x`, скелет по скелет.

Питање: да ли скелет (дискретни део решења) мења СТРУКТУРУ спреге међу континуалним
променљивим, а не само положај оптимума. То је Тип-IV интеракција из
Ong, Shirakawa, Akimoto (2026), arXiv:2606.12885 — класа за коју рад показује да је
методи са производ-расподелом не могу научити.

Поступак, по покретању из `results/runs/`:
  1. `best_genome` из лога → `to_sequence` → скелет, активне позиције, замрзнуте ρ, x*
  2. канонизација размере: цео механизам се скалира тако да полупречник путање буде 1
     (фитнес је на то тачно инваријантан, DECISIONS §17 — али Хесијан НИЈЕ, па без овог
     корака поређење међу покретањима мери размеру уместо облика пејзажа)
  3. централне коначне разлике → Хесијан по x
  4. пројекција ван познатог тачно равног правца (униформна размера) → кондициони број

Излаз: `results/hesijan/nalaz.json` + табела на stdout.
"""

import json
import os
import sys
from glob import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pantograph.config import DEFAULT_CONFIG
from pantograph.curve import TargetCurve, center_and_radius
from pantograph.experiment import RunLog
from pantograph.genome import from_sequence, to_sequence
from pantograph.geometry_vector import (
    active_positions,
    evaluate_vector,
    skeleton_of,
    to_sequence_from_x,
)
from pantograph.simulator import CallCounter, simulate
from pantograph.validation import InvalidTopology, validate

N_FITNESS = 720
INVALID_ABOVE = 1e6          # валидан Chamfer је реда 1e-5..1e-2; све изнад је казна
CFG = DEFAULT_CONFIG


def core_x(seq, active):
    """`to_x` али без увоза — вектор [q1x,q1y,q2x,q2y, ρ_a,ρ_b,...] за активне позиције."""
    parts = [seq.core[1], seq.core[2]]
    for k in active:
        gene = seq.genes[k - 3]
        parts.append(np.array([gene.rho_a, gene.rho_b]))
    return np.concatenate(parts)


def rebuild(x, skel, frozen, active):
    seq = to_sequence_from_x(x, skel, frozen, active)
    return from_sequence(seq)


def path_radius(x, skel, frozen, active):
    res = rebuild(x, skel, frozen, active)
    if res is None:
        return None
    topo, coords = res
    try:
        order = validate(topo)
    except InvalidTopology:
        return None
    path = simulate(topo, coords, order, N_FITNESS, CFG)
    if path is None:
        return None
    _c, r = center_and_radius(path)
    return r


def tracer_depth(skel):
    """Дужина најдужег ланца зависности од трагача до језгра (0,1,2)."""
    depth = {}

    def d(node):
        if node < 3:
            return 0
        if node in depth:
            return depth[node]
        a, b, _s = skel.genes[node - 3]
        depth[node] = 1 + max(d(a), d(b))
        return depth[node]

    return d(skel.n_nodes - 1)


def hessian(f, x, h):
    """Централне коначне разлике; враћа (H, ok) — ok=False ако је стенцил изашао из изводљивог."""
    d = len(x)
    f0 = f(x)
    if f0 > INVALID_ABOVE:
        return None, False
    H = np.zeros((d, d))
    fp, fm = np.zeros(d), np.zeros(d)
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        fp[i] = f(x + e)
        fm[i] = f(x - e)
        if fp[i] > INVALID_ABOVE or fm[i] > INVALID_ABOVE:
            return None, False
        H[i, i] = (fp[i] - 2 * f0 + fm[i]) / h**2
    for i in range(d):
        for j in range(i + 1, d):
            ei, ej = np.zeros(d), np.zeros(d)
            ei[i] = h
            ej[j] = h
            vals = [f(x + ei + ej), f(x + ei - ej), f(x - ei + ej), f(x - ei - ej)]
            if max(vals) > INVALID_ABOVE:
                return None, False
            H[i, j] = H[j, i] = (vals[0] - vals[1] - vals[2] + vals[3]) / (4 * h**2)
    return H, True


def reduced_spectrum(H, v):
    """Собствене вредности Хесијана на ортогоналном комплементу правца `v`."""
    d = H.shape[0]
    v = v / np.linalg.norm(v)
    # ортонормална база комплемента преко QR
    A = np.eye(d) - np.outer(v, v)
    Q, _R = np.linalg.qr(A)
    B = Q[:, : d - 1]
    # осигурај да је B заиста ортогонално на v
    B = B - np.outer(v, v @ B)
    B, _ = np.linalg.qr(B)
    Hr = B.T @ H @ B
    return np.linalg.eigvalsh((Hr + Hr.T) / 2)


def analyze(log_path, h):
    log = RunLog.load(log_path)
    if log.best_genome is None:
        return None
    curve_file = log.curve if log.curve.endswith(".txt") else f"data/curves/{log.curve}.txt"
    target = TargetCurve.from_file(curve_file)

    # Чвор 0 се враћа у (0,0): `run.py` на крају примени `place_on_target` (транслација+размера)
    # на координате најбоље јединке, па снимљени геном више нема чвор 0 у координатном почетку.
    # База вектора `x` (§22) подразумева да јесте — без овог корака `to_x`/`to_sequence_from_x`
    # тихо помере чвор 0 у односу на остале и механизам се распадне.
    coords0 = log.best_genome.coords - log.best_genome.coords[0]
    try:
        seq = to_sequence(log.best_genome.topology, coords0)
    except InvalidTopology as exc:
        return {"run": os.path.basename(os.path.dirname(log_path)), "status": f"to_sequence: {exc}"}

    skel = skeleton_of(seq)
    active = active_positions(skel)
    frozen = {
        k: (seq.genes[k - 3].rho_a, seq.genes[k - 3].rho_b)
        for k in range(3, skel.n_nodes)
        if k not in set(active)
    }
    x = core_x(seq, active)

    counter = CallCounter()

    def f(z):
        return evaluate_vector(z, skel, frozen, active, target, N_FITNESS, counter, CFG)

    f_raw = f(x)
    r = path_radius(x, skel, frozen, active)
    name = os.path.basename(os.path.dirname(log_path))
    curve_name = os.path.splitext(os.path.basename(log.curve))[0]
    if r is None or r < 1e-9:
        return {"run": name, "status": "путања дегенерисана"}

    xc = x.copy()
    xc[:4] = xc[:4] / r
    f_canon = f(xc)

    for h_try in (h, h / 3.0, h / 10.0, h / 30.0):
        H, ok = hessian(f, xc, h_try)
        if ok:
            h = h_try
            break
    if not ok:
        return {
            "run": name, "curve": curve_name, "method": log.method, "seed": log.seed,
            "status": "стенцил излази из изводљивог (оптимум на рубу ограничења)",
            "d": len(x), "n_nodes": skel.n_nodes, "depth": tracer_depth(skel),
            "f": f_canon,
        }

    v_scale = np.zeros(len(xc))
    v_scale[:4] = xc[:4]
    eig_full = np.linalg.eigvalsh((H + H.T) / 2)
    lam_scale = float(v_scale @ H @ v_scale / (v_scale @ v_scale))
    eig_red = reduced_spectrum(H, v_scale)

    absr = np.abs(eig_red)
    kappa = float(absr.max() / absr.min()) if absr.min() > 0 else float("inf")
    # κ преко λ_min је шумом одређен (најмања собствена вредност је на нивоу грешке коначних
    # разлика) — отуда и робуснија варијанта преко доњег квартила спектра.
    kappa_q = float(absr.max() / np.quantile(absr, 0.25))
    # Ефективан број праваца који носе кривину: (Σ|λ|)² / Σλ². За изотропан пејзаж = d,
    # за пејзаж са једним доминантним правцем → 1.
    eff_dim = float(absr.sum() ** 2 / (absr**2).sum())

    # Удео енергије ван дијагонале: 0 = потпуно раздвојив пејзаж (свака координата за себе),
    # близу 1 = јака спрега међу континуалним променљивим. Ово је мера коју метод са
    # производ-расподелом НЕ може да научи.
    off = H - np.diag(np.diag(H))
    off_energy = float(np.linalg.norm(off) / np.linalg.norm(H)) if np.linalg.norm(H) > 0 else float("nan")

    return {
        "run": name, "curve": curve_name, "method": log.method, "seed": log.seed,
        "status": "ок",
        "d": int(len(xc)), "n_nodes": int(skel.n_nodes),
        "aktivnih": int(len(active)), "mrtvih": int(skel.n_nodes - 3 - len(active)),
        "depth": int(tracer_depth(skel)),
        "f": float(f_canon), "f_raw": float(f_raw), "radius": float(r),
        "lam_abs_max": float(absr.max()), "lam_abs_min": float(absr.min()),
        "n_negativnih": int((eig_red < 0).sum()),
        "kappa": kappa, "kappa_q": kappa_q, "eff_dim": eff_dim,
        "off_energy": off_energy, "h": h,
        "lam_scale_rel": float(abs(lam_scale) / max(abs(eig_full).max(), 1e-300)),

        "eig": [float(e) for e in eig_red],
        "pozivi": int(counter.calls),
    }


def main():
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
    out = sys.argv[2] if len(sys.argv) > 2 else "results/hesijan/nalaz.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    paths = sorted(glob("results/runs/*/log.json"))
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else len(paths)
    paths = paths[start:end]
    rows = []
    for p in paths:
        try:
            row = analyze(p, h)
        except Exception as exc:  # мерење не сме да падне на једном покретању
            row = {"run": os.path.basename(os.path.dirname(p)), "status": f"грешка: {exc!r}"}
        if row is not None:
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    old = []
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            old = json.load(fh).get("rows", [])
    known = {r["run"] for r in rows}
    merged = [r for r in old if r["run"] not in known] + rows
    merged.sort(key=lambda r: r["run"])
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"h": h, "rows": merged}, fh, ensure_ascii=False, indent=2)
    print(f"\nуписано: {out}  ({len(rows)} покретања)")


if __name__ == "__main__":
    main()
