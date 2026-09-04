"""CLI улазна тачка: покретање једног експеримента (README Део III) и demo провера
(BASELINE_SPEC корак Д).

Примери:
    python run.py demo --curve data/curves/circle.txt
    python run.py run --method baseline --curve data/curves/circle.txt --seed 1
    python run.py run --method baseline --curve data/curves/circle.txt --seed 1 --quick --snapshot-every 10
    python run.py show results/latest
    python run.py replay results/latest
"""

import argparse
import dataclasses
import json
import os
import platform
from datetime import datetime

import numpy as np

from pantograph.baseline import evolve
from pantograph.config import DEFAULT_CONFIG, QUICK_CONFIG
from pantograph.curve import TargetCurve, apply_similarity_transform, place_on_target
from pantograph.experiment import (
    RunLog,
    _genome_to_dict,
    curve_file_hash,
    git_commit_hash,
)
from pantograph.fitness import chamfer
from pantograph.genome import Genome, Topology
from pantograph.progress import ProgressReporter
from pantograph.simulator import simulate
from pantograph.validation import validate
from pantograph.visualization import (
    animate,
    plot_comparison,
    plot_error_curve,
    plot_mechanism,
    tracer_path,
)


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


def _run_directory(out: str, method: str, curve: str, seed: int) -> str:
    """Фолдер по покретању: `results/<временска ознака>_<метод>_<крива>_seed<seed>`
    (козметика 31.08.) — уместо старе равне путање `results/{method}_seed{seed}.json`."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    curve_stem = os.path.splitext(os.path.basename(curve))[0]
    return os.path.join(out, f"{timestamp}_{method}_{curve_stem}_seed{seed}")


def _update_latest_link(out: str, run_dir: str) -> None:
    """Симболичка веза `results/latest` → фолдер последњег покретања."""
    latest = os.path.join(out, "latest")
    if os.path.islink(latest) or os.path.exists(latest):
        os.remove(latest)
    os.symlink(os.path.basename(run_dir), latest)


def _write_command_file(run_dir: str, command: str, curve: str) -> None:
    """`command.txt` — тачна команда за поновно покретање плус git commit, хеш криве и
    верзија Python-а (README 4.3, козметика 31.08.)."""
    path = os.path.join(run_dir, "command.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(command + "\n\n")
        f.write(f"git commit: {git_commit_hash()}\n")
        f.write(f"хеш криве (sha256): {curve_file_hash(curve)}\n")
        f.write(f"python: {platform.python_version()}\n")


def run_experiment(args: argparse.Namespace) -> None:
    """Покреће baseline ГА до краја (README 3.3, BASELINE_SPEC §9) и снима резултат у
    фолдер по покретању (козметика 31.08.): `log.json`, `best_genome.json`,
    `error_curve.png`, `best_path.png`, `mechanism.png`, `snapshots/`, `command.txt`.

    `bilevel` остаје `NotImplementedError` — `bilevel.py` још не постоји.
    """
    if args.method != "baseline":
        raise NotImplementedError(f"метод „{args.method}” још није имплементиран")

    config = QUICK_CONFIG if args.quick else DEFAULT_CONFIG
    if args.max_link_ratio is not None:
        # Config је frozen — вредност се уводи преко dataclasses.replace (PROMPT_ITERACIJA
        # целина В). Подразумевано (None) значи: узми вредност из профила без измене.
        config = dataclasses.replace(config, max_link_to_radius_ratio=args.max_link_ratio)
    budget = args.budget if args.budget is not None else config.total_budget
    population = args.population if args.population is not None else config.population_size

    target = TargetCurve.from_file(args.curve)

    run_dir = _run_directory(args.out, args.method, args.curve, args.seed)
    os.makedirs(run_dir, exist_ok=True)

    reporter = ProgressReporter(
        total_budget=budget,
        population_size=population,
        target=target,
        run_dir=run_dir,
        log_every=args.log_every,
        snapshot_every=args.snapshot_every,
        snapshot_n=720,
        config=config,
    )
    reporter.start(
        method=args.method,
        curve=args.curve,
        seed=args.seed,
        n_schedule=config.n_schedule,
        git_commit=git_commit_hash(),
        profile="quick" if args.quick else "",
    )

    log = evolve(
        target,
        budget=budget,
        seed=args.seed,
        population_size=population,
        config=config,
        reporter=reporter,
    )

    # Сличносна трансформација (README 1.5, DECISIONS §17) — ОДМАХ после `evolve`, ПРЕ
    # репортера: `reporter.final_snapshot`/`finish` и снимци читају `log.best_genome`, па
    # трансформација мора да им претходи да завршни снимак и резиме описују исти
    # (трансформисани) механизам као `best_genome.json`/`mechanism.png`/`best_path.png`.
    # Ван буџета, једном по покретању.
    pose = None
    if log.best_genome is not None:
        best_path = tracer_path(log.best_genome, n=720, config=config)
        if best_path is not None:
            tx, ty, angle, scale = place_on_target(best_path)
            log.best_genome.coords = apply_similarity_transform(
                log.best_genome.coords, tx, ty, angle, scale
            )
            pose = {"tx": tx, "ty": ty, "angle_rad": angle, "scale": scale}

    reporter.final_snapshot(log)
    reporter.finish(log)

    log.save(os.path.join(run_dir, "log.json"))
    if log.best_genome is not None:
        genome_dict = _genome_to_dict(log.best_genome)
        if pose is not None:
            genome_dict["pose"] = pose
        with open(os.path.join(run_dir, "best_genome.json"), "w", encoding="utf-8") as f:
            json.dump(genome_dict, f, ensure_ascii=False, indent=2)

    plot_error_curve(
        [log], title=f"{log.method}, seed={log.seed}",
        save=os.path.join(run_dir, "error_curve.png"),
    )
    if log.best_genome is not None:
        best_path = tracer_path(log.best_genome, n=720, config=config)
        if best_path is not None:
            plot_comparison(
                best_path, target, title="Путања трагача наспрам циљне криве",
                save=os.path.join(run_dir, "best_path.png"),
            )
        plot_mechanism(
            log.best_genome, title="Механизам најбоље јединке",
            save=os.path.join(run_dir, "mechanism.png"),
        )

    command = (
        f"python run.py run --method {args.method} --curve {args.curve} "
        f"--seed {args.seed} --budget {budget} --population {population} "
        f"--max-link-ratio {config.max_link_to_radius_ratio}"
        + (" --quick" if args.quick else "")
        + (f" --log-every {args.log_every}" if args.log_every != 1 else "")
        + (f" --snapshot-every {args.snapshot_every}" if args.snapshot_every else "")
    )
    _write_command_file(run_dir, command, args.curve)
    _update_latest_link(args.out, run_dir)


def run_show(args: argparse.Namespace) -> None:
    """Учитава сачувано покретање и приказује механизам, путању и криву грешке — БЕЗ
    поновног тренирања (козметика 31.08.)."""
    log = RunLog.load(os.path.join(args.run_dir, "log.json"))
    print(f"Метод: {log.method}, seed={log.seed}, циљна крива: {log.curve}")
    if log.records:
        print(f"Генерација: {len(log.records)}, потрошено позива: {log.records[-1].calls_spent}")
    print(f"Коначна грешка (Chamfer): {log.final_error:.6g}")

    if log.best_genome is None:
        print("Нема сачуване најбоље јединке у овом логу.")
        return
    print(
        f"Најбоља јединка: n_nodes={log.best_genome.topology.n_nodes}, "
        f"ивица={len(log.best_genome.topology.edges)}"
    )

    plot_mechanism(log.best_genome, title="Механизам најбоље јединке")

    target = None
    if log.curve:
        try:
            target = TargetCurve.from_file(log.curve)
        except OSError:
            print(f"Циљна крива „{log.curve}” није нађена — прескачем путању и поређење.")
    if target is not None:
        best_path = tracer_path(log.best_genome, n=720)
        if best_path is not None:
            plot_comparison(best_path, target, title="Путања трагача наспрам циљне криве")
        else:
            print("Најбоља јединка није решива на пуном обртају (circuit defect).")

    if log.records:
        plot_error_curve([log], title=f"{log.method}, seed={log.seed}")


def run_replay(args: argparse.Namespace) -> None:
    """Анимира најбољу јединку сачуваног покретања. Подразумевано снима
    `best_animation.gif`; `--live` приказује уживо (козметика 31.08.)."""
    log = RunLog.load(os.path.join(args.run_dir, "log.json"))
    if log.best_genome is None:
        print("Нема сачуване најбоље јединке за анимацију.")
        return

    target = None
    if log.curve:
        try:
            target = TargetCurve.from_file(log.curve)
        except OSError:
            target = None

    if args.live:
        animate(log.best_genome, n=args.n, target=target)
        return

    out_path = os.path.join(args.run_dir, "best_animation.gif")
    animate(log.best_genome, n=args.n, save=out_path, target=target)
    print(f"Анимација сачувана у {out_path}")


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
    run_p.add_argument("--budget", type=int, default=None,
                        help="максималан број позива симулатора; подразумевано из профила "
                             "(--quick или обичан, README 3.3, BASELINE_SPEC §7)")
    run_p.add_argument("--population", type=int, default=None,
                        help="величина популације; подразумевано из профила (BASELINE_SPEC §7)")
    run_p.add_argument("--seed", type=int, default=0, help="семе случајности (README 4.3)")
    run_p.add_argument("--out", default="results", help="директоријум за фолдере покретања")
    run_p.add_argument("--quick", action="store_true",
                        help="демонстрациони профил QUICK_CONFIG — НИЈЕ мерење за H1 (config.py)")
    run_p.add_argument("--log-every", type=int, default=1,
                        help="испиши сваку K-ту генерацију у конзоли (подразумевано све)")
    run_p.add_argument("--snapshot-every", type=int, default=0,
                        help="сними PNG најбоље јединке сваких K генерација (0 = искључено)")
    run_p.add_argument("--max-link-ratio", type=float, default=None,
                        help="горња граница односа largest_link/path_radius; подразумевано "
                             "из профила (5.0 у DEFAULT_CONFIG, DECISIONS §18/§20)")

    show_p = sub.add_parser("show", help="прикажи сачувано покретање без поновног тренирања")
    show_p.add_argument("run_dir", help="фолдер покретања (нпр. results/latest)")

    replay_p = sub.add_parser("replay", help="анимирај најбољу јединку сачуваног покретања")
    replay_p.add_argument("run_dir", help="фолдер покретања (нпр. results/latest)")
    replay_p.add_argument("--live", action="store_true",
                           help="прикажи уживо (plt.show) уместо да снимиш GIF")
    replay_p.add_argument("--n", type=int, default=200,
                           help="број корака анимације по пуном обртају crank-а")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "demo":
        run_demo(args)
    elif args.command == "show":
        run_show(args)
    elif args.command == "replay":
        run_replay(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
