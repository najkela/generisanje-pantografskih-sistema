"""Буџет, метрике, семе, логовање и плато-детекција (README 2.4, 3.2, 3.3, 4.3).

Заједничка инфраструктура за baseline и bilevel (BASELINE_SPEC §9, §10 корак 4.1) —
ниједна функција овде не зна ништа о конкретном методу. Метод-специфичну петљу (иницијализација
популације, генерацијска шема) држи сваки метод сам (`baseline.evolve`), позивајући ово исто.
"""

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field, fields as dataclasses_fields

import numpy as np

from .config import Config, DEFAULT_CONFIG
from .genome import Genome, Topology
from .simulator import CallCounter


@dataclass
class Budget:
    """Позив симулатора је јединица трошка, заједничка за оба метода (README 3.3).

    `spent` је computed преко унутрашњег `CallCounter` (симулатор.py) — исти бројач који
    `fitness.evaluate` инкрементира, нема дуплог бројања (план 31.08., питање 1).
    """

    max_calls: int = 150
    counter: CallCounter = field(default_factory=CallCounter)

    def spend(self, amount: int = 1) -> None:
        self.counter.increment(amount)

    @property
    def spent(self) -> int:
        return self.counter.calls

    @property
    def exhausted(self) -> bool:
        return self.counter.calls >= self.max_calls


def plateau_detected(history: list[float], window: int, epsilon: float) -> bool:
    """Клизни прозор `window` и праг `epsilon` над кривом најбољег фитнеса по генерацији.

    Плато = релативно побољшање најбољег резултата унутар последњих `window` уноса је
    мање од `epsilon` (README 2.4). Исти механизам се користи на ТРИ места — динамички N,
    динамички K (bilevel) и критеријум заустављања — прагови се разликују (BASELINE_SPEC §7).
    """
    if len(history) < window:
        return False
    segment = history[-window:]
    start_value = segment[0]
    best_in_window = min(segment)
    if start_value <= 0:
        return True  # већ на нули/дегенерисано — нема простора за даље побољшање
    improvement = (start_value - best_in_window) / start_value
    return improvement < epsilon


def resolution_schedule(
    generation: int,
    history: list[float],
    config: Config = DEFAULT_CONFIG,
) -> int:
    """Динамички N: 90→180→360→720, напредује на плато (README 2.4, BASELINE_SPEC §7).

    Statelessна — сваки позив изнова прође кроз `history` и утврди тренутни ниво преко
    `plateau_detected`; ниједно стање се не памти између позива (план 31.08., питање 2). То
    је оно што омогућава да и bilevel користи ИСТУ функцију са СВОЈОМ листом историје.
    `generation` је информативан (очекивано `== len(history)`), логика гледа само `history`.
    """
    n_values = config.n_schedule
    window = config.plateau_window_grow
    epsilon = config.plateau_eps_grow

    level = 0
    start = 0
    while level < len(n_values) - 1 and start + window <= len(history):
        segment = history[start : start + window]
        if plateau_detected(segment, window, epsilon):
            level += 1
            start += window
        else:
            start += 1
    return n_values[level]


@dataclass
class GenerationRecord:
    """Један ред лога — исти облик за baseline и bilevel (README 4.3, BASELINE_SPEC §9).

    `best_so_far` и `invalid_count` додати 31.08. ради испраћивања покретања (испис у
    конзоли, `ProgressReporter`). `working_nodes` додат 01.09. (docs/NALAZ_31_08.md) —
    величина предачког стабла трагача (радни чворови, за разлику од укупног `best_n_nodes`
    који укључује и мртав терет). `min_transmission_angle_deg`/`path_jump_count`/
    `path_loop_closure` додати 01.09. (docs/NALAZ_01_09_ugao_prenosa.md) — здравље путање
    најбоље јединке генерације. `link_to_radius_ratio` додат 02.09. (DECISIONS §17) —
    гломазност механизма (највећа полуга / полупречник путање), дијагностика, НЕ улази у
    оцену. Ниједно поље не мења ток претраге — сва се рачунају из већ израчунатих
    `scores`/топологије/путање, ван буџета (без `CallCounter`-а).
    """

    generation: int
    calls_spent: int
    best_fitness: float
    mean_fitness: float
    n_curve: int
    best_n_nodes: int
    best_so_far: float = float("nan")   # најбољи фитнес од почетка покретања, не само у овој генерацији
    invalid_count: int = 0              # број јединки које су добиле казну (в. `fitness.PENALTY`)
    working_nodes: int = 0              # величина предачког стабла трагача (docs/NALAZ_31_08.md, 01.09.)
    min_transmission_angle_deg: float = float("nan")  # мин. угао преноса кроз обртај (docs/NALAZ_01_09...)
    path_jump_count: int = 0                          # скокова у путањи (корак > 8× медијане)
    path_loop_closure: float = float("nan")           # |последња−прва тачка| / медијана корака
    link_to_radius_ratio: float = float("nan")        # највећа полуга / полупречник путање (DECISIONS §17)


@dataclass
class RunLog:
    """Запис једног покретања — заједнички облик за baseline и bilevel, основа за све три
    метрике поређења H1 (README 3.2, BASELINE_SPEC §9).

    Проширено 31.08.: стари `error_curve`/`final_error` нису довољни за оно што SPEC §9
    тражи (mean fitness, N, n најбоље јединке по генерацији) — сада су у `records`.
    `error_curve` остаје као computed својство ради компатибилности облика
    `[(позиви, најбољи фитнес), ...]` (README 3.2, `plot_error_curve`).
    """

    method: str
    curve: str
    seed: int
    curve_hash: str = ""
    git_commit: str = ""
    config: dict = field(default_factory=dict)
    records: list[GenerationRecord] = field(default_factory=list)
    best_genome: Genome | None = None
    final_error: float = float("nan")
    # Бројач: колико пута је почетна тачка ЦМА-ЕС-а имала `ρ` ван декларисаних граница
    # (DECISIONS §24, А1) — bilevel-специфично, baseline га никад не увећава. Ако остаје 0
    # кроз мерење, то потврђује да су границе добро постављене; ако није, види се одмах.
    rho_out_of_bounds: int = 0

    @property
    def error_curve(self) -> list[tuple[int, float]]:
        """(позиви, најбољи фитнес) — READ ONLY, изведено из `records`."""
        return [(r.calls_spent, r.best_fitness) for r in self.records]

    def record(
        self,
        generation: int,
        calls_spent: int,
        best_fitness: float,
        mean_fitness: float,
        n_curve: int,
        best_n_nodes: int,
        best_so_far: float = float("nan"),
        invalid_count: int = 0,
        working_nodes: int = 0,
        min_transmission_angle_deg: float = float("nan"),
        path_jump_count: int = 0,
        path_loop_closure: float = float("nan"),
        link_to_radius_ratio: float = float("nan"),
    ) -> GenerationRecord:
        """Дописује ред и враћа га — позивалац (`baseline.evolve`) га прослеђује репортеру."""
        entry = GenerationRecord(
            generation, calls_spent, best_fitness, mean_fitness, n_curve, best_n_nodes,
            best_so_far, invalid_count, working_nodes,
            min_transmission_angle_deg, path_jump_count, path_loop_closure,
            link_to_radius_ratio,
        )
        self.records.append(entry)
        self.final_error = best_fitness
        return entry

    def save(self, path: str) -> None:
        """Снима у JSON (README 4.3) — numpy низови и `Genome` преко `_genome_to_dict`."""
        data = {
            "method": self.method,
            "curve": self.curve,
            "curve_hash": self.curve_hash,
            "seed": self.seed,
            "git_commit": self.git_commit,
            "config": self.config,
            "final_error": self.final_error,
            "rho_out_of_bounds": self.rho_out_of_bounds,
            "records": [asdict(r) for r in self.records],
            "best_genome": _genome_to_dict(self.best_genome) if self.best_genome is not None else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


    @classmethod
    def load(cls, path: str) -> "RunLog":
        """Учитава лог снимљен `save`-ом (README 4.3) — обрнута операција, ништа више.

        Омогућава `run.py show` и `run.py replay` да раде над сачуваним резултатом, без
        поновног тренирања. Толерантна на записе снимљене пре 31.08.: непозната поља се
        игноришу, недостајућа добијају подразумеване вредности из `GenerationRecord`.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        allowed = {f.name for f in dataclasses_fields(GenerationRecord)}
        records = [
            GenerationRecord(**{k: v for k, v in entry.items() if k in allowed})
            for entry in data.get("records", [])
        ]
        return cls(
            method=data["method"],
            curve=data.get("curve", ""),
            seed=data.get("seed", 0),
            curve_hash=data.get("curve_hash", ""),
            git_commit=data.get("git_commit", ""),
            config=data.get("config", {}),
            records=records,
            best_genome=_genome_from_dict(data["best_genome"]) if data.get("best_genome") else None,
            final_error=data.get("final_error", float("nan")),
            rho_out_of_bounds=data.get("rho_out_of_bounds", 0),
        )


def _genome_to_dict(genome: Genome) -> dict:
    """JSON-серијализабилан облик генома (README 4.3) — помоћна функција за `RunLog.save`."""
    return {
        "n_nodes": genome.topology.n_nodes,
        "edges": [list(edge) for edge in genome.topology.edges],
        "coords": genome.coords.tolist(),
    }


def _genome_from_dict(data: dict) -> Genome:
    """Обрнуто од `_genome_to_dict` — реконструкција генома из JSON-a (README 4.3).

    `edges` се враћају као листа торки јер `Topology` тако очекује (списак из JSON-a би
    прошао кроз већину кода, али би пао на упоређивање и на кључеве у `link_lengths`).
    """
    topology = Topology(
        n_nodes=int(data["n_nodes"]),
        edges=[tuple(edge) for edge in data["edges"]],
    )
    return Genome(topology=topology, coords=np.asarray(data["coords"], dtype=float))


def make_rng(seed: int) -> np.random.Generator:
    """Застарео потпис — један заједнички генератор крши изолацију токова (README 4.3).
    Замењено са `spawn_rng_streams` (BASELINE_SPEC §9, „четири RNG тока", 30.08.).
    Не користити у новом коду.
    """
    raise NotImplementedError


def spawn_rng_streams(seed: int) -> tuple[np.random.Generator, ...]:
    """Четири независна RNG тока из једног главног seed-а (BASELINE_SPEC §9).

    `rng_init` — иницијализација популације; `rng_select` — селекција родитеља;
    `rng_cross` — избор реза при укрштању; `rng_mut` — избор оператора, ослонаца, alpha,
    rho, s, gene-wise новчићи (в. `baseline.evolve_generation`, план 31.08. питање 4).

    И baseline и bilevel зову ово исто (§22 В5 је bilevel-у додавало пети ток `rng_cma`
    преко `n_streams=5` — укинуто 05.09., DECISIONS §24: унутрашњи ЦМА-ЕС сад добија
    позиционо семе преко `bilevel.cma_rng`, независно од овог механизма — в.
    `bilevel.outer_ga`).
    """
    streams = np.random.SeedSequence(seed).spawn(4)
    return tuple(np.random.default_rng(s) for s in streams)


def curve_file_hash(path: str) -> str:
    """SHA-256 садржаја фајла циљне криве — за поновљивост лога (README 4.3, BASELINE_SPEC §9)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_commit_hash() -> str:
    """Хеш тренутног git commit-а за логовање (README 4.3); `"unknown"` ако није доступан
    (није гит репо, `git` недоступан, итд.) — логовање не сме да падне због овога."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"
