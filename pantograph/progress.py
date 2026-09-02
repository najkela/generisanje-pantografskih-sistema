"""Испраћивање покретања: испис у конзоли и периодични снимци (додато 31.08.).

Овај модул је ЧИСТО посматрач. Ништа овде не сме да утиче на ток претраге, па се држе
три правила, свако везано за једну ствар која би тихо покварила поређење метода:

1. **Не троши буџет.** Бројач позива (`simulator.CallCounter`) инкрементира искључиво
   `fitness.evaluate`. Цртање зато иде преко `validate` + `simulate` директно и никад
   преко `evaluate` — трошак остаје нула по конструкцији, не по договору.
2. **Не троши насумичност.** Ниједан избор овде није случајан, па нема ни петог RNG тока
   поред `rng_init`/`rng_select`/`rng_cross`/`rng_mut`. Њихово стање остаје нетакнуто и
   покретања са истим seed-ом остају упоредива (README 4.3).
3. **Не блокира петљу.** Снимци се цртају преко `Figure` + `FigureCanvasAgg` (в.
   `visualization.snapshot`), без `pyplot`-а и без `show()`.

Стоји у засебном модулу, а не у `experiment.py`, да алгоритамска путања (`baseline` →
`experiment`) не увлачи matplotlib. `baseline.evolve` не увози ништа одавде — прима објекат
и зове му `update`.
"""

import sys
import time

from .curve import TargetCurve
from .experiment import GenerationRecord, RunLog
from .genome import Genome
from .visualization import snapshot

SEPARATOR = "═" * 104
THIN = "─" * 104


def _thousands(value: int) -> str:
    """12345 → „12 345" — размак као раздвајач хиљада, како се пише ћирилицом."""
    return f"{value:,}".replace(",", " ")


def _elapsed(seconds: float) -> str:
    """Протекло време као ММ:СС, односно Ч:ММ:СС кад пређе сат."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ProgressReporter:
    """Испис по генерацији и периодични снимци најбоље јединке.

    Живот објекта прати једно покретање: `start()` пре петље, `update()` једном по
    генерацији (зове га `baseline.evolve`), `finish()` после петље.

    `snapshot_every=0` гаси снимке; `log_every` проређује испис (свака генерација
    подразумевано). Генерација 0 и последња генерација се увек снимају, без обзира на
    период — прва слика показује одакле се кренуло, последња докле се стигло.
    """

    def __init__(
        self,
        total_budget: int,
        population_size: int = 0,
        target: TargetCurve | None = None,
        run_dir: str | None = None,
        log_every: int = 1,
        snapshot_every: int = 0,
        snapshot_n: int = 720,
        stream=sys.stdout,
    ) -> None:
        self.total_budget = total_budget
        self.population_size = population_size
        self.target = target
        self.run_dir = run_dir
        self.log_every = max(1, log_every)
        self.snapshot_every = max(0, snapshot_every)
        self.snapshot_n = snapshot_n
        self.stream = stream

        self._t0 = time.monotonic()
        self._last_n: int | None = None
        self._last_improved = 0
        self._best_so_far = float("inf")
        self._last_record: GenerationRecord | None = None
        self._snapshots: list[str] = []

    # ------------------------------------------------------------------ испис

    def _write(self, line: str = "") -> None:
        print(line, file=self.stream, flush=True)

    def start(
        self,
        method: str,
        curve: str,
        seed: int,
        n_schedule: tuple[int, ...],
        git_commit: str = "",
        profile: str = "",
    ) -> None:
        """Заглавље: све што одређује покретање, на једном месту пре прве генерације."""
        self._t0 = time.monotonic()
        self._write(SEPARATOR)
        self._write(f"  ГЕНЕРИСАЊЕ ПАНТОГРАФСКИХ МЕХАНИЗАМА — метод: {method}")
        self._write(SEPARATOR)
        if profile:
            self._write(f"  ПРОФИЛ: {profile} — демонстрација, НИЈЕ мерење за поређење метода")
            self._write(THIN)
        rows = [
            ("циљна крива", curve),
            ("seed", str(seed)),
            ("популација", str(self.population_size)),
            ("буџет", f"{_thousands(self.total_budget)} позива симулатора"),
            ("распоред N", " → ".join(str(n) for n in n_schedule)),
            ("git commit", (git_commit or "непознат")[:12]),
        ]
        if self.run_dir:
            rows.append(("фолдер", self.run_dir))
        rows.append((
            "снимци",
            f"на сваких {self.snapshot_every} генерација" if self.snapshot_every else "искључени",
        ))
        for label, value in rows:
            self._write(f"  {label:<16}{value}")
        self._write(SEPARATOR)

    def update(self, record: GenerationRecord, best_genome: Genome | None = None) -> None:
        """Једна генерација: линија у конзоли и, по периоду, снимак.

        Зове се ПОСЛЕ `RunLog.record`, са редом који је она вратила. Не додирује ни буџет
        ни RNG токове (в. докстринг модула).
        """
        self._last_record = record

        if record.best_so_far < self._best_so_far:
            self._best_so_far = record.best_so_far
            self._last_improved = record.generation

        if self._last_n is not None and record.n_curve != self._last_n:
            # Chamfer рачунат на различитом N није упоредив (исти разлог због ког
            # `evolve` намерно не кешира елиту), па бројач стагнације креће испочетка.
            self._write(
                f"  ── резолуција криве: N {self._last_n} → {record.n_curve} "
                f"(генерација {record.generation})"
            )
            self._last_improved = record.generation
        self._last_n = record.n_curve

        if record.generation % self.log_every == 0:
            self._write(self._format(record))

        if self._due_for_snapshot(record.generation):
            self._snapshot(record, best_genome)

    def _format(self, record: GenerationRecord) -> str:
        percent = 100.0 * record.calls_spent / self.total_budget if self.total_budget else 0.0
        stagnation = record.generation - self._last_improved
        return (
            f"ген {record.generation:>5} │ "
            f"{_elapsed(time.monotonic() - self._t0):>7} │ "
            f"најбољи {record.best_so_far:.4e} │ "
            f"стагнација {stagnation:>4} ген. │ "
            f"N={record.n_curve:>3} │ "
            f"чворова {record.best_n_nodes:>2} (радних {record.working_nodes:>2}) │ "
            f"угао {record.min_transmission_angle_deg:>5.1f}° │ "
            f"гломазност {record.link_to_radius_ratio:>5.2f} │ "
            f"невалидних {self._invalid_share(record):>5} │ "
            f"буџет {percent:>5.1f}%"
        )

    def _invalid_share(self, record: GenerationRecord) -> str:
        """Удео јединки које су добиле казну — колико је покретање „болесно".

        Тачан је захваљујући одлуци од 30.08. да се валидан фитнес одсеца на 1e8 а казна
        износи 1e9: граница `scores >= PENALTY` је оштра, нема преклапања.
        """
        if not self.population_size:
            return str(record.invalid_count)
        return f"{100.0 * record.invalid_count / self.population_size:.0f}%"

    # ---------------------------------------------------------------- снимци

    def _due_for_snapshot(self, generation: int) -> bool:
        if not self.snapshot_every or self.run_dir is None or self.target is None:
            return False
        return generation == 0 or generation % self.snapshot_every == 0

    def _snapshot(self, record: GenerationRecord, genome: Genome | None) -> None:
        """Снима PNG најбоље јединке. Тиха на грешку — покретање се не сме срушити због
        цртања; једна изгубљена слика није вредна изгубљеног тренинга."""
        if genome is None:
            return
        import os

        folder = os.path.join(self.run_dir, "snapshots")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"gen_{record.generation:05d}.png")
        title = (
            f"генерација {record.generation} · Chamfer {record.best_so_far:.4e} · "
            f"N={record.n_curve} · {record.best_n_nodes} чворова"
        )
        try:
            if snapshot(genome, self.target, path, title=title, n=self.snapshot_n):
                self._snapshots.append(path)
        except Exception as error:  # цртање никад не обара тренинг
            self._write(f"  ── снимак генерације {record.generation} није успео: {error}")

    # ---------------------------------------------------------------- завршетак

    def final_snapshot(self, log: RunLog) -> None:
        """Снимак последње генерације, независно од периода (в. докстринг класе)."""
        if self.run_dir is None or self.target is None or not log.records:
            return
        last = log.records[-1]
        if not self._due_for_snapshot(last.generation) and self.snapshot_every:
            self._snapshot(last, log.best_genome)

    def finish(self, log: RunLog) -> None:
        """Резиме: зашто је стало, колико је трајало, шта је испало."""
        self._write(THIN)
        if not log.records:
            self._write("  Нема ниједне генерације — покретање је стало пре прве евалуације.")
            self._write(SEPARATOR)
            return

        last = log.records[-1]
        reason = (
            "исцрпљен буџет" if last.calls_spent >= self.total_budget
            else "плато на највећем N (критеријум заустављања)"
        )
        edges = len(log.best_genome.topology.edges) if log.best_genome is not None else 0
        nodes = log.best_genome.topology.n_nodes if log.best_genome is not None else 0

        percent = 100.0 * last.calls_spent / self.total_budget if self.total_budget else 0.0
        rows = [
            ("заустављено", reason),
            ("генерација", str(last.generation + 1)),
            ("потрошено", f"{_thousands(last.calls_spent)} позива ({percent:.1f}% буџета)"),
            ("трајање", _elapsed(time.monotonic() - self._t0)),
            ("коначна грешка", f"{log.final_error:.6e} (Chamfer)"),
            ("најбоља јединка", f"{nodes} чворова, {edges} полуга"),
            ("угао преноса", f"{last.min_transmission_angle_deg:.1f}° (мин. кроз обртај)"),
            ("скокова у путањи", str(last.path_jump_count)),
            ("затварање петље", f"{last.path_loop_closure:.2f}× медијане корака"),
            ("гломазност", f"{last.link_to_radius_ratio:.2f}× (полуга/путања)"),
        ]
        if self._snapshots:
            rows.append(("снимака", f"{len(self._snapshots)} у snapshots/"))
        if self.run_dir:
            rows.append(("фолдер", self.run_dir))
        for label, value in rows:
            self._write(f"  {label:<16}{value}")
        self._write(SEPARATOR)
