"""Пресликавање секвенца ↔ вектор геометрије за bilevel (DECISIONS §22).

Вектор `x` је у бази `(q1, q2, ρ_a, ρ_b, ...)` изведеној из секвенце склапања, не сирове
координате свих чворова — знак гране `s` је закуцан ген топологије (§17), геометрија носи
само оно на шта ЦМА-ЕС стварно смеmo да утиче непрекидно (§22, образложење базе). Мртви
чворови (ван предачког стабла трагача) не заузимају димензије — њихов `ρ` се замрзава на
затеченим вредностима и враћа при реконструкцији (§22, А2).

Овај модул је једина капија ка фитнесу за унутрашњи ЦМА-ЕС (`evaluate_vector`) — исто као
што је `fitness.evaluate` јединствена капија за baseline.
"""

import numpy as np

from .config import Config, DEFAULT_CONFIG
from .curve import TargetCurve
from .fitness import PENALTY, evaluate
from .genome import Genome, Sequence, SequenceGene, TopologySkeleton, from_sequence
from .simulator import CallCounter


def skeleton_of(seq: Sequence) -> TopologySkeleton:
    """Одбаци `ρ` из секвенце — остаје дискретни скелет (позиције ослонаца + знак гране)."""
    n_nodes = 3 + len(seq.genes)
    genes = [(gene.a, gene.b, gene.s) for gene in seq.genes]
    return TopologySkeleton(n_nodes=n_nodes, genes=genes)


def active_positions(skeleton: TopologySkeleton) -> list[int]:
    """Позиције `k ≥ 3` из предачког стабла трагача (DECISIONS §22, А2).

    Предачко стабло се рачуна из самог скелета (`a`, `b` показују уназад, инваријанта
    редоследа §16 — solving order је увек `[3, ..., n-1]`, позиције гена = индекси чворова),
    не преко `operators.prune_dead_nodes` (та функција ради над `Topology`/ивицама, овде
    немамо ивице, само скелет). Резултат увек садржи трагача самог (`n_nodes - 1`) — он је
    корен сопственог предачког стабла.
    """
    n = skeleton.n_nodes
    tracer_position = n - 1
    parents_of = {3 + i: (a, b) for i, (a, b, _s) in enumerate(skeleton.genes)}

    active: set[int] = set()
    stack = [tracer_position]
    while stack:
        node = stack.pop()
        if node < 3 or node in active:
            continue
        active.add(node)
        for parent in parents_of[node]:
            if parent >= 3 and parent not in active:
                stack.append(parent)
    return sorted(active)


def to_x(seq: Sequence, active: list[int]) -> np.ndarray:
    """Секвенца → раван вектор `[q1x, q1y, q2x, q2y, ρ_a, ρ_b, ...]` (DECISIONS §22).

    Језгро (`q1` = `core[1]`, `q2` = `core[2]`) је увек присутно; чвор 0 остаје закуцан у
    (0,0) и није део вектора (README 1.3). `ρ` иду само за позиције из `active`, истим
    редом којим су наведене.
    """
    parts = [seq.core[1], seq.core[2]]
    for k in active:
        gene = seq.genes[k - 3]
        parts.append(np.array([gene.rho_a, gene.rho_b]))
    return np.concatenate(parts)


def to_sequence_from_x(
    x: np.ndarray,
    skeleton: TopologySkeleton,
    frozen_rho: dict[int, tuple[float, float]],
    active: list[int],
) -> Sequence:
    """Инверз функције `to_x` — саставља секвенцу из вектора, скелета и замрзнутих `ρ`.

    За позиције ван `active` (мртав терет) `ρ` се узима из `frozen_rho`, не из `x`
    (DECISIONS §22, А2) — мртав чвор не утиче на путању, па га унутрашњи ЦМА-ЕС не сме ни
    видети. `frozen_rho` мора имати унос за сваку позицију `k ≥ 3` која НИЈЕ у `active`.
    """
    core = np.zeros((3, 2))
    core[1] = x[0:2]
    core[2] = x[2:4]

    active_set = set(active)
    genes: list[SequenceGene] = []
    idx = 4
    for i, (a, b, s) in enumerate(skeleton.genes):
        k = 3 + i
        if k in active_set:
            rho_a, rho_b = float(x[idx]), float(x[idx + 1])
            idx += 2
        else:
            rho_a, rho_b = frozen_rho[k]
        genes.append(SequenceGene(a=a, b=b, rho_a=rho_a, rho_b=rho_b, s=s))

    return Sequence(core=core, genes=genes)


def evaluate_vector(
    x: np.ndarray,
    skeleton: TopologySkeleton,
    frozen_rho: dict[int, tuple[float, float]],
    active: list[int],
    target: TargetCurve,
    n: int,
    counter: CallCounter,
    config: Config = DEFAULT_CONFIG,
) -> float:
    """Јединствена капија ка фитнесу за унутрашњи ЦМА-ЕС (DECISIONS §22, А5).

    Склапа секвенцу → `genome.from_sequence` → ако падне на circuit defect (`None`),
    бројач се инкрементира РУЧНО (isto pravilo kao za baseline — jedan pokušaj = jedan
    позив, без обзира да ли је геометрија решива, README 1.4/1.6) и враћа се `fitness.PENALTY`;
    иначе позив иде преко `fitness.evaluate`, која инкрементира бројач тачно једном сама.
    Тако свака инвокација ове функције троши тачно један позив буџета, без изузетка.
    """
    seq = to_sequence_from_x(x, skeleton, frozen_rho, active)
    result = from_sequence(seq)
    if result is None:
        counter.increment()
        return PENALTY
    topology, coords = result
    return evaluate(Genome(topology=topology, coords=coords), target, n, counter=counter, config=config)
