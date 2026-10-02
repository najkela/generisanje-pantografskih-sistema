"""Режим фиксног n (DECISIONS §26.3) — спецификација, по узору на
tests/test_rezim_poredjenja.py. Свака тврдња из налога има овде свој тест.

Најважније што ови тестови чувају: `add_node`/`delete_node` морају бити структурно
недостижни при `config.fixed_n_nodes is not None`, а понашање без те заставице (default,
`None`) мора остати нетакнуто — регресија.
"""

import dataclasses

import numpy as np
import pytest
from scipy.spatial import cKDTree

import run
from pantograph.baseline import crossover, evolve, evolve_generation, initialize_population
from pantograph.bilevel import _initialize_population, _next_generation, outer_ga
from pantograph.config import QUICK_CONFIG, Config
from pantograph.curve import TargetCurve
from pantograph.experiment import Budget, spawn_rng_streams
from pantograph.genome import to_sequence, to_vector
from pantograph.geometry_vector import active_positions, skeleton_of, to_x


# ---------------------------------------------------------------- профил/фикстуре

@pytest.fixture
def short_baseline_config() -> Config:
    """Фиксно n=6, мали буџет — стане у тест."""
    return dataclasses.replace(
        QUICK_CONFIG, fixed_n_nodes=6, population_size=20, total_budget=400,
    )


@pytest.fixture
def short_bilevel_config() -> Config:
    """Фиксно n=6, мала спољашња популација и кратак унутрашњи плато-прозор."""
    return dataclasses.replace(
        QUICK_CONFIG, fixed_n_nodes=6, outer_population=6, total_budget=1_500,
        k_max=6, plateau_window_k=2,
    )


# -------------------------------------------------------------- иницијализација

def test_baseline_initial_population_has_exactly_fixed_n(short_baseline_config):
    rng_init, *_ = spawn_rng_streams(0)
    population = initialize_population(short_baseline_config, rng_init)
    assert all(g.topology.n_nodes == 6 for g in population)


def test_bilevel_initial_population_has_exactly_fixed_n(short_bilevel_config):
    rng_init, *_ = spawn_rng_streams(0)
    population = _initialize_population(short_bilevel_config, rng_init)
    assert all(r.skeleton.n_nodes == 6 for r in population)


# ------------------------------------------------------- кроз цео ток, по генерацији

def test_baseline_population_stays_at_fixed_n_every_generation(short_baseline_config):
    """Директан позив `evolve_generation` кроз неколико генерација — популација мора
    остати на `n=6` у свакој од њих, не само на старту (DECISIONS §26.3)."""
    config = short_baseline_config
    rng_init, rng_select, rng_cross, rng_mut = spawn_rng_streams(0)
    population = initialize_population(config, rng_init)
    scores = np.random.default_rng(0).uniform(size=len(population))
    for _ in range(5):
        population, _known = evolve_generation(
            population, scores, rng_select, rng_cross, rng_mut, k=0.1, config=config
        )
        assert all(g.topology.n_nodes == 6 for g in population)
        scores = np.random.default_rng(0).uniform(size=len(population))


def test_bilevel_population_stays_at_fixed_n_every_generation(short_bilevel_config):
    """Исто за спољашњи ГА bilevel-а — мртви записи (`skeleton is None`) немају n, само
    живи се проверавају."""
    config = short_bilevel_config
    rng_init, rng_select, _rng_cross, rng_mut = spawn_rng_streams(0)
    budget = Budget(max_calls=config.total_budget)
    population = _initialize_population(config, rng_init)
    scores = np.array([r.best_fitness for r in population])
    for _ in range(5):
        population = _next_generation(population, scores, config, rng_select, rng_mut, budget)
        for record in population:
            if record.skeleton is not None:
                assert record.skeleton.n_nodes == 6
        scores = np.array([r.best_fitness for r in population])


# ------------------------------------------------------------- add_node/delete_node недостижни

def test_baseline_never_calls_add_or_delete_node(monkeypatch, circle, short_baseline_config):
    """Кроз цео `evolve()` под фиксним n, `add_node`/`delete_node` се никад не позивају —
    monkeypatch циља увезена имена у `pantograph.baseline`, не `pantograph.operators`."""
    import pantograph.baseline as baseline_module

    def _boom(*args, **kwargs):
        raise AssertionError("add_node/delete_node позвано у режиму фиксног n")

    monkeypatch.setattr(baseline_module, "add_node", _boom)
    monkeypatch.setattr(baseline_module, "delete_node", _boom)

    target = TargetCurve(points=circle, tree=cKDTree(circle))
    log = evolve(
        target, budget=short_baseline_config.total_budget, seed=1,
        population_size=short_baseline_config.population_size, config=short_baseline_config,
    )
    assert log.records  # ток је стварно одрадио бар једну генерацију


def test_bilevel_never_calls_add_or_delete_node(monkeypatch, circle, short_bilevel_config):
    """Исто за bilevel — циља `pantograph.bilevel.add_node`/`.delete_node_with_target`."""
    import pantograph.bilevel as bilevel_module

    def _boom(*args, **kwargs):
        raise AssertionError("add_node/delete_node_with_target позвано у режиму фиксног n")

    monkeypatch.setattr(bilevel_module, "add_node", _boom)
    monkeypatch.setattr(bilevel_module, "delete_node_with_target", _boom)

    target = TargetCurve(points=circle, tree=cKDTree(circle))
    log = outer_ga(target, budget=short_bilevel_config.total_budget, seed=1, config=short_bilevel_config)
    assert log.records


# --------------------------------------------------------------------------- d_x

def test_baseline_dx_is_always_2n_minus_2(short_baseline_config):
    """Baseline не креше мртав терет током претраге — цео `coords` низ увек присутан, па је
    `len(to_vector(coords)) == 2n-2` без изузетка (за разлику од bilevel-а испод)."""
    rng_init, *_ = spawn_rng_streams(0)
    population = initialize_population(short_baseline_config, rng_init)
    for genome in population:
        assert len(to_vector(genome.coords)) == 2 * 6 - 2


def test_bilevel_dx_matches_active_count_formula(short_bilevel_config):
    """d_x = 4 + 2·|active| увек важи (§22 В2/В3) — интерна конзистентност конструкције."""
    rng_init, *_ = spawn_rng_streams(0)
    population = _initialize_population(short_bilevel_config, rng_init)
    for record in population:
        assert len(record.init_x0) == 4 + 2 * len(record.active)
        assert len(record.init_x0) <= 2 * 6 - 2  # горња граница, иста за целу популацију


def test_bilevel_dx_equals_2n_minus_2_without_dead_weight(sixbar):
    """Позитиван случај: кад нема мртвог терета, горња граница се стварно достиже —
    d_x = 2n-2, исто као и baseline (`sixbar` нема мртве чворове по конструкцији)."""
    topology, coords = sixbar
    seq = to_sequence(topology, coords)
    skeleton = skeleton_of(seq)
    active = active_positions(skeleton)
    assert len(active) == topology.n_nodes - 3
    x0 = to_x(seq, active)
    assert len(x0) == 2 * topology.n_nodes - 2


# ------------------------------------------------------------------------ укрштање

def test_crossover_of_equal_fixed_n_genomes_preserves_n(short_baseline_config):
    """Резом секвенце склапања дете увек величине `n_B` (README 2.2) — кад су оба родитеља
    исте величине `n` (иницијализација то гарантује), дете је аутоматски исте величине."""
    config = short_baseline_config
    rng_init, _rng_select, rng_cross, _rng_mut = spawn_rng_streams(0)
    population = initialize_population(config, rng_init)
    made_a_child = False
    for _ in range(100):
        a = population[int(rng_cross.integers(len(population)))]
        b = population[int(rng_cross.integers(len(population)))]
        child = crossover(a, b, rng_cross)
        if child is not None:
            made_a_child = True
            assert child.topology.n_nodes == 6
    assert made_a_child


# ------------------------------------------------------------------- именовање фолдера

def test_run_directory_gets_n_segment_when_fixed():
    """`_run_directory` уграђује `_n{N}` непосредно пре `_seed{seed}` кад је режим фиксног
    n активан — штити логику прескакања завршених прогона у
    `tools/matrica_poredjenja.sh` (DECISIONS §26.3)."""
    path = run._run_directory("results", "baseline", "data/curves/egg.txt", 3, fixed_n_nodes=6)
    assert path.endswith("_baseline_egg_n6_seed3")


def test_run_directory_unchanged_without_fixed_n():
    """Регресија: без `fixed_n_nodes` (или са `None`) име остаје бит-идентично данашњем
    облику — нема `_n` сегмента."""
    path_default = run._run_directory("results", "baseline", "data/curves/egg.txt", 3)
    path_explicit_none = run._run_directory(
        "results", "baseline", "data/curves/egg.txt", 3, fixed_n_nodes=None
    )
    assert path_default == path_explicit_none
    assert path_default.endswith("_baseline_egg_seed3")
