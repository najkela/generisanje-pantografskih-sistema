"""Режим поређења за H1 (DECISIONS §24.2, имплементирано 14.09.).

Спецификација режима, тачка по тачка: фиксно N=720 за оба метода, фиксан буџет идентичан
за оба, без раног заустављања тока на плато, и крива грешка-по-позивима на заједничкој
решетки израженој у позивима. Свака тврдња из §24.2 има овде свој тест.

Најважније што ови тестови чувају јесте да режим остане ДОПУНА, не замена — подразумевани
профил (§18) мора проћи кроз све измене нетакнут.
"""

import dataclasses

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pantograph.baseline import evolve
from pantograph.config import DEFAULT_CONFIG, POREDJENJE_CONFIG, QUICK_CONFIG, Config
from pantograph.curve import TargetCurve
from pantograph.experiment import RunLog, resolution_schedule


# ------------------------------------------------------------------- профил

def test_comparison_profile_has_the_three_required_properties():
    """§24.2: фиксно N=720, буџет 100 000, без раног заустављања."""
    assert POREDJENJE_CONFIG.fixed_n_curve == 720
    assert POREDJENJE_CONFIG.total_budget == 100_000
    assert POREDJENJE_CONFIG.early_stop is False


def test_default_profile_is_untouched():
    """Режим је допуна, не замена — §18 остаје подразумеван за свакодневни рад."""
    assert DEFAULT_CONFIG.fixed_n_curve is None
    assert DEFAULT_CONFIG.early_stop is True
    assert DEFAULT_CONFIG.total_budget == 50_000
    assert DEFAULT_CONFIG.n_schedule == (90, 180, 360, 720)


# ---------------------------------------------------------- фиксна резолуција

def test_fixed_n_bypasses_the_schedule():
    """Са `fixed_n_curve` распоред се не примењује ни за једну историју."""
    config = dataclasses.replace(DEFAULT_CONFIG, fixed_n_curve=720)
    for history in ([], [1.0], [1.0] * 200, [1.0, 0.5, 0.25] * 50):
        assert resolution_schedule(len(history), history, config) == 720


def test_schedule_unchanged_without_fixed_n():
    """Регресија: подразумевани профил и даље креће од 90 и напредује на плато."""
    assert resolution_schedule(0, [], DEFAULT_CONFIG) == 90
    plateau = [1.0] * DEFAULT_CONFIG.plateau_window_grow
    assert resolution_schedule(len(plateau), plateau, DEFAULT_CONFIG) == 180


# -------------------------------------------------------------------- решетка

def test_grid_has_no_holes_and_is_monotone():
    log = RunLog(method="baseline", curve="", seed=1)
    log.grid_step = 100
    for calls, best in [(90, 5.0), (310, 3.0), (1000, 1.0)]:
        log.update_grid(calls, best)
    assert [c for c, _ in log.calls_curve] == list(range(100, 1001, 100))
    values = [v for _, v in log.calls_curve]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_grid_is_a_step_function_not_interpolated():
    """Вредност у тачки `g` је последње осматрање са `calls_spent <= g` (§24.2)."""
    log = RunLog(method="baseline", curve="", seed=1)
    log.grid_step = 100
    log.update_grid(90, 5.0)     # прво осматрање, пре прве тачке решетке
    log.update_grid(550, 2.0)
    assert log.calls_curve == [(100, 5.0), (200, 5.0), (300, 5.0), (400, 5.0), (500, 5.0)]


def test_grid_before_first_observation_uses_first_value():
    """Bilevel потроши ≈2 900 позива пре првог осматрања — те тачке не смеју бити NaN."""
    log = RunLog(method="bilevel", curve="", seed=1)
    log.grid_step = 500
    log.update_grid(2900, 7.0)
    assert [v for _, v in log.calls_curve] == [7.0] * 5
    assert not any(np.isnan(v) for _, v in log.calls_curve)


def test_flush_fills_grid_up_to_full_budget():
    log = RunLog(method="baseline", curve="", seed=1)
    log.grid_step = 100
    log.update_grid(250, 4.0)
    log.flush_grid(1000)
    assert [c for c, _ in log.calls_curve] == list(range(100, 1001, 100))
    assert log.calls_curve[-1] == (1000, 4.0)


def test_save_load_round_trips_the_grid(tmp_path):
    log = RunLog(method="baseline", curve="", seed=1)
    log.grid_step = 100
    log.update_grid(250, 4.0)
    log.cached_calls = 17
    path = tmp_path / "log.json"
    log.save(str(path))
    loaded = RunLog.load(str(path))
    assert loaded.grid_step == 100
    assert loaded.calls_curve == [(100, 4.0), (200, 4.0)]
    assert loaded.cached_calls == 17


# --------------------------------------------------------------- цео ток

@pytest.fixture
def short_comparison_config() -> Config:
    """Режим поређења сведен да стане у тест: исте три особине, мањи бројеви."""
    return dataclasses.replace(
        QUICK_CONFIG, fixed_n_curve=90, early_stop=False, total_budget=1_500, grid_step=100
    )


@pytest.fixture
def short_stoppable_config() -> Config:
    """Контролни пар за `early_stop`: једини ниво N је 90, па критеријум заустављања
    (који важи само на `n_schedule[-1]`) стварно може да окине, а праг је намерно груб."""
    return dataclasses.replace(
        QUICK_CONFIG, n_schedule=(90,), total_budget=1_500, grid_step=100,
        plateau_window_stop=3, plateau_eps_stop=0.99,
    )


def test_comparison_run_spends_whole_budget_at_fixed_n(circle, short_comparison_config):
    target = TargetCurve(points=circle, tree=cKDTree(circle))
    log = evolve(target, budget=1_500, seed=1, population_size=30, config=short_comparison_config)
    assert log.records[-1].calls_spent >= 1_500
    assert all(r.n_curve == 90 for r in log.records)
    assert [c for c, _ in log.calls_curve] == list(range(100, 1_501, 100))


def test_early_stop_true_stops_before_budget(circle, short_stoppable_config):
    """Контрола А: са укљученим раним заустављањем и грубим прагом ток стане рано."""
    target = TargetCurve(points=circle, tree=cKDTree(circle))
    log = evolve(target, budget=1_500, seed=1, population_size=30, config=short_stoppable_config)
    assert log.records[-1].calls_spent < 1_000


def test_early_stop_false_ignores_the_same_plateau(circle, short_stoppable_config):
    """Контрола Б: исти профил, само `early_stop=False` — ток иде до буџета.

    Разлика између ова два теста долази искључиво од заставице, не од нечег другог
    у режиму.
    """
    config = dataclasses.replace(short_stoppable_config, early_stop=False)
    target = TargetCurve(points=circle, tree=cKDTree(circle))
    log = evolve(target, budget=1_500, seed=1, population_size=30, config=config)
    assert log.records[-1].calls_spent >= 1_500


def test_elite_cache_saves_calls_at_fixed_n(circle, short_comparison_config):
    """Под фиксним N кеш се никад не поништава — број уштеђених позива мора бити видљив."""
    target = TargetCurve(points=circle, tree=cKDTree(circle))
    log = evolve(target, budget=1_500, seed=1, population_size=30, config=short_comparison_config)
    assert log.cached_calls > 0
