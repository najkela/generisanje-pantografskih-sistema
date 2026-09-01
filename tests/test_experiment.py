"""Спецификација заједничке инфраструктуре: буџет, распоред N, RNG токови, лог (README 2.4,
3.2, 4.3; BASELINE_SPEC §7, §9)."""

import numpy as np

from pantograph.config import Config
from pantograph.experiment import (
    Budget,
    curve_file_hash,
    plateau_detected,
    resolution_schedule,
    spawn_rng_streams,
)


def test_plateau_detected_false_before_window_filled():
    assert plateau_detected([1.0, 0.5], window=5, epsilon=0.01) is False


def test_plateau_detected_true_when_no_improvement():
    history = [1.0] * 10  # нема побољшања уопште
    assert plateau_detected(history, window=5, epsilon=0.01) is True


def test_plateau_detected_false_with_strong_improvement():
    history = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3]
    assert plateau_detected(history, window=5, epsilon=0.01) is False


def test_resolution_schedule_starts_at_first_level():
    config = Config(n_schedule=(90, 180, 360, 720), plateau_window_grow=5, plateau_eps_grow=0.01)
    assert resolution_schedule(0, [], config) == 90


def test_resolution_schedule_advances_on_plateau():
    config = Config(n_schedule=(90, 180, 360, 720), plateau_window_grow=5, plateau_eps_grow=0.01)
    history = [1.0] * 5  # плато одмах у првом прозору
    assert resolution_schedule(5, history, config) == 180


def test_resolution_schedule_never_exceeds_last_level():
    config = Config(n_schedule=(90, 180, 360, 720), plateau_window_grow=3, plateau_eps_grow=0.01)
    history = [1.0] * 50  # трајан плато кроз читаву историју
    assert resolution_schedule(50, history, config) == 720


def test_budget_spend_and_exhausted():
    budget = Budget(max_calls=5)
    assert budget.spent == 0
    assert not budget.exhausted
    budget.spend(3)
    assert budget.spent == 3
    assert not budget.exhausted
    budget.spend(2)
    assert budget.spent == 5
    assert budget.exhausted


def test_spawn_rng_streams_returns_four_independent_generators():
    streams = spawn_rng_streams(42)
    assert len(streams) == 4
    draws = [s.uniform() for s in streams]
    assert len(set(draws)) == 4  # практично никад једнаки за независне токове

    # исти seed → исти токови (репродуцибилност, README 4.3)
    streams_again = spawn_rng_streams(42)
    draws_again = [s.uniform() for s in streams_again]
    assert draws == draws_again


def test_curve_file_hash_is_deterministic(tmp_path):
    path = tmp_path / "curve.txt"
    path.write_text("0.0,0.0\n1.0,0.0\n")
    h1 = curve_file_hash(str(path))
    h2 = curve_file_hash(str(path))
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 64  # sha256 hex
