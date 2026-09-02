"""Спецификација заједничке инфраструктуре: буџет, распоред N, RNG токови, лог (README 2.4,
3.2, 4.3; BASELINE_SPEC §7, §9)."""

import json

import numpy as np

from pantograph.config import Config
from pantograph.experiment import (
    Budget,
    GenerationRecord,
    RunLog,
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


def test_generation_record_link_to_radius_ratio_defaults_to_nan():
    """Ново поље (DECISIONS §17) — гломазност — има подразумевану вредност `nan` кад се не
    наведе (потребно да `RunLog.load` учита старе логове без овог поља)."""
    record = GenerationRecord(
        generation=0, calls_spent=1, best_fitness=0.1, mean_fitness=0.2, n_curve=90, best_n_nodes=4,
    )
    assert np.isnan(record.link_to_radius_ratio)


def test_run_log_load_tolerates_records_without_link_to_radius_ratio(tmp_path):
    """`RunLog.load` мора учитати стари лог чији записи немају `link_to_radius_ratio`
    (додато 02.09., DECISIONS §17) — филтрира по именима поља `GenerationRecord`, непозната
    поља игнорише, недостајућа добијају подразумевану вредност (`nan`)."""
    old_style_record = {
        "generation": 0,
        "calls_spent": 10,
        "best_fitness": 0.05,
        "mean_fitness": 0.1,
        "n_curve": 90,
        "best_n_nodes": 4,
        "best_so_far": 0.05,
        "invalid_count": 0,
        "working_nodes": 4,
        "min_transmission_angle_deg": 12.0,
        "path_jump_count": 0,
        "path_loop_closure": 1.0,
        # НЕМА "link_to_radius_ratio" — симулира лог снимљен пре 02.09.
    }
    data = {
        "method": "baseline",
        "curve": "data/curves/circle.txt",
        "seed": 1,
        "curve_hash": "",
        "git_commit": "",
        "config": {},
        "final_error": 0.05,
        "records": [old_style_record],
        "best_genome": None,
    }
    path = tmp_path / "log.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    log = RunLog.load(str(path))

    assert len(log.records) == 1
    assert np.isnan(log.records[0].link_to_radius_ratio)
