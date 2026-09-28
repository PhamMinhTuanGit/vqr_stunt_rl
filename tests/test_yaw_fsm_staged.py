"""CPU checks for the staged yaw schedule, cycle, and promotion gates."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).parents[1]
MDP = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, MDP / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_signed_schedule_is_balanced_and_has_one_four_second_request():
    schedule = _load("fsm").StagedYawSchedule(5, "cpu", .1)
    schedule.reset(torch.arange(5))
    assert schedule.sign.tolist() == [1, -1, 1, -1, 1]
    for _ in range(4):
        assert not schedule.step(3, .15).any()
    assert torch.allclose(schedule.step(3, .15), torch.tensor([.15, -.15, .15, -.15, .15]))
    for _ in range(39):
        schedule.step(3, .15)
    assert not schedule.step(3, .15).any()
    schedule.reset(torch.tensor([0, 1, 2]))
    assert schedule.sign[:3].tolist() == [-1, 1, -1]
    assert not schedule.step(3, .15, torch.zeros(5, dtype=torch.bool)).any()
    assert schedule.ticks[:3].tolist() == [0, 0, 0]
    for _ in range(5):
        phase_zero = schedule.step(0, .15)
    assert not phase_zero.any()

    schedule.reset(torch.arange(5))
    phases = torch.tensor([0, 1, 2, 3, 3])
    limits = torch.tensor([.15, .15, .25, .40, .55])
    for _ in range(5):
        mixed = schedule.step(phases, limits)
    assert torch.allclose(mixed, torch.tensor([0.0, -.15, .25, -.40, .55]))


def test_promotion_is_latched_only_for_newly_reset_environments():
    fsm = _load("fsm")
    command = fsm.StagedYawCommand.__new__(fsm.StagedYawCommand)
    command.num_envs = 2
    command.device = "cpu"
    command._command = torch.zeros(2, 1)
    command.schedule = fsm.StagedYawSchedule(2, "cpu", .02)
    command.episode_phase = torch.zeros(2, dtype=torch.long)
    command.episode_clearance_index = torch.zeros(2, dtype=torch.long)
    command.episode_yaw_index = torch.zeros(2, dtype=torch.long)
    command.episode_yaw_limit = torch.zeros(2)
    command.episode_target_clearance = torch.zeros(2)
    command.cfg = SimpleNamespace(staged_phase=2, staged_clearance_index=0,
                                  staged_yaw_index=0, yaw_rate_limit=.15, target_clearance=.02)
    command._resample_command(torch.arange(2))
    command.cfg.staged_phase = 3
    command.cfg.staged_clearance_index = 2
    command.cfg.staged_yaw_index = 1
    command.cfg.yaw_rate_limit = .25
    command.cfg.target_clearance = .05
    command._resample_command(torch.tensor([0]))
    assert command.episode_phase.tolist() == [3, 2]
    assert command.episode_clearance_index.tolist() == [2, 0]
    assert command.episode_yaw_index.tolist() == [1, 0]
    assert torch.allclose(command.episode_yaw_limit, torch.tensor([.25, .15]))
    assert torch.allclose(command.episode_target_clearance, torch.tensor([.05, .02]))


def test_full_cycle_requires_intended_yaw_zero_return_and_fresh_four_dwell():
    fsm = _load("fsm")
    tracker = fsm.StagedCycleTracker(1, "cpu", .02, .20)
    sign = torch.tensor([1])
    true = torch.tensor([True])
    false = torch.tensor([False])
    zero = torch.zeros(1)

    def step(previous, state, command=0.0, ready=True):
        tracker.update(torch.tensor([previous]), torch.tensor([state]), sign,
                       torch.tensor([command]), torch.tensor([ready]), true, true,
                       false, torch.tensor([command]), zero)

    state = fsm.VQRFsmState
    step(state.FOUR_STAND, state.TRANSITION_POS, .15)
    step(state.TRANSITION_POS, state.YAW_POS, .15)
    step(state.YAW_POS, state.RETURN_TO_4, 0.0)
    assert tracker.zero_return_seen.item()
    assert not tracker.full_cycle_complete.item()
    step(state.RETURN_TO_4, state.FOUR_STAND)
    assert tracker.landed_four.item()
    for _ in range(9):
        step(state.FOUR_STAND, state.FOUR_STAND)
    assert not tracker.full_cycle_complete.item()
    step(state.FOUR_STAND, state.FOUR_STAND)
    assert tracker.full_cycle_complete.item()

    tracker.reset(slice(None))
    step(state.FOUR_STAND, state.TRANSITION_POS, .15)
    step(state.TRANSITION_POS, state.RETURN_TO_4)
    step(state.RETURN_TO_4, state.FOUR_STAND)
    for _ in range(10):
        step(state.FOUR_STAND, state.FOUR_STAND)
    assert tracker.invalid.item()
    assert not tracker.full_cycle_complete.item()


def test_four_hold_requires_current_dwell_and_pays_one_dwell_event():
    fsm = _load("fsm")
    tracker = fsm.StagedCycleTracker(1, "cpu", .02, .20)
    state = torch.tensor([int(fsm.VQRFsmState.FOUR_STAND)])
    sign = torch.tensor([1])
    true = torch.tensor([True])
    false = torch.tensor([False])
    zero = torch.zeros(1)

    def step(ready):
        tracker.update(state, state, sign, zero, torch.tensor([ready]),
                       false, false, false, zero, zero)

    for _ in range(9):
        step(True)
        assert not tracker.four_hold_success.item()
        assert not tracker.just_four_dwell.item()
    step(True)
    assert tracker.four_hold_success.item()
    assert tracker.just_four_dwell.item()
    step(True)
    assert not tracker.just_four_dwell.item()
    step(False)
    assert not tracker.four_hold_success.item()
    for _ in range(10):
        step(True)
    assert tracker.four_hold_success.item()
    assert not tracker.just_four_dwell.item()


def _record_window(promotion, pos_success: int, neg_success: int, phase: int):
    for direction, successes in (("pos", pos_success), ("neg", neg_success)):
        for index in range(2):
            good = index < successes
            promotion.record(direction, four_hold=int(good), entry=int(good), yaw=int(good),
                             **{"return": int(good)}, cycle=int(good), support=int(good),
                             pose=int(good), command_sum=1.0, error_sum=.1, drift_sum=.04,
                             drift_samples=1, yaw_samples=1)
    return promotion.evaluate((.02, .03, .05), (.15, .25), (.3, .3),
                              hold_window=4, directional_window=2, required_windows=3)


def test_clearance_ladder_and_directional_refusal():
    promotion = _load("staged_yaw_curriculum").StagedPromotion()
    for _ in range(3):
        _record_window(promotion, 2, 2, 0)
    assert promotion.phase == 1
    for _ in range(3):
        _record_window(promotion, 2, 0, 1)
    assert promotion.phase == 1
    assert promotion.consecutive_passes == 0
    for _ in range(3):
        _record_window(promotion, 2, 2, 1)
    assert promotion.phase == 2
    for expected in (1, 2):
        for _ in range(3):
            _record_window(promotion, 2, 2, 2)
        assert promotion.clearance_index == expected
    for _ in range(3):
        _record_window(promotion, 2, 2, 2)
    assert promotion.phase == 3
    assert promotion.clearance_index == 2
    for _ in range(3):
        _record_window(promotion, 2, 2, 3)
    assert promotion.yaw_index == 1
    restored = _load("staged_yaw_curriculum").StagedPromotion.restore(promotion.export())
    assert restored.export() == promotion.export()


def test_timeout_and_failure_remain_in_each_direction_denominator():
    module = _load("staged_yaw_curriculum")
    fsm = _load("fsm")
    cycle = fsm.StagedCycleTracker(2, "cpu", .02, .20)
    cycle.entered_transition[:] = True
    command = SimpleNamespace(
        schedule=SimpleNamespace(sign=torch.tensor([1, -1])), cycle=cycle,
        episode_phase=torch.tensor([1, 1]),
        cfg=SimpleNamespace(),
    )
    event_terms = {
        name: SimpleNamespace(params={}) for name in (
            "randomize_apply_external_force_torque", "randomize_actuator_gains",
            "randomize_push_robot",
        )
    }
    events = SimpleNamespace(get_term_cfg=lambda name: event_terms[name],
                             set_term_cfg=lambda name, cfg: event_terms.__setitem__(name, cfg))
    done = SimpleNamespace(
        terminated=torch.tensor([True, False]),
        get_term=lambda name: {"staged_complete": torch.tensor([False, False]),
                               "time_out": torch.tensor([False, True])}[name],
    )
    env = SimpleNamespace(
        device="cpu", num_envs=2, step_dt=.02, episode_length_buf=torch.tensor([5, 5]),
        command_manager=SimpleNamespace(get_term=lambda _: command),
        termination_manager=done, event_manager=events,
        reward_manager=SimpleNamespace(active_terms=[]),
        _staged_yaw_promotion=module.StagedPromotion(phase=1),
    )
    module.staged_yaw_task_levels(env, [0, 1], "yaw_rate_cmd", (.02, .03, .05),
                                  (.15, .25), (.3, 1.0), (.3, .3))
    counts = env._staged_yaw_promotion.window
    assert [counts[d]["attempts"] for d in ("pos", "neg")] == [1, 1]
    assert [counts[d]["entry"] for d in ("pos", "neg")] == [0, 0]
    env._staged_yaw_promotion.phase = 2
    module.staged_yaw_task_levels(env, [0, 1], "yaw_rate_cmd", (.02, .03, .05),
                                  (.15, .25), (.3, 1.0), (.3, .3))
    assert [counts[d]["attempts"] for d in ("pos", "neg")] == [1, 1]


def test_safety_failure_overrides_simultaneous_completion():
    module = _load("staged_yaw_curriculum")
    done = SimpleNamespace(
        active_terms=["staged_complete", "torso_contact", "time_out"],
        terminated=torch.tensor([True, True]),
        get_term=lambda name: {
            "staged_complete": torch.tensor([True, True]),
            "torso_contact": torch.tensor([False, True]),
            "time_out": torch.tensor([False, False]),
        }[name],
    )
    assert module._staged_failure_mask(done).tolist() == [False, True]
