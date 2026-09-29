"""CPU checks for the staged yaw schedule, cycle, and promotion gates."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
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


def test_phase0_hold_requires_one_continuous_second_and_keeps_fsm_dwell_event():
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
        assert not tracker.phase0_hold_success.item()
        assert not tracker.just_four_dwell.item()
    step(True)
    assert not tracker.phase0_hold_success.item()
    assert tracker.just_four_dwell.item()
    for _ in range(39):
        step(True)
    assert tracker.max_continuous_four_ready_s.item() == pytest.approx(.98)
    assert not tracker.phase0_hold_success.item()
    assert not tracker.just_four_dwell.item()
    step(True)
    assert tracker.phase0_hold_success.item()
    assert tracker.max_continuous_four_ready_s.item() == pytest.approx(1.0)


def test_phase0_readiness_interruption_resets_certification_timer():
    fsm = _load("fsm")
    tracker = fsm.StagedCycleTracker(1, "cpu", .02, .20)
    state = torch.tensor([int(fsm.VQRFsmState.FOUR_STAND)])
    sign = torch.tensor([1])
    false = torch.tensor([False])
    zero = torch.zeros(1)

    def step(ready, unsafe=False):
        tracker.update(state, state, sign, zero, torch.tensor([ready]),
                       false, false, torch.tensor([unsafe]), zero, zero)

    for _ in range(30):
        step(True)
    step(False)
    assert tracker.phase0_ready_time.item() == 0.0
    for _ in range(49):
        step(True)
    assert not tracker.phase0_hold_success.item()
    step(True)
    assert tracker.phase0_hold_success.item()
    step(True, unsafe=True)
    assert not tracker.phase0_hold_success.item()

    tracker.reset(slice(None))
    recovery = torch.tensor([int(fsm.VQRFsmState.SAFE_RECOVERY)])
    tracker.update(state, recovery, sign, zero, torch.tensor([True]),
                   false, false, false, zero, zero)
    assert not tracker.phase0_hold_success.item()


def test_phase0_reset_time_command_update_does_not_count_toward_hold():
    fsm = _load("fsm")
    tracker = fsm.StagedCycleTracker(1, "cpu", .02, .20)
    state = torch.tensor([int(fsm.VQRFsmState.FOUR_STAND)])
    sign = torch.tensor([1])
    true = torch.tensor([True])
    false = torch.tensor([False])
    zero = torch.zeros(1)

    def step(active):
        tracker.update(state, state, sign, zero, true, false, false, false,
                       zero, zero, phase0_active=torch.tensor([active]))

    step(False)
    assert tracker.phase0_ready_time.item() == 0.0
    for _ in range(49):
        step(True)
    assert not tracker.phase0_hold_success.item()
    step(True)
    assert tracker.phase0_hold_success.item()


def test_phase0_hold_terminates_and_phase1_to_3_completion_stays_the_same():
    fsm = _load("fsm")
    module = _load("staged_yaw_curriculum")
    tracker = fsm.StagedCycleTracker(1, "cpu", .02, .20)
    state = torch.tensor([int(fsm.VQRFsmState.FOUR_STAND)])
    sign = torch.tensor([1])
    true = torch.tensor([True])
    false = torch.tensor([False])
    zero = torch.zeros(1)
    command = fsm.StagedYawCommand.__new__(fsm.StagedYawCommand)
    command.episode_phase = torch.tensor([0])
    command.cycle = tracker
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    for _ in range(49):
        tracker.update(state, state, sign, zero, true, false, false, false, zero, zero)
    assert not module.staged_complete(env).item()
    tracker.update(state, state, sign, zero, true, false, false, false, zero, zero)
    assert module.staged_complete(env).item()

    command.episode_phase = torch.tensor([1, 2, 3])
    command.cycle = SimpleNamespace(
        phase0_hold_success=torch.ones(3, dtype=torch.bool),
        entered_transition=torch.tensor([True, False, False]),
        entered_yaw=torch.tensor([False, True, False]),
        full_cycle_complete=torch.tensor([False, False, True]),
        invalid=torch.zeros(3, dtype=torch.bool),
    )
    assert module.staged_complete(env).tolist() == [True, True, True]
    command.cycle.entered_transition[0] = False
    command.cycle.entered_yaw[1] = False
    command.cycle.full_cycle_complete[2] = False
    assert module.staged_complete(env).tolist() == [False, False, False]


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
    env = SimpleNamespace(
        termination_manager=done,
        command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(episode_phase=torch.tensor([1, 1]))),
    )
    assert module._staged_failure_mask(env).tolist() == [False, True]


def test_staged_four_contact_has_bounded_request_grace_and_idle_reward():
    module = _load("staged_yaw_curriculum")
    fsm_module = _load("fsm")
    fsm = fsm_module.YawFSMVectorized(1, dt=.02, four_stand_ready_dwell=.20,
                                      four_reward_grace_s=.40)
    command = SimpleNamespace(
        fsm_state=fsm.fsm_state, four_stand_ready=torch.tensor([True]),
        four_reward_gate=fsm.four_reward_gate,
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    false = torch.tensor([False])
    yaw = torch.tensor([.15])

    fsm.update(torch.zeros(1), false, false, command.four_stand_ready, false)
    assert module.staged_four_contact(env).item() == 1.0
    for step in range(30):
        # Each ready pulse is too short to complete the FOUR dwell. The
        # request stays latched, so readiness flicker cannot rearm grace.
        ready = torch.tensor([step % 2 == 0])
        command.four_stand_ready = ready
        fsm.update(yaw, false, false, ready, false)
        if step >= 20 and ready.item():
            assert module.staged_four_contact(env).item() == 0.0
    assert fsm.fsm_state.item() == int(fsm_module.VQRFsmState.FOUR_STAND)
    command.four_stand_ready = torch.tensor([True])
    fsm.update(torch.zeros(1), false, false, command.four_stand_ready, false)
    assert module.staged_four_contact(env).item() == 1.0


def test_timeout_failure_is_phase_aware_and_one_shot_with_safety():
    module = _load("staged_yaw_curriculum")
    phase = torch.tensor([0, 1, 2, 3, 3])
    terms = {
        "staged_complete": torch.tensor([False, False, False, False, False]),
        "staged_promotion_reset": torch.zeros(5, dtype=torch.bool),
        "torso_contact": torch.tensor([False, False, False, False, True]),
        "time_out": torch.ones(5, dtype=torch.bool),
    }
    done = SimpleNamespace(
        active_terms=list(terms), terminated=terms["torso_contact"],
        get_term=lambda name: terms[name],
    )
    env = SimpleNamespace(
        step_dt=.02, termination_manager=done,
        command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(episode_phase=phase)),
    )
    assert module._staged_failure_mask(env).tolist() == [False, True, True, True, True]
    assert module.staged_failure_cost(env).tolist() == [0.0, 50.0, 50.0, 50.0, 50.0]
    # A valid completion at the episode limit is not an incomplete timeout.
    terms["staged_complete"][3] = True
    assert module._staged_failure_mask(env).tolist() == [False, True, True, False, True]


@pytest.mark.parametrize("phase", [0, 1, 2, 3])
def test_timeout_accounting_keeps_attempt_and_rejects_incomplete_maneuver(phase):
    module = _load("staged_yaw_curriculum")
    fsm = _load("fsm")
    cycle = fsm.StagedCycleTracker(1, "cpu", .02, .20)
    cycle.episode_steps[:] = 10
    cycle.four_ready_steps[:] = 10
    cycle.four_ready_time[:] = .20
    cycle.entered_transition[:] = True
    cycle.entered_yaw[:] = True
    cycle.landed_four[:] = True
    cycle.full_cycle_complete[:] = True
    command = SimpleNamespace(
        schedule=SimpleNamespace(sign=torch.tensor([1])), cycle=cycle,
        episode_phase=torch.tensor([phase]), cfg=SimpleNamespace(),
        unsafe=torch.tensor([False]), four_stand_ready=torch.tensor([True]),
    )
    terms = {
        "staged_complete": torch.tensor([False]),
        "staged_promotion_reset": torch.tensor([False]),
        "time_out": torch.tensor([True]),
    }
    done = SimpleNamespace(
        active_terms=list(terms), terminated=torch.tensor([False]),
        get_term=lambda name: terms[name],
    )
    event_cfgs = {name: SimpleNamespace(params={}) for name in (
        "randomize_apply_external_force_torque", "randomize_actuator_gains", "randomize_push_robot",
    )}
    env = SimpleNamespace(
        num_envs=1, device="cpu", step_dt=.02, episode_length_buf=torch.tensor([10]),
        command_manager=SimpleNamespace(get_term=lambda _: command), termination_manager=done,
        reward_manager=SimpleNamespace(active_terms=[]),
        event_manager=SimpleNamespace(
            get_term_cfg=lambda name: event_cfgs[name],
            set_term_cfg=lambda name, value: event_cfgs.__setitem__(name, value),
        ),
        _staged_yaw_promotion=module.StagedPromotion(phase=phase),
    )
    module.staged_yaw_task_levels(env, [0], "yaw_rate_cmd", (.02, .03, .05),
                                  (.15, .25), (.30, 1.0), (.30, .30))
    row = env._staged_yaw_promotion.window["pos"]
    assert row["attempts"] == 1
    assert row["four_hold"] == 0
    if phase:
        assert row[("entry", "yaw", "yaw", "cycle")[phase]] == 0


@pytest.mark.parametrize("phase", [0, 2, 3])
def test_promotion_resets_all_episodes_before_global_reward_change(phase):
    module = _load("staged_yaw_curriculum")
    clearance_levels = (.02, .03, .05)
    yaw_levels = (.15, .25)
    dr_levels = (.30, 1.0)
    tracking = (.30, .30)
    stage = module.StagedPromotion(phase=phase)
    stage.consecutive_passes = 2
    stage.record("pos", four_hold=1, entry=1, yaw=1, support=1, pose=1, cycle=1,
                 command_sum=1.0, error_sum=.1, drift_sum=.02, drift_samples=1)
    target_before = .05 if phase == 3 else .02
    cfg = SimpleNamespace(
        staged_phase=phase, staged_clearance_index=0, staged_yaw_index=0,
        yaw_rate_limit=.15, yaw_rate_range=(-.15, .15), target_clearance=target_before,
    )
    cycle = SimpleNamespace(
        phase0_hold_success=torch.ones(2, dtype=torch.bool),
        entered_transition=torch.ones(2, dtype=torch.bool),
        entered_yaw=torch.ones(2, dtype=torch.bool),
        landed_four=torch.ones(2, dtype=torch.bool),
        full_cycle_complete=torch.ones(2, dtype=torch.bool),
        invalid=torch.zeros(2, dtype=torch.bool),
        support_ready_seen=torch.ones(2, dtype=torch.bool),
        pose_ready_seen=torch.ones(2, dtype=torch.bool),
        command_sum=torch.ones(2), error_sum=torch.full((2,), .1),
        yaw_samples=torch.ones(2, dtype=torch.long),
        drift_sum=torch.full((2,), .02), drift_samples=torch.ones(2, dtype=torch.long),
    )
    command = SimpleNamespace(
        cfg=cfg, cycle=cycle, schedule=SimpleNamespace(sign=torch.tensor([1, -1])),
        episode_phase=torch.full((2,), phase),
        episode_clearance_index=torch.zeros(2, dtype=torch.long),
        episode_yaw_index=torch.zeros(2, dtype=torch.long),
        episode_yaw_limit=torch.full((2,), .15),
        episode_target_clearance=torch.full((2,), target_before),
        unsafe=torch.zeros(2, dtype=torch.bool),
        four_stand_ready=torch.tensor([True, phase != 0]),
    )
    terms = {
        "staged_complete": torch.tensor([False, True]),
        "staged_promotion_reset": torch.zeros(2, dtype=torch.bool),
        "time_out": torch.tensor([False, False]),
    }
    done = SimpleNamespace(
        active_terms=list(terms), terminated=terms["staged_complete"],
        get_term=lambda name: terms[name],
    )
    reward_cfg = SimpleNamespace(weight=2.0, params={"target_clearance": target_before})
    rewards = SimpleNamespace(
        active_terms=["fsm_gated_tracking"], get_term_cfg=lambda _: reward_cfg,
        set_term_cfg=lambda name, value: None,
    )
    event_cfgs = {name: SimpleNamespace(params={}) for name in (
        "randomize_apply_external_force_torque", "randomize_actuator_gains", "randomize_push_robot",
    )}
    event_cfgs["randomize_apply_external_force_torque"].params["force_range"] = (-3.0, 3.0) if phase == 3 else (0.0, 0.0)
    events = SimpleNamespace(
        get_term_cfg=lambda name: event_cfgs[name],
        set_term_cfg=lambda name, value: event_cfgs.__setitem__(name, value),
    )
    env = SimpleNamespace(
        num_envs=2, device="cpu", step_dt=.02,
        episode_length_buf=torch.tensor([10, 10]),
        _staged_yaw_promotion=stage,
        command_manager=SimpleNamespace(get_term=lambda _: command),
        termination_manager=done, reward_manager=rewards, event_manager=events,
    )
    args = (env, [1], "yaw_rate_cmd", clearance_levels, yaw_levels, dr_levels, tracking)
    module.staged_yaw_task_levels(*args, hold_window=2, directional_window=1, required_windows=3)
    assert env._staged_yaw_promotion_reset_pending
    assert (stage.phase, stage.clearance_index, stage.yaw_index) == (
        (1, 0, 0) if phase == 0 else (2, 1, 0) if phase == 2 else (3, 0, 1)
    )
    assert cfg.target_clearance == target_before
    assert reward_cfg.params["target_clearance"] == target_before
    assert cfg.yaw_rate_limit == .15
    assert event_cfgs["randomize_apply_external_force_torque"].params["force_range"] == (
        (-3.0, 3.0) if phase == 3 else (0.0, 0.0)
    )
    reset_mask = module.staged_promotion_reset(env)
    assert reset_mask.tolist() == [True, True]
    assert not module._staged_failure_mask(env).any()

    # Isaac Lab's next step selects every ID from the promotion termination,
    # then applies curriculum before command_manager.reset(all_ids).
    terms["staged_promotion_reset"] = reset_mask
    terms["staged_complete"] = torch.zeros(2, dtype=torch.bool)
    terms["time_out"] = torch.zeros(2, dtype=torch.bool)
    done.terminated = torch.zeros(2, dtype=torch.bool)
    assert not module._staged_failure_mask(env).any()
    all_ids = reset_mask.nonzero().flatten()
    module.staged_yaw_task_levels(env, all_ids, "yaw_rate_cmd", clearance_levels,
                                  yaw_levels, dr_levels, tracking,
                                  hold_window=2, directional_window=1, required_windows=3)
    assert not env._staged_yaw_promotion_reset_pending
    assert stage.window["pos"]["attempts"] == stage.window["neg"]["attempts"] == 0
    command.episode_phase[all_ids] = cfg.staged_phase
    command.episode_clearance_index[all_ids] = cfg.staged_clearance_index
    command.episode_yaw_index[all_ids] = cfg.staged_yaw_index
    command.episode_yaw_limit[all_ids] = cfg.yaw_rate_limit
    command.episode_target_clearance[all_ids] = cfg.target_clearance
    assert command.episode_phase.unique().tolist() == [cfg.staged_phase]
    assert torch.all(command.episode_target_clearance == reward_cfg.params["target_clearance"])
    assert torch.all(command.episode_yaw_limit == cfg.yaw_rate_limit)
    assert cfg.target_clearance == (.03 if phase == 2 else target_before)
    assert cfg.yaw_rate_limit == (.25 if phase == 3 else .15)
    assert event_cfgs["randomize_apply_external_force_torque"].params["force_range"] == (
        (-10.0, 10.0) if phase == 3 else (-0.0, 0.0)
    )
