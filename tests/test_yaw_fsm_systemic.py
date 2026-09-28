"""CPU regressions for reward stalling, watchdog chatter, and survivor bias."""

import ast
from types import SimpleNamespace

import pytest
import torch

from test_yaw_curriculum import _fsm_checkpoint_env, _fsm_curriculum_params, _load_curriculums_module
from test_yaw_fsm import FSM_PATH, REWARDS_PATH, _load_fsm_module, _load_fsm_tracking_reward


def _load_nodes(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def _reward_scene():
    rewards, state = _load_fsm_tracking_reward()
    math = SimpleNamespace(
        euler_xyz_from_quat=lambda q: (q[:, 0], q[:, 1], q[:, 2]),
        wrap_to_pi=lambda x: x,
    )
    rewards.update(math_utils=math, Articulation=object)
    _load_nodes(REWARDS_PATH, {"TransitionProgress", "yaw_balance", "four_stand_stability",
                              "yaw_base_height_tracking"}, rewards)
    command = SimpleNamespace(
        fsm_state=torch.tensor([state.TRANSITION_POS, state.TRANSITION_NEG]),
        support_diagonal=torch.tensor([1, -1]), state_time=torch.zeros(2),
        just_switched=torch.ones(2, dtype=torch.bool),
        four_reward_gate=torch.ones(2),
        cfg=SimpleNamespace(yaw_rate_range=(-0.25, 0.25)),
    )
    forces = torch.zeros(2, 4, 3)
    forces[0, [0, 3], 2] = 80
    forces[1, [1, 2], 2] = 80

    class Scene(dict):
        sensors = {"contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))}
        env_origins = torch.zeros(2, 3)

    env = SimpleNamespace(
        num_envs=2, device="cpu", step_dt=.02, common_step_counter=0,
        scene=Scene(robot=SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.tensor([[0., 0., .49], [0., 0., .49]]),
            root_quat_w=torch.zeros(2, 4), root_ang_vel_b=torch.zeros(2, 3),
        ))),
        lift_progress={"pos": torch.zeros(2, 2), "neg": torch.zeros(2, 2)},
        command_manager=SimpleNamespace(get_term=lambda _: command,
                                        get_command=lambda _: torch.tensor([[.2], [-.2]])),
    )
    return rewards, state, env, command


def _pose_rewards(rewards, env):
    fsm = {"fsm_command_name": "yaw_rate_cmd"}
    lift = dict(asset_cfg=SimpleNamespace(name="pos"), asset_cfg_mirror=SimpleNamespace(name="neg"),
                wheel_radius=.091, target_clearance=.05, **fsm)
    support = dict(sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
                   sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]))
    geom = dict(asset_cfg=SimpleNamespace(name="robot"), asset_cfg_mirror=SimpleNamespace(name="robot"))
    return torch.stack([
        2 * rewards["yaw_balance"](env, std=.25, **fsm),
        3 * rewards["yaw_com_support"](env, std=.08, **geom, **fsm),
        2 * rewards["yaw_com_inside_support_segment"](env, std=.05, **geom, **fsm),
        3 * rewards["yaw_lift_clearance"](env, **lift,
            support_sensor_cfg=support["sensor_cfg"], support_sensor_cfg_mirror=support["sensor_cfg_mirror"]),
        8 * rewards["fsm_gated_tracking"](env, command_name="yaw_rate_cmd", **fsm,
            support_sensor_cfg=support["sensor_cfg"], support_sensor_cfg_mirror=support["sensor_cfg_mirror"],
            lifted_asset_cfg=lift["asset_cfg"], lifted_asset_cfg_mirror=lift["asset_cfg_mirror"],
            wheel_radius=.091, target_clearance=.05, std=.3),
        3 * rewards["four_stand_stability"](env, target_height=.49, **fsm),
        .49 * rewards["yaw_base_height_tracking"](env, target_height=.49, error_scale=.1, **fsm),
    ])


def test_four_reward_grace_is_bounded_across_readiness_flicker_and_neutral_rearms():
    rewards, state, env, command = _reward_scene()
    fsm = _load_fsm_module().YawFSMVectorized(num_envs=2, dt=.02)
    command.fsm_state = fsm.fsm_state
    command.state_time = fsm.state_time
    command.just_switched = fsm.just_switched
    command.four_reward_gate = fsm.four_reward_gate
    false = torch.zeros(2, dtype=torch.bool)
    yaw = torch.full((2,), .2)

    def four_rewards():
        env.common_step_counter += 1
        return (
            rewards["four_stand_stability"](env, target_height=.49, fsm_command_name="yaw_rate_cmd"),
            rewards["yaw_balance"](env, std=.25, fsm_command_name="yaw_rate_cmd"),
        )

    for step in range(25):
        # Each true pulse is shorter than the required continuous 0.2 s dwell.
        ready = torch.full((2,), step % 2 == 0)
        fsm.update(yaw, false, false, ready, false)
        stability, balance = four_rewards()
        expected = torch.ones(2) if step < 20 else torch.zeros(2)
        assert torch.equal(stability, expected)
        assert torch.equal(balance, expected)
        assert torch.equal(fsm.fsm_state, torch.full((2,), state.FOUR_STAND))

    # Hysteresis retains the request and cannot renew its grace.
    fsm.update(torch.full((2,), .08), false, false, false, false)
    stability, balance = four_rewards()
    assert torch.equal(stability, torch.zeros(2))
    assert torch.equal(balance, torch.zeros(2))
    assert fsm.maneuver_requested.all()

    fsm.update(torch.zeros(2), false, false, false, false)
    stability, balance = four_rewards()
    assert torch.equal(stability, torch.ones(2))
    assert torch.equal(balance, torch.ones(2))
    assert not fsm.maneuver_requested.any()


def test_four_reward_remains_full_during_continuous_ready_dwell():
    rewards, state, env, command = _reward_scene()
    fsm = _load_fsm_module().YawFSMVectorized(num_envs=2, dt=.02)
    command.fsm_state = fsm.fsm_state
    command.state_time = fsm.state_time
    command.just_switched = fsm.just_switched
    command.four_reward_gate = fsm.four_reward_gate
    false = torch.zeros(2, dtype=torch.bool)
    true = ~false
    for _ in range(9):
        fsm.update(torch.tensor([.2, -.2]), false, false, true, false)
        env.common_step_counter += 1
        assert torch.equal(fsm.fsm_state, torch.full((2,), state.FOUR_STAND))
        assert torch.equal(rewards["four_stand_stability"](
            env, target_height=.49, fsm_command_name="yaw_rate_cmd"), torch.ones(2))
        assert torch.equal(rewards["yaw_balance"](
            env, std=.25, fsm_command_name="yaw_rate_cmd"), torch.ones(2))
    fsm.update(torch.tensor([.2, -.2]), false, false, true, false)
    assert fsm.fsm_state.tolist() == [state.TRANSITION_POS, state.TRANSITION_NEG]
    assert fsm.dwell_completion_count.tolist() == [1, 1]
    assert fsm.four_to_transition_count.tolist() == [1, 1]


def test_four_gate_does_not_change_transition_yaw_or_return_rewards():
    rewards, state, env, command = _reward_scene()
    command.state_time.fill_(.5)
    for fsm_state in (state.TRANSITION_POS, state.YAW_POS, state.RETURN_TO_4):
        command.fsm_state.fill_(fsm_state)
        outputs = []
        for gate in (1.0, 0.0):
            command.four_reward_gate.fill_(gate)
            env.common_step_counter += 1
            outputs.append((
                rewards["four_stand_stability"](
                    env, target_height=.49, fsm_command_name="yaw_rate_cmd"),
                rewards["yaw_balance"](env, std=.25, fsm_command_name="yaw_rate_cmd"),
            ))
        for full, gated in zip(*outputs):
            assert torch.equal(full, gated)


def test_curriculum_reports_entry_rates_and_terminal_transition_exit(monkeypatch):
    curriculum = _load_curriculums_module(monkeypatch)
    env = _fsm_checkpoint_env(curriculum, 10_000)
    params = _fsm_curriculum_params()
    fsm = _load_fsm_module().YawFSMVectorized(
        num_envs=3, dt=.02, four_stand_ready_dwell=.04, yaw_pose_ready_dwell=0.0,
    )
    env.command_manager.get_term("yaw_rate_cmd").fsm_diagnostics = fsm
    false = torch.zeros(3, dtype=torch.bool)
    for _ in range(2):
        fsm.update(torch.full((3,), .2), false, false,
                   torch.tensor([True, False, True]), false)
    fsm.update(torch.full((3,), .2), torch.tensor([True, False, False]),
               false, false, false)
    assert fsm.fsm_state.tolist() == [2, 0, 1]
    env.episode_length_buf.fill_(100)
    termination_masks = {
        "torso_contact": false, "base_height_failure": false, "tilt_failure": false,
        "fsm_transition_timeout": torch.tensor([False, False, True]),
        "time_out": torch.tensor([False, True, True]),
    }
    env.termination_manager = SimpleNamespace(get_term=termination_masks.__getitem__)

    result = curriculum.yaw_fsm_task_levels(env, torch.arange(3), **params)
    assert result["entry/yaw_request_rate"] == pytest.approx(1.0)
    assert result["entry/four_ready_rate"] == pytest.approx(4 / 7)
    assert result["entry/dwell_completion_rate"] == pytest.approx(2 / 3)
    assert result["entry/yaw_request_count"] == 3
    assert result["entry/four_to_transition_count"] == 2
    assert result["exit/yaw"] == 1
    assert result["exit/transition_timeout"] == 1
    assert sum(result[f"exit/{reason}"] for reason in fsm.transition_exit_reasons) == 2


@pytest.mark.parametrize("clearance", [0., .5, .79, 1.])
def test_stalled_transition_has_no_pose_income_and_handoff_does_not_drop(clearance):
    rewards, state, env, command = _reward_scene()
    for progress in env.lift_progress.values():
        progress.fill_(clearance)
    for elapsed in (0., .5, 1., 2.9):
        env.common_step_counter += 1
        command.state_time.fill_(elapsed)
        transition = _pose_rewards(rewards, env)
        assert torch.equal(transition, torch.zeros_like(transition))
        assert torch.allclose(transition[:, 0], transition[:, 1])
    # Identical physical pose, changed FSM gate: no acquisition reward cliff.
    command.fsm_state[:] = torch.tensor([state.YAW_POS, state.YAW_NEG])
    env.common_step_counter += 1
    yaw = _pose_rewards(rewards, env)
    assert (yaw.sum(0) >= transition.sum(0)).all()
    assert torch.allclose(yaw[:, 0], yaw[:, 1])


def test_lift_progress_cannot_be_refarmed_after_lowering_or_yaw_chatter():
    rewards, state, env, command = _reward_scene()
    progress_reward = rewards["TransitionProgress"](None, env)
    args = dict(asset_cfg=SimpleNamespace(name="pos"), asset_cfg_mirror=SimpleNamespace(name="neg"),
                wheel_radius=.091, target_clearance=.05, fsm_command_name="yaw_rate_cmd",
                support_sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
                support_sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]),
                com_asset_cfg=SimpleNamespace(name="robot"),
                com_asset_cfg_mirror=SimpleNamespace(name="robot"))

    def step(clearance, states):
        env.common_step_counter += 1
        command.fsm_state[:] = torch.tensor(states)
        for progress in env.lift_progress.values():
            progress.fill_(clearance)
        return progress_reward(env, **args)

    trans = [state.TRANSITION_POS, state.TRANSITION_NEG]
    yaw = [state.YAW_POS, state.YAW_NEG]
    assert torch.equal(step(0., trans), torch.zeros(2))
    assert (step(.5, trans) > 0).all()
    for _ in range(20):
        assert (step(.5, trans) <= 0).all()
    assert (step(.2, trans) <= 0).all()
    assert torch.equal(step(.5, trans), torch.zeros(2))
    assert torch.equal(step(.8, yaw), torch.zeros(2))
    assert torch.equal(step(.8, trans), torch.zeros(2))
    assert (step(1., trans) > 0).all()
    progress_reward.reset(torch.tensor([0]))
    result = step(.5, trans)
    assert result[0] == 0 and result[1] == 0
    result = step(.75, trans)
    assert result[0] > 0 and result[1] == 0


def test_zero_selected_support_blocks_lift_and_com_acquisition_for_both_signs():
    rewards, _, env, _ = _reward_scene()
    term = rewards["TransitionProgress"](None, env)
    args = dict(asset_cfg=SimpleNamespace(name="pos"), asset_cfg_mirror=SimpleNamespace(name="neg"),
                wheel_radius=.091, target_clearance=.05, fsm_command_name="yaw_rate_cmd",
                support_sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
                support_sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]),
                com_asset_cfg=SimpleNamespace(name="robot"),
                com_asset_cfg_mirror=SimpleNamespace(name="robot"))
    forces = env.scene.sensors["contact_forces"].data.net_forces_w
    forces.zero_()
    forces[0, [1, 2], 2] = 80.0  # Load the opposite diagonal only.
    forces[1, [0, 3], 2] = 80.0
    distance = torch.full((2,), .32)
    projection = torch.full((2,), -1.0)
    rewards["_yaw_support_geometry"] = lambda env, cfg: (distance, projection, torch.ones(2))

    assert torch.equal(term(env, **args), torch.zeros(2))
    for progress in env.lift_progress.values():
        progress.fill_(1.0)
    distance.zero_()
    projection.fill_(.5)
    env.common_step_counter += 1
    assert torch.equal(term(env, **args), torch.zeros(2))

    # Restoring the selected support can pay only the remaining new maximum.
    forces.zero_()
    forces[0, [0, 3], 2] = 80.0
    forces[1, [1, 2], 2] = 80.0
    env.common_step_counter += 1
    credit = term(env, **args) * env.step_dt
    assert torch.allclose(credit, torch.full((2,), 14.0))
    forces.zero_()
    env.common_step_counter += 1
    assert torch.equal(term(env, **args), torch.zeros(2))
    forces[0, [0, 3], 2] = 80.0
    forces[1, [1, 2], 2] = 80.0
    env.common_step_counter += 1
    assert torch.equal(term(env, **args), torch.zeros(2))


def test_return_progress_and_stability_keep_their_original_values():
    rewards, state, env, command = _reward_scene()
    progress_reward = rewards["TransitionProgress"](None, env)
    args = dict(asset_cfg=SimpleNamespace(name="pos"), asset_cfg_mirror=SimpleNamespace(name="neg"),
                wheel_radius=.091, target_clearance=.05, fsm_command_name="yaw_rate_cmd",
                support_sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
                support_sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]),
                com_asset_cfg=SimpleNamespace(name="robot"),
                com_asset_cfg_mirror=SimpleNamespace(name="robot"))
    command.fsm_state.fill_(state.RETURN_TO_4)
    for progress in env.lift_progress.values():
        progress.fill_(1.)
    assert torch.equal(progress_reward(env, **args), torch.zeros(2))
    command.just_switched.fill_(False)
    command.state_time.fill_(.5)
    env.common_step_counter += 1
    for progress in env.lift_progress.values():
        progress.fill_(.5)
    assert torch.allclose(progress_reward(env, **args), torch.full((2,), .495))
    stability = rewards["four_stand_stability"](env, target_height=.49, fsm_command_name="yaw_rate_cmd")
    assert torch.equal(stability, torch.full((2,), .5))
    command.fsm_state.fill_(state.FOUR_STAND)
    env.common_step_counter += 1
    assert torch.equal(progress_reward(env, **args), torch.zeros(2))
    assert torch.equal(rewards["four_stand_stability"](
        env, target_height=.49, fsm_command_name="yaw_rate_cmd"), torch.ones(2))


def test_failure_cost_is_one_episode_scale_event_and_waiting_beats_early_failure():
    rewards, _, env, _ = _reward_scene()
    _load_nodes(REWARDS_PATH, {"fsm_failure_cost"}, rewards)
    env.termination_manager = SimpleNamespace(terminated=torch.tensor([True, False]))
    pulse = rewards["fsm_failure_cost"](env)
    assert torch.equal(pulse * env.step_dt, torch.tensor([1.0, 0.0]))

    # Even at the former -19/s from tracking, lift and support load, the
    # episode-scale cost makes trying for another second better than failing
    # immediately. The current transition pose terms no longer pay this cost.
    gamma = .99
    steps = int(1.0 / env.step_dt)
    early_failure = -60.0
    continued_attempt = sum(-19.0 * env.step_dt * gamma**step for step in range(steps))
    continued_attempt -= 60.0 * gamma**steps
    assert continued_attempt > early_failure


def test_safe_recovery_entry_cost_is_one_pulse():
    rewards, state, env, command = _reward_scene()
    _load_nodes(REWARDS_PATH, {"safe_recovery_entry"}, rewards)
    command.fsm_state.fill_(state.SAFE_RECOVERY)
    command.just_switched.fill_(True)
    env.common_step_counter += 1
    assert torch.equal(
        rewards["safe_recovery_entry"](env, "yaw_rate_cmd") * env.step_dt,
        torch.ones(2),
    )
    command.just_switched.fill_(False)
    env.common_step_counter += 1
    assert torch.equal(rewards["safe_recovery_entry"](env, "yaw_rate_cmd"), torch.zeros(2))


def test_watchdog_survives_chatter_pauses_in_yaw_and_resets_only_at_maneuver_end():
    module = _load_fsm_module()
    fsm = module.YawFSMVectorized(2, four_stand_ready_dwell=0.0)
    namespace = {"torch": torch, "ManagerBasedRLEnv": object, "VQRFsmState": module.VQRFsmState}
    _load_nodes(FSM_PATH.with_name("terminations.py"), {"fsm_transition_timeout"}, namespace)
    env = SimpleNamespace(num_envs=2, device="cpu", command_manager=SimpleNamespace(get_term=lambda _: fsm))
    false = torch.zeros(2, dtype=torch.bool)
    true = ~false
    command = torch.tensor([.2, -.2])
    for _ in range(400):
        ready = (fsm.fsm_state != 2) & (fsm.fsm_state != 4)
        fsm.update(command, ready, ready, false, false)
        if namespace["fsm_transition_timeout"](env).all():
            break
    else:
        pytest.fail("Chatter escaped the cumulative transition deadline")
    assert torch.all(fsm.state_time < .11)
    assert torch.all(fsm.transition_time >= 3.)
    assert torch.allclose(fsm.transition_time[0], fsm.transition_time[1])
    fsm.reset()
    for _ in range(6):
        fsm.update(command, true, true, false, false)
    budget = fsm.transition_time.clone()
    for _ in range(500):
        fsm.update(command, true, true, false, false)
        assert not namespace["fsm_transition_timeout"](env).any()
    assert torch.equal(fsm.transition_time, budget)
    # Both command exits and unsafe priority still end the maneuver.
    fsm.update(-command, true, true, false, torch.tensor([False, True]))
    assert fsm.fsm_state.tolist() == [5, 6]
    assert not fsm.transition_time.any()


def _attempt_recorder(env):
    namespace = {"torch": torch}
    _load_nodes(REWARDS_PATH, {"_fsm_attempt_telemetry"}, namespace)

    def record(states, terminal=False, failed=False, quality=1.):
        states = torch.tensor(states)
        env.reset_buf = torch.full((env.num_envs,), terminal)
        env.reset_terminated = torch.full((env.num_envs,), failed)
        gates = dict(b_trans=(states == 1) | (states == 3), b_yaw=(states == 2) | (states == 4),
                     b_safe=states == 6, support_diagonal=torch.tensor([1, -1, 1]))
        namespace["_fsm_attempt_telemetry"](env, gates, torch.full((env.num_envs,), quality),
                                           torch.full((env.num_envs,), quality))
    return record


@pytest.mark.parametrize("phase", [0, 1, 2])
def test_curriculum_counts_failed_attempts_even_with_no_yaw_or_a_prior_success(monkeypatch, phase):
    curriculum = _load_curriculums_module(monkeypatch)
    env = _fsm_checkpoint_env(curriculum, 10000)
    params = _fsm_curriculum_params()
    params.update(min_directional_episodes=2, required_consecutive_windows=1,
                  min_clearance_stage_steps=0, min_yaw_stage_steps=0)
    env.cfg.curriculum.task_levels.params = params
    env._yaw_fsm_task_curriculum_phase = phase
    env.episode_length_buf.fill_(100)
    record = _attempt_recorder(env)
    # First maneuver succeeds, then another times out without ever entering YAW.
    record([1, 3, 0])
    record([2, 4, 0])
    record([5, 5, 0])
    record([0, 0, 0])
    record([1, 3, 0])
    record([1, 3, 0], terminal=True, failed=True)
    # Favorable YAW-only tracking/drift must not bypass the attempt denominator.
    for suffix, index in (("pos", 0), ("neg", 1)):
        for field, value in (("yaw_samples", 1), ("command_abs_sum", .2),
                             ("yaw_abs_error_sum", 0.), ("drift_sum", 0.), ("drift_samples", 1)):
            buffer = torch.zeros(3)
            buffer[index] = value
            setattr(env, f"_yaw_fsm_{suffix}_{field}", buffer)
    result = curriculum.yaw_fsm_task_levels(env, torch.arange(3), **params)
    assert result["pos/score"] == .5 and result["neg/score"] == .5
    assert result["window_passed"] == 0
    assert result["phase"] == phase and result["lift_stage"] == 0 and result["yaw_stage"] == 0
    assert not env._yaw_fsm_attempt_diagonal.any()
    assert not env._yaw_fsm_pos_transition_attempted.any()


@pytest.mark.parametrize("phase", [0, 1, 2])
def test_curriculum_successful_attempts_still_promote(monkeypatch, phase):
    curriculum = _load_curriculums_module(monkeypatch)
    env = _fsm_checkpoint_env(curriculum, 10000)
    params = _fsm_curriculum_params()
    params.update(min_directional_episodes=1, required_consecutive_windows=1,
                  min_clearance_stage_steps=0, min_yaw_stage_steps=0)
    env.cfg.curriculum.task_levels.params = params
    env._yaw_fsm_task_curriculum_phase = phase
    env.episode_length_buf.fill_(100)
    record = _attempt_recorder(env)
    record([1, 3, 0])
    record([2, 4, 0], terminal=True)  # Healthy episode truncation is not a failure.
    for suffix, index in (("pos", 0), ("neg", 1)):
        for field, value in (("yaw_samples", 1), ("command_abs_sum", .2), ("yaw_abs_error_sum", 0.),
                             ("drift_sum", 0.), ("drift_samples", 1)):
            buffer = torch.zeros(3)
            buffer[index] = value
            setattr(env, f"_yaw_fsm_{suffix}_{field}", buffer)
    result = curriculum.yaw_fsm_task_levels(env, torch.arange(3), **params)
    assert result["window_passed"] == 1
    assert result["pos/score"] == 1 and result["neg/score"] == 1
    assert result[{0: "lift_stage", 1: "phase", 2: "yaw_stage"}[phase]] == (2 if phase == 1 else 1)


def test_reacquisition_timeout_revokes_provisional_success_and_counts_chatter_once(monkeypatch):
    env = _fsm_checkpoint_env(_load_curriculums_module(monkeypatch), 0)
    record = _attempt_recorder(env)
    for _ in range(10):
        record([1, 3, 0])
        record([2, 4, 0])
    record([1, 3, 0], terminal=True, failed=True)
    assert env._yaw_fsm_pos_transition_attempted.tolist() == [1, 0, 0]
    assert env._yaw_fsm_neg_transition_attempted.tolist() == [0, 1, 0]
    assert not env._yaw_fsm_pos_transition_succeeded.any()
    assert not env._yaw_fsm_neg_pose_succeeded.any()


def test_safe_recovery_fails_attempt_without_requiring_episode_termination(monkeypatch):
    env = _fsm_checkpoint_env(_load_curriculums_module(monkeypatch), 0)
    record = _attempt_recorder(env)
    record([1, 3, 0])
    record([2, 4, 0])
    record([6, 6, 0])  # Phase C safety handles this without reset_terminated.
    assert env._yaw_fsm_pos_transition_attempted.sum() == 1
    assert env._yaw_fsm_neg_transition_attempted.sum() == 1
    assert not env._yaw_fsm_pos_transition_succeeded.any()
    assert not env._yaw_fsm_neg_pose_succeeded.any()
