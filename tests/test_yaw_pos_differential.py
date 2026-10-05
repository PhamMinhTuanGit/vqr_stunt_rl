"""Physical differential rolling, neutral anchors and append-only POS resume."""

import ast
import math
import sys
import copy
import importlib
from types import SimpleNamespace as NS

import pytest
import torch

from test_yaw_pos_task import CONFIG, EntityCfg, MDP, ROOT, _env, _load, _reward_module
from test_yaw_curriculum import _ConfigManager, _load_curriculums_module, _load_train_curriculum_helpers, _online_dr_manager
from test_yaw_pos_ground_heading import _quaternion


class Scene(dict):
    pass


def _rolling_env(commands=(1.,)):
    env = _env(commands)
    n = len(commands)
    data = env.scene["robot"].data
    data.body_quat_w = torch.tensor([1., 0., 0., 0.]).expand(n, 4, 4).clone()
    data.body_pos_w[:, 0, 1], data.body_pos_w[:, 3, 1] = .3, -.3
    data.body_link_lin_vel_w = torch.zeros(n, 4, 3)
    data.body_ang_vel_w = torch.zeros(n, 4, 3)
    data.body_com_lin_vel_w = torch.zeros(n, 4, 3)
    data.root_pos_w = torch.zeros(n, 3)
    data.root_lin_vel_w = torch.zeros(n, 3)
    data.joint_vel = torch.zeros(n, 4)
    for i, command in enumerate(commands):
        data.body_link_lin_vel_w[i, 0, 0] = -.3 * command
        data.body_link_lin_vel_w[i, 3, 0] = .3 * command
    data.body_ang_vel_w[..., 1] = data.body_link_lin_vel_w[..., 0] / .091
    data.joint_vel[:] = data.body_ang_vel_w[..., 1]
    env.scene = Scene(env.scene)
    forces = torch.zeros(n, 4, 3)
    forces[..., 2] = 10.
    env.scene.sensors = {"contact_forces": NS(data=NS(net_forces_w=forces))}
    env.num_envs, env.device, env.step_dt, env.common_step_counter = n, "cpu", .02, 0
    env.episode_length_buf = torch.full((n,), 2, dtype=torch.long)
    return env


def _differential(reward, env):
    support = EntityCfg("robot", body_names=["FL_WHEEL", "HR_WHEEL"])
    joints = EntityCfg("robot", joint_names=["FL_WHEEL", "HR_WHEEL"])
    joints.joint_ids = [0, 3]
    sensor = EntityCfg("contact_forces", body_names=["FL_WHEEL", "HR_WHEEL"])
    return reward.yaw_pos_rolling_tracking(
        env, support, joints, sensor, "yaw_rate_cmd", .1, .091, settle_time=0.,
    )


@pytest.mark.parametrize("flip_wheel", [0, 3])
def test_axle_and_joint_sign_reversal_leave_reward_unchanged(monkeypatch, flip_wheel):
    reward = _reward_module(monkeypatch)
    env = _rolling_env()
    original = _differential(reward, env)
    original_target = env._yaw_pos_differential_metrics_current["differential_fl_target"].clone()
    # Rotation around X by pi reverses local +Y; physical world motion stays unchanged.
    env.scene["robot"].data.body_quat_w[:, flip_wheel] = torch.tensor([0., 1., 0., 0.])
    env.scene["robot"].data.joint_vel[:, flip_wheel] *= -1
    assert torch.allclose(_differential(reward, env), original)
    assert original.item() == pytest.approx(1.)
    assert env._yaw_pos_differential_pass_sum.item() == 2.
    if flip_wheel == 0:
        assert torch.allclose(env._yaw_pos_differential_metrics_current["differential_fl_target"], -original_target)


@pytest.mark.parametrize("yaw,phase", [(0., 1.1), (.8, 0.), (2., 2.7)])
def test_targets_follow_geometry_under_heading_translation_and_wheel_phase(monkeypatch, yaw, phase):
    module = _load(monkeypatch, "yaw_pos_kinematics", MDP / "yaw_pos_kinematics.py")
    # Unequal lever arms require unequal speeds; neither is a fixed motor target.
    positions = torch.tensor([[[.12, .2, .091], [-.08, -.4, .091]]], dtype=torch.float64)
    velocities = torch.tensor([[[-.4, .24, 0.], [.8, -.16, 0.]]], dtype=torch.float64)
    orientation = _quaternion(0., phase, yaw)[:, None, :].expand(1, 2, 4)
    rotation = torch.tensor([[math.cos(yaw), -math.sin(yaw), 0.],
                             [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]], dtype=torch.float64)
    translation = torch.tensor([2., -5., 0.], dtype=torch.float64)
    target, measured, valid = module.differential_rolling_kinematics(
        orientation, positions @ rotation.T + translation, velocities @ rotation.T,
        translation[None, :2], torch.tensor([2.], dtype=torch.float64),
    )
    assert valid.all()
    assert target.tolist()[0] == pytest.approx([-.4, .8], abs=1e-12)
    assert torch.allclose(target, measured, atol=1e-12)


@pytest.mark.parametrize("failure", ["motor_stopped", "pivot", "sliding", "contact", "invalid_axle", "zero_target", "wrong_direction"])
def test_one_wheel_cannot_be_compensated_by_the_other(monkeypatch, failure):
    reward = _reward_module(monkeypatch)
    env = _rolling_env()
    data = env.scene["robot"].data
    if failure == "motor_stopped":
        data.joint_vel[:, 0] = 0.
    elif failure == "pivot":
        data.body_link_lin_vel_w[:, 0] = 0.
        data.body_ang_vel_w[:, 0] = 0.
    elif failure == "sliding":
        data.body_ang_vel_w[:, 0] = 0.
    elif failure == "contact":
        env.contacts[:, 0] = 0.
    elif failure == "invalid_axle":
        data.body_quat_w[:, 0] = _quaternion(math.pi / 2, 0., 0.).float()
    elif failure == "zero_target":
        data.body_pos_w[:, 0, 1] = 0.
    else:
        data.body_link_lin_vel_w[:, 0] *= -1
    score = _differential(reward, env)
    assert torch.isfinite(score).all()
    if failure in ("contact", "invalid_axle", "zero_target"):
        assert score.item() == 0.
    elif failure in ("motor_stopped", "sliding"):
        # Ground tracking does not use motor magnitude or duplicate slip shaping;
        # the unchanged strict certificate still rejects both cases.
        assert score.item() == pytest.approx(1.)
    else:
        assert 0. < score.item() < 1.
    assert env._yaw_pos_differential_pass_sum.item() == 0.


@pytest.mark.parametrize("error", [.3, .6, 1., 3.])
def test_large_error_retains_directional_tracking_gradient(monkeypatch, error):
    reward = _reward_module(monkeypatch)
    target = torch.tensor([[-.2, .2]], dtype=torch.float64)
    measured = (target + torch.tensor([[error, 0.]])).requires_grad_()
    score = reward.rolling_tracking_score(target, measured, torch.full_like(target, .05))
    gradient, = torch.autograd.grad(score.sum(), measured)
    assert score.item() > 0.
    assert gradient[0, 0] < -1.e-4  # Retain a correcting slope even at 60x scale.
    assert gradient[0, 1] == 0.
    smaller = measured.detach().clone()
    smaller[0, 0] -= .01
    improved = reward.rolling_tracking_score(target, smaller, torch.full_like(target, .05))
    assert improved > score
    gaussian = torch.exp(-((measured - target) / .05).square()).amin(1)
    assert score > 1000. * gaussian


def test_worse_wheel_controls_tracking_independently_of_motor_activity(monkeypatch):
    reward = _reward_module(monkeypatch)
    target = torch.tensor([[-.2, .2]])

    def score(errors):
        return reward.rolling_tracking_score(target, target + torch.tensor([errors]),
                                             torch.full_like(target, .05)).item()

    assert score((.3, .1)) == pytest.approx(score((.3, 0.)))
    assert score((.29, .1)) > score((.3, .1))
    assert score((.1, .3)) == pytest.approx(score((.3, .1)))
    assert score((0., 0.)) == 1.


def test_signed_participation_rejects_wrong_direction_and_uses_worse_wheel(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _rolling_env((1., 1., 1., 1., 0.))
    data = env.scene["robot"].data
    data.body_link_lin_vel_w[:, 0, 0] = torch.tensor([-.3, -.15, .3, -.6, -.3])
    data.body_link_lin_vel_w[:, 3, 0] = torch.tensor([.3, .6, .3, .3, .3])
    data.joint_vel[:] = 0.  # Participation deliberately ignores motor activity.
    support = EntityCfg("robot", body_names=["FL_WHEEL", "HR_WHEEL"])
    sensor = EntityCfg("contact_forces", body_names=["FL_WHEEL", "HR_WHEEL"])
    result = reward.yaw_pos_signed_ground_participation(env, support, sensor, "yaw_rate_cmd", .1)
    assert result.tolist() == pytest.approx([1., .5, 0., 1., 0.])
    env.contacts[0, 3] = 0.
    assert reward.yaw_pos_signed_ground_participation(env, support, sensor, "yaw_rate_cmd", .1)[0] == 0.


def test_active_com_stationarity_uses_mass_weighted_body_velocity_and_excludes_neutral(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _rolling_env((1., 1., 0.))
    env.scene["robot"].data.body_com_lin_vel_w[1, :, 0] = .1
    env.scene["robot"].data.root_lin_vel_w[:] = 4.  # Root velocity is not the CoM measurement.
    result = reward.yaw_pos_active_com_stationary(env, "yaw_rate_cmd", .1, std=.05)
    assert result.tolist() == pytest.approx([1., .2, 0.])
    env.scene["robot"].data.body_com_lin_vel_w[0, :, 0] = torch.tensor([.1, 0., 0., 0.])
    env.scene["robot"].root_physx_view.get_masses = lambda: torch.tensor([[3., 1., 1., 1.]]).expand(3, 4)
    assert reward.yaw_pos_active_com_stationary(env, "yaw_rate_cmd", .1, std=.05)[0].item() == pytest.approx(.5)


def test_dense_shaping_does_not_change_strict_certificate(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _rolling_env((1.,) * 128)
    data = env.scene["robot"].data
    generator = torch.Generator().manual_seed(23)
    measured = torch.randn((128, 2), generator=generator) * .3
    motor = torch.randn((128, 2), generator=generator) * 3.
    slip = torch.randn((128, 2), generator=generator) * .08
    target = torch.tensor([[-.3, .3]]).expand(128, 2)
    scale = .075
    measured[0] = target[0]
    motor[0] = target[0] / .091
    slip[0] = 0.
    measured[1] = .5 * target[1]
    motor[1] = .5 * target[1] / .091
    slip[1] = 0.
    for i, body in enumerate((0, 3)):
        data.body_link_lin_vel_w[:, body, 0] = measured[:, i]
        data.joint_vel[:, body] = motor[:, i]
        data.body_ang_vel_w[:, body, 1] = (measured[:, i] - slip[:, i]) / .091
    _differential(reward, env)
    expected = ((measured * target > 0.) & (measured.abs() >= .5 * target.abs())
                & (.091 * motor.abs() / target.abs() >= .5) & (slip.abs() <= scale)).all(1)
    assert torch.equal(env._yaw_pos_differential_metrics_current["differential_pass"], expected)
    assert expected[:2].all() and 0 < expected.sum() < 128


def _hold(reward, env):
    sensor = EntityCfg("contact_forces", body_names=["FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL"])
    result = reward.yaw_pos_neutral_position(env, sensor, "yaw_rate_cmd", .1, contact_dwell=.04)
    env.episode_length_buf += 1
    return result


def test_neutral_velocity_rewards_stillness_during_landing(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _rolling_env((0., 0., 1.))
    env.scene["robot"].data.root_lin_vel_w[1, 0] = -.1
    score = reward.yaw_pos_neutral_velocity(env, "yaw_rate_cmd", .1)
    assert score[0] == 1 and score[1] < .02 and score[2] == 0


def test_anchor_waits_for_contact_then_survives_drift_and_contact_loss(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _rolling_env((0.,))
    data = env.scene["robot"].data
    env.contacts[:, 1] = 0.
    assert _hold(reward, env).item() == 0.
    env.contacts[:, 1] = 1.
    assert _hold(reward, env).item() == 0.
    assert _hold(reward, env).item() == 1.
    anchor = env._yaw_pos_neutral_anchor.clone()
    data.root_pos_w[:, 0] = -.10
    env.contacts[:, 1] = 0.
    assert _hold(reward, env).item() < .02
    env.contacts[:, 1] = 1.
    assert _hold(reward, env).item() < .02
    assert torch.equal(env._yaw_pos_neutral_anchor, anchor)
    assert env._yaw_pos_neutral_hold_pass_sum.item() == 1.
    env.command[:] = 1.
    assert _hold(reward, env).item() == 0.
    assert not env._yaw_pos_neutral_anchored.any()
    env.command[:] = 0.
    assert _hold(reward, env).item() == 0.
    assert _hold(reward, env).item() == 1.
    assert env._yaw_pos_neutral_anchor[0, 0].item() == pytest.approx(-.1)


def test_playback_reset_discards_anchor_without_curriculum(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _rolling_env((0.,))
    _hold(reward, env)
    _hold(reward, env)
    env.scene["robot"].data.root_pos_w[:, 0] = 5.
    env.episode_length_buf[:] = 1
    assert _hold(reward, env).item() == 0.
    assert _hold(reward, env).item() == 1.
    assert env._yaw_pos_neutral_anchor[0, 0].item() == 5.


def _curriculum_env(monkeypatch):
    baseline = _load_curriculums_module(monkeypatch)
    monkeypatch.setitem(sys.modules, "yaw_pos_test_package.curriculums", baseline)
    module = _load(monkeypatch, "yaw_pos_curriculums", MDP / "yaw_pos_curriculums.py")
    levels = (.25, .4, .55, .7, .85, 1., 1.25, 1.5, 1.75, 2., 2.5, 3.)
    params = dict(
        command_name="yaw_rate_cmd", clearance_levels=(.05, .10, .15, .20), yaw_rate_levels=levels,
        dr_scale_levels=(.3, .4, .55, .7, .85, 1.) + (1.,) * 6,
        tracking_ratio_thresholds=(.3, .35, .4, .45, .5, .55) + (.55,) * 6,
        edge_tracking_ratio_thresholds=(.2, .25, .3, .35, .4, .45) + (.45,) * 6,
        lift_reward_name="lift_clearance", balance_reward_name="balance", yaw_reward_name="gated_yaw_tracking",
        torso_contact_termination_name="torso_contact", minimum_base_height=.35,
        support_threshold=.85, lift_progress_threshold=.80, balance_threshold=.75, yaw_threshold=.65,
        min_evaluated_episodes=1, required_success_rate=.85, required_consecutive_windows=1,
        min_clearance_stage_steps=0, min_yaw_stage_steps=0, certify_behavior=True,
    )
    rewards = _ConfigManager({
        "lift_clearance": NS(weight=3., params={"target_clearance": .20}),
        "balance": NS(weight=2., params={}),
        "gated_yaw_tracking": NS(weight=8., params={"target_clearance": .20}),
        "neutral_landing_progress": NS(weight=3., params={"target_clearance": .20}),
    })
    rewards._episode_sums = {"balance": torch.tensor([4., 4.])}
    command = NS(cfg=NS(yaw_rate_range=(0., .25)), reset=lambda ids: None)
    env = NS(num_envs=2, device="cpu", step_dt=.02, common_step_counter=100,
             episode_length_buf=torch.tensor([100, 100]), reward_manager=rewards,
             command_manager=NS(get_term=lambda _: command), event_manager=_online_dr_manager(),
             termination_manager=NS(get_term=lambda _: torch.zeros(2, dtype=torch.bool)),
             cfg=NS(curriculum=NS(task_levels=NS(func=module.yaw_pos_task_levels, params=params))),
             _yaw_task_curriculum_stage=3, _yaw_task_curriculum_yaw_stage=5)
    return module, env, params


def _evidence(env, differential=100., neutral=100., neutral_samples=100):
    env._yaw_base_height_min = torch.ones(2)
    # Row 1 is a neutral-only episode and must still contribute hold evidence.
    for name, values in {
        "_yaw_support_score": [100., 0.], "_yaw_gate_open": [100., 0.],
        "_yaw_lift_min_progress": [100., 0.], "_yaw_active_yaw_score": [100., 0.],
        "_yaw_command_abs": [100., 0.], "_yaw_rate_abs_error": [0., 0.],
        "_yaw_edge_command_abs": [100., 0.], "_yaw_edge_rate_abs_error": [0., 0.],
        "_yaw_pos_differential_pass": [differential, 0.],
    }.items():
        setattr(env, f"{name}_sum", torch.tensor(values))
        setattr(env, f"{name}_samples", torch.tensor([100, 0]))
    env._yaw_tracking_metric_samples = torch.tensor([100, 0])
    env._yaw_edge_tracking_samples = torch.tensor([100, 0])
    env._yaw_pos_neutral_hold_pass_sum = torch.tensor([0., neutral])
    env._yaw_pos_neutral_hold_pass_samples = torch.tensor([0, neutral_samples])


@pytest.mark.parametrize("differential,neutral,samples,expected", [
    (100., 100., 100, True), (89., 100., 100, False), (100., 89., 100, False), (100., 0., 0, False),
])
def test_promotion_requires_differential_and_neutral_including_neutral_only_episodes(
    monkeypatch, differential, neutral, samples, expected,
):
    module, env, params = _curriculum_env(monkeypatch)
    _evidence(env, differential, neutral, samples)
    result = module.yaw_pos_task_levels(env, [0, 1], **params)
    assert bool(result["window_passed"].item()) is expected
    assert env._yaw_task_curriculum_yaw_stage == (6 if expected else 5)
    assert env.command_manager.get_term("").cfg.yaw_rate_range == (0., 1.25 if expected else 1.)
    assert env._yaw_pos_neutral_hold_pass_samples.sum().item() == 0


def test_new_stages_progress_to_three_with_full_dr(monkeypatch):
    module, env, params = _curriculum_env(monkeypatch)
    for limit in (1.25, 1.5, 1.75, 2., 2.5, 3., 3.):
        _evidence(env)
        result = module.yaw_pos_task_levels(env, [0, 1], **params)
        assert result["yaw_limit"].item() == limit
        assert result["dr_scale"].item() == 1.


def test_selected_checkpoint_restores_existing_stages_and_elapsed(monkeypatch):
    _, env, params = _curriculum_env(monkeypatch)
    helpers = _load_train_curriculum_helpers()
    # The schema and scalar state of the selected model_25500.pt.
    payload = {"version": 1, "values": {
        "_yaw_task_curriculum_stage": 3, "_yaw_task_curriculum_yaw_stage": 5,
        "_yaw_task_curriculum_consecutive_passes": 0,
    }, "stage_elapsed_steps": 2541,
        "yaw_rate_levels": list(params["yaw_rate_levels"][:6]), "dr_scale_levels": list(params["dr_scale_levels"][:6])}
    assert helpers._restore_yaw_curriculum_state(env, {"yaw_curriculum": payload})
    assert env._yaw_task_curriculum_stage == 3 and env._yaw_task_curriculum_yaw_stage == 5
    assert env.common_step_counter - env._yaw_task_curriculum_stage_start_step == 2541
    assert env.command_manager.get_term("").cfg.yaw_rate_range == (0., 1.)
    assert env.event_manager.get_term_cfg("randomize_apply_external_force_torque").params["force_range"] == (-10., 10.)
    payload["yaw_rate_levels"][2] = .56
    assert not helpers._restore_yaw_curriculum_state(env, {"yaw_curriculum": payload})


def test_pending_certification_survives_checkpoint_and_selected_reset(monkeypatch):
    module, env, params = _curriculum_env(monkeypatch)
    params["min_evaluated_episodes"] = 3
    _evidence(env)
    env._yaw_pos_neutral_anchor = torch.ones(2, 2)
    env._yaw_pos_neutral_anchored = torch.ones(2, dtype=torch.bool)
    module.yaw_pos_task_levels(env, [1], **params)
    assert env._yaw_pos_neutral_anchored.tolist() == [True, False]
    helpers = _load_train_curriculum_helpers()
    payload = helpers._export_yaw_curriculum_state(env)
    _, resumed, resumed_params = _curriculum_env(monkeypatch)
    resumed_params["min_evaluated_episodes"] = 3
    assert helpers._restore_yaw_curriculum_state(resumed, {"yaw_curriculum": payload})
    assert resumed._yaw_task_curriculum_certification_neutral_hold_pass_samples == 100
    assert resumed._yaw_task_curriculum_certification_neutral_hold_pass_passed == 100
    assert resumed._yaw_task_curriculum_evaluated == 0


def test_config_stage_extension_is_pos_only():
    namespace = {}
    for filename in ("yaw_env_cfg.py", "yaw_env_pos_cfg.py"):
        tree = ast.parse((CONFIG / filename).read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and node.targets[0].id in {
                "YAW_RATE_LEVELS", "ONLINE_DR_SCALE_LEVELS", "YAW_TRACKING_RATIO_THRESHOLDS",
                "YAW_EDGE_TRACKING_RATIO_THRESHOLDS", "POS_YAW_RATE_LEVELS", "POS_DR_SCALE_LEVELS",
                "POS_TRACKING_RATIO_THRESHOLDS", "POS_EDGE_TRACKING_RATIO_THRESHOLDS",
            }:
                exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
    assert namespace["YAW_RATE_LEVELS"][-1] == 1.
    assert namespace["POS_YAW_RATE_LEVELS"][-1] == 3.
    assert all(len(namespace[name]) == 12 for name in (
        "POS_YAW_RATE_LEVELS", "POS_DR_SCALE_LEVELS", "POS_TRACKING_RATIO_THRESHOLDS", "POS_EDGE_TRACKING_RATIO_THRESHOLDS",
    ))


@pytest.mark.skipif(
    not (ROOT / "logs/rsl_rl/vqr_wheel_yaw_flat_pos/2026-10-01_18-28-44/model_25500.pt").exists(),
    reason="Selected local POS checkpoint required",
)
def test_real_pos_resume_preserves_policy_optimizer_std_and_curriculum(monkeypatch, tmp_path):
    import yaml
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict

    checkpoint = ROOT / "logs/rsl_rl/vqr_wheel_yaw_flat_pos/2026-10-01_18-28-44/model_25500.pt"
    monkeypatch.syspath_prepend(str(ROOT / "scripts/reinforcement_learning/rsl_rl"))
    loader = importlib.import_module("yaw_pos_adaptation")
    with (checkpoint.parent / "params/agent.yaml").open() as stream:
        config = yaml.load(stream, Loader=loader._ConfigLoader)
    config["device"] = "cpu"
    obs = TensorDict({"policy": torch.zeros(2, 55), "critic": torch.zeros(2, 83)}, batch_size=[2])
    vec = NS(num_envs=2, num_actions=16, device="cpu", get_observations=lambda: obs)
    runner = OnPolicyRunner(vec, copy.deepcopy(config), log_dir=str(tmp_path), device="cpu")
    infos = runner.load(str(checkpoint), load_optimizer=True, map_location="cpu")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for name, tensor in saved["model_state_dict"].items():
        assert torch.equal(runner.alg.policy.state_dict()[name], tensor), name
    optimizer = runner.alg.optimizer.state_dict()
    assert optimizer["param_groups"] == saved["optimizer_state_dict"]["param_groups"]
    for index, state in saved["optimizer_state_dict"]["state"].items():
        for name, value in state.items():
            actual = optimizer["state"][index][name]
            assert torch.equal(actual, value) if isinstance(value, torch.Tensor) else actual == value
    assert runner.current_learning_iteration == 25500
    _, env, _ = _curriculum_env(monkeypatch)
    assert _load_train_curriculum_helpers()._restore_yaw_curriculum_state(env, infos)
    assert env._yaw_task_curriculum_stage == 3 and env._yaw_task_curriculum_yaw_stage == 5
    assert env.common_step_counter - env._yaw_task_curriculum_stage_start_step == 2541
