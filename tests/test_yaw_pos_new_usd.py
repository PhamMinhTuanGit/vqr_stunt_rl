"""New-USD geometry, target continuity and checkpoint/export regressions on CPU."""

import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from test_yaw_pos_task import CONFIG, MDP, ROOT, _env, _load, _reward_module, EntityCfg
from test_yaw_pos_posture_rewards import height_env


def direct_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def asset_profile():
    return direct_module(MDP / "yaw_pos_asset.py").inspect_pos_asset(
        str(ROOT / "deep_robotics_model/VQRWheel/VQRWheel_usd/VQRWheel.usd"))


def test_composed_new_usd_has_valid_mass_pose_and_collision_centers(asset_profile):
    p = asset_profile
    assert p["total_mass_kg"] == pytest.approx(36.58756, abs=1e-5)
    assert len(p["default_leg_positions"]) == 12 and len(p["usd_sha256"]) == 64
    q = np.asarray(p["default_leg_positions"])
    limits = np.asarray(p["leg_position_limits"])
    assert np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1])
    assert np.max(np.abs(np.array(p["neutral_wheel_centers"])[:, 2] + .49 - p["neutral_wheel_vertical_extents"])) < 1e-8
    assert p["wheel_center_offsets"]["FL_WHEEL"][1] == pytest.approx(.0303)
    assert p["wheel_center_offsets"]["HR_WHEEL"][1] == pytest.approx(-.0303)
    assert set(p["wheel_motor_axis_signs"].values()) == {-1.}
    json.dumps(p)


def test_new_usd_rejects_unreachable_standing_height():
    with pytest.raises(ValueError, match="Cannot stand"):
        direct_module(MDP / "yaw_pos_asset.py").inspect_pos_asset(
            str(ROOT / "deep_robotics_model/VQRWheel/VQRWheel_usd/VQRWheel.usd"), target_height=.65)


def test_adversarial_position_targets_obey_velocity_acceleration_and_hard_limits():
    limiter = direct_module(MDP / "yaw_pos_action_limits.py")
    torch.manual_seed(41)
    lower = torch.full((16, 12), -.785, dtype=torch.float64)
    upper = -lower
    q, v = torch.zeros_like(lower), torch.zeros_like(lower)
    for tick in range(1500):
        if tick % 31 == 0:
            goal = 100. * torch.randn_like(q)
        nq, nv = limiter.limit_position_target(goal, q, v, lower, upper, 2., 10., .005)
        assert (nq >= lower).all() and (nq <= upper).all()
        assert nv.abs().max() <= 2. + 1e-12
        assert (nv - v).abs().max() <= .05 + 1e-12
        torch.testing.assert_close(nq - q, .005 * nv, atol=1e-12, rtol=0.)
        q, v = nq, nv
    # Holding a limit must converge without a nonzero target velocity.
    for _ in range(1000):
        q, v = limiter.limit_position_target(upper, q, v, lower, upper, 2., 10., .005)
    torch.testing.assert_close(q, upper, atol=1e-8, rtol=0.)
    assert v.abs().max() < 1e-6


def test_wheel_target_rate_limit_applies_at_every_physics_tick():
    limiter = direct_module(MDP / "yaw_pos_action_limits.py")
    current = torch.zeros(4, dtype=torch.float64)
    for _ in range(4):
        following = limiter.limit_velocity_target(torch.full_like(current, 1000.), current, 58.9, 20., .005)
        assert (following - current).abs().max() <= .1 + 1e-12
        current = following
    torch.testing.assert_close(current, torch.full_like(current, .4))
    for _ in range(800):
        current = limiter.limit_velocity_target(torch.full_like(current, -1000.), current, 58.9, 20., .005)
    torch.testing.assert_close(current, torch.full_like(current, -58.9))


def test_asset_configuration_shares_one_reference_without_mutating_parent(asset_profile):
    tree = ast.parse((CONFIG / "yaw_env_pos_cfg.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "configure_pos_asset")
    original = NS(spawn=NS(usd_path="test.usd"), init_state=NS(pos=(0., 0., .45), joint_pos={".*": 0.}))
    original.copy = lambda: copy.deepcopy(original)
    policy = NS(actions=NS(), base_ang_vel=NS(), projected_gravity=NS(), joint_vel=NS(), joint_pos=NS())
    cfg = NS(scene=NS(robot=original), rewards=NS(lift=NS(params={"wheel_radius": .1})),
             actions=NS(joint_pos=NS(max_target_velocity=2., max_target_acceleration=10.),
                        joint_vel=NS(max_target_velocity=58.9, max_target_acceleration=20.)),
             observations=NS(policy=policy, critic=NS(actions=NS())), sim=NS(dt=.005), decimation=4)
    namespace = dict(inspect_pos_asset=lambda *a, **kw: copy.deepcopy(asset_profile),
                     TARGET_BASE_HEIGHT=.49, POS_YAW_HEIGHT=.455, POS_YAW_RATE_LEVELS=(.25, 3.),
                     SUPPORT_Y_MIN_SEPARATION=.2, YAW_DEADBAND=.1, Unoise=NS, NoiseModelWithAdditiveBiasCfg=NS,
                     pos_actions=NS(applied_target_history=object()))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "configuration", "exec"), namespace)
    namespace["configure_pos_asset"](cfg)
    assert original.init_state.pos == (0., 0., .45) and original.init_state.joint_pos == {".*": 0.}
    assert cfg.scene.robot.init_state.pos[2] == .49
    assert list(cfg.scene.robot.init_state.joint_pos.values())[:12] == asset_profile["default_leg_positions"]
    assert cfg.yaw_pos_contract["action_offsets"][:12] == asset_profile["default_leg_positions"]
    assert cfg.yaw_pos_contract["history"] == "normalized_applied_targets"
    assert cfg.actions.joint_pos.clip["FL_Knee_joint"][0] > .78
    assert cfg.observations.critic.actions.func is policy.actions.func


def test_mode_height_targets_and_hipx_free_band(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = height_env([0., .25, 3.], [.49, .455, .455])
    score = reward.yaw_pos_base_height_tracking(env, .49, .1, "yaw_rate_cmd", .1, active_target_height=.455)
    torch.testing.assert_close(score, torch.ones_like(score))
    cost = reward.yaw_pos_base_height_deficit_l2(env, "yaw_rate_cmd", .46, low_speed_multiplier=1.,
                                              active_minimum_height=.44)
    assert cost.count_nonzero() == 0
    env.scene["robot"].data.root_pos_w[:, 2] -= .06
    cost = reward.yaw_pos_base_height_deficit_l2(env, "yaw_rate_cmd", .46, low_speed_multiplier=1.,
                                              active_minimum_height=.44)
    assert cost[1] == cost[2] and cost[1] > 0
    data = env.scene["robot"].data
    data.joint_pos[:, 0] = torch.tensor([.41, -.49, .6])
    hip = reward.yaw_pos_hipx_deviation_l2(env, NS(name="robot", joint_ids=[0]), free_angle=.5)
    assert hip[0] == hip[1] == 0 and hip[2] > 0


def test_signed_motor_tracking_rewards_correct_direction(monkeypatch):
    reward = _reward_module(monkeypatch)
    target = torch.tensor([[-.05, .05]])
    motor_fraction = torch.ones_like(target)
    scale = torch.full_like(target, .015)
    good = reward.differential_rolling_score(target, target, torch.zeros_like(target), motor_fraction, scale, .5,
                                            motor_rolling_speed=target)
    wrong = reward.differential_rolling_score(target, target, torch.zeros_like(target), motor_fraction, scale, .5,
                                             motor_rolling_speed=-target)
    assert good.item() == 1. and wrong.item() < .2


def test_adaptive_yaw_shaping_retains_original_curriculum_score(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _env([.15, 3.])
    env.contacts[:] = torch.tensor([1., 0., 0., 1.])
    env.clearance[:, [1, 2]] = 1.
    env.scene["robot"].data.root_ang_vel_w[:, 2] = torch.tensor([0., 2.8])
    kwargs = dict(command_name="yaw_rate_cmd", support_sensor_cfg=EntityCfg("contact_forces", body_names=["FL_WHEEL", "HR_WHEEL"]),
                  lifted_asset_cfg=EntityCfg("robot", body_names=["FR_WHEEL", "HL_WHEEL"]),
                  wheel_radius=.091, target_clearance=.05, std=.20, deadband=.1)
    original = reward.yaw_pos_gated_tracking(env, **kwargs)
    old_sum = env._yaw_active_yaw_score_sum.clone()
    changed = reward.yaw_pos_gated_tracking(env, **kwargs, tracking_relative_std=.20, tracking_min_std=.04)
    assert changed[0] < original[0] and changed[1] > original[1]
    torch.testing.assert_close(env._yaw_active_yaw_score_sum, 2. * old_sum)


def test_differential_reward_includes_parent_rotation_and_retains_certificate_floor(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _env([.15])
    class Scene(dict):
        pass
    env.scene = Scene(env.scene)
    forces = torch.zeros(1, 4, 3)
    forces[..., 2] = 2.
    env.scene.sensors = {"contact_forces": NS(data=NS(net_forces_w=forces))}
    env.episode_length_buf, env.step_dt = torch.tensor([30]), .02
    robot = env.scene["robot"]
    data = robot.data
    data.body_pos_w[:, [0, 3], 1] = torch.tensor([.3, -.3])
    data.body_quat_w = torch.tensor([[[1., 0., 0., 0.]]]).repeat(1, 4, 1)
    data.body_link_lin_vel_w = torch.zeros(1, 4, 3)
    data.body_link_lin_vel_w[:, [0, 3], 0] = torch.tensor([-.045, .045])
    data.body_com_lin_vel_w = torch.zeros(1, 4, 3)
    data.body_ang_vel_w = torch.zeros(1, 6, 3)
    data.body_ang_vel_w[:, [0, 3], 1] = torch.tensor([-.045, .045]) / .091
    data.body_ang_vel_w[:, [4, 5], 1] = -.1
    data.joint_vel[:, [0, 3]] = -(data.body_ang_vel_w[:, [0, 3], 1] + .1)
    robot._yaw_pos_wheel_motor_axis_signs = -torch.ones(4)
    kwargs = dict(support_asset_cfg=EntityCfg("robot", body_names=["FL_WHEEL", "HR_WHEEL"]),
                  support_joint_cfg=NS(joint_ids=[0, 3]), support_sensor_cfg=EntityCfg("contact_forces", body_names=["FL_WHEEL", "HR_WHEEL"]),
                  support_parent_cfg=NS(body_ids=[4, 5]), command_name="yaw_rate_cmd", deadband=.1,
                  wheel_radius=.091, speed_std=.015, certification_speed_std=.05,
                  motor_tracking_weight=1., settle_time=0.)
    score = reward.yaw_pos_differential_rolling(env, **kwargs)
    assert score.item() == pytest.approx(1., abs=1e-6)
    assert env._yaw_pos_differential_pass_sum.item() == 1.
    # A 0.02 m/s slip exceeds the dense floor but remains below the unchanged certification floor.
    data.body_ang_vel_w[:, [0, 3], 1] -= .02 / .091
    data.joint_vel[:, [0, 3]] = -(data.body_ang_vel_w[:, [0, 3], 1] + .1)
    slipped = reward.yaw_pos_differential_rolling(env, **kwargs)
    assert slipped < score
    assert env._yaw_pos_differential_pass_sum.item() == 2.


def test_active_translation_anchor_latches_after_settling_and_resets_per_episode(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _env([.25, .25])
    env.episode_length_buf = torch.tensor([10, 10])
    env._yaw_pos_active_age = torch.tensor([.49, .50])
    data = env.scene["robot"].data
    data.body_com_lin_vel_w = torch.zeros(2, 4, 3)
    data.body_com_lin_vel_w[..., 0] = .1
    cost = reward.yaw_pos_active_translation(env, "yaw_rate_cmd", .1)
    assert cost[0] == 0 and cost[1] > 0
    assert env._yaw_pos_active_anchored.tolist() == [False, True]
    data.body_com_lin_vel_w.zero_()
    data.whole_body_com_xy[1, 0] = .05
    cost = reward.yaw_pos_active_translation(env, "yaw_rate_cmd", .1)
    assert cost[1] > 0
    env.episode_length_buf[1] = 0
    data.whole_body_com_xy[1, 0] = 100.
    reward.yaw_pos_active_translation(env, "yaw_rate_cmd", .1)
    assert env._yaw_pos_active_com_position_drift[1] == 0


def test_checkpoint_contract_and_initial_exploration(tmp_path):
    helper = direct_module(MDP / "yaw_pos_contract.py")
    policy = NS(noise_std_type="log", log_std=torch.nn.Parameter(torch.zeros(16)))
    helper.initialize_pos_exploration(policy)
    torch.testing.assert_close(policy.log_std.exp(), torch.tensor([.5] * 12 + [.3] * 4))
    path = tmp_path / "checkpoint.pt"
    contract = {"version": 2, "usd_sha256": "abc", "usd_path": "/one", "history": "normalized_applied_targets"}
    torch.save({"infos": {"yaw_pos_contract": contract}}, path)
    helper.validate_pos_checkpoint(path, {**contract, "usd_path": "/two"})
    with pytest.raises(ValueError, match="usd_sha256"):
        helper.validate_pos_checkpoint(path, {**contract, "usd_sha256": "changed"})
    torch.save({"infos": {}}, path)
    with pytest.raises(ValueError, match="from scratch"):
        helper.validate_pos_checkpoint(path, contract)


def test_onnx_export_matches_checkpoint_mean_and_carries_trained_limit(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    exporter = direct_module(ROOT / "scripts/tools/export_yaw_pos_policy.py")
    from rsl_rl.modules import ActorCritic
    from tensordict import TensorDict
    inputs = TensorDict({"policy": torch.zeros(7, 55), "critic": torch.zeros(7, 83)}, batch_size=[7])
    policy = ActorCritic(inputs, {"policy": ["policy"], "critic": ["critic"]}, 16,
                         actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                         noise_std_type="log", init_noise_std=.5).eval()
    actor = policy.actor
    contract = dict(version=2, empirical_normalization=False, activation="elu", actor_observations=55,
                    actor_hidden_dims=[512, 256, 128], actions=16, history="normalized_applied_targets")
    checkpoint = tmp_path / "policy.pt"
    torch.save({"model_state_dict": policy.state_dict(), "iter": 250,
                "infos": {"yaw_pos_contract": contract, "yaw_curriculum": {
                    "values": {"_yaw_task_curriculum_yaw_stage": 1}, "yaw_rate_levels": [.25, .4, 3.]}}}, checkpoint)
    output = tmp_path / "policy.onnx"
    metadata = exporter.export_checkpoint(checkpoint, output)
    assert metadata["trained_yaw_limit"] == .4
    obs = torch.randn(7, 55)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(session.run(None, {"obs": obs.numpy()})[0], actor(obs).detach().numpy(), atol=1e-6)
    assert json.loads(session.get_modelmeta().custom_metadata_map["yaw_pos_contract"]) == metadata
    assert json.loads(output.with_suffix(".contract.json").read_text()) == metadata
