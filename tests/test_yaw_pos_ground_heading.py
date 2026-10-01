"""Ground-heading, tilt rejection, POS oscillation and wheel telemetry contracts."""

import math
from types import SimpleNamespace as NS

import pytest
import torch

from test_yaw_pos_task import EntityCfg, MDP, _env, _load, _reward_module


def _kinematics(monkeypatch):
    return _load(monkeypatch, "yaw_pos_kinematics", MDP / "yaw_pos_kinematics.py")


def _quaternion(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = [f(a / 2) for a in (roll, pitch, yaw) for f in (math.cos, math.sin)]
    return torch.tensor([[cy*cp*cr + sy*sp*sr, cy*cp*sr - sy*sp*cr,
                          cy*sp*cr + sy*cp*sr, sy*cp*cr - cy*sp*sr]], dtype=torch.float64)


@pytest.mark.parametrize("roll,pitch,yaw", [(0., 0., 0.), (.12, .3, .6), (-.4, -.5, -2.)])
@pytest.mark.parametrize("roll_dot,pitch_dot", [(1.2, 0.), (0., -3.3), (1.2, -3.3)])
def test_pure_euler_roll_pitch_has_zero_ground_heading_rate(monkeypatch, roll, pitch, yaw, roll_dot, pitch_dot):
    helper = _kinematics(monkeypatch)
    body_x = torch.tensor([[math.cos(yaw)*math.cos(pitch), math.sin(yaw)*math.cos(pitch), -math.sin(pitch)]])
    heading_y = torch.tensor([[-math.sin(yaw), math.cos(yaw), 0.]])
    omega_w = (roll_dot*body_x + pitch_dot*heading_y).double()
    # Physical Euler-rate conversion for fixed heading, not an expectation
    # computed from the helper's own yaw-rate formula.
    rate, valid = helper.ground_heading_yaw_rate(_quaternion(roll, pitch, yaw), omega_w)
    assert valid.item() and abs(rate.item()) < 1e-7


@pytest.mark.parametrize("roll,pitch,yaw", [(0., 0., 0.), (.12, .3, .6), (-.4, -.5, -2.)])
@pytest.mark.parametrize("speed", [-.7, 0., .4, .7])
def test_pure_world_z_yaw_is_exact_even_when_tilted(monkeypatch, roll, pitch, yaw, speed):
    helper = _kinematics(monkeypatch)
    omega_w = torch.tensor([[0., 0., speed]], dtype=torch.float64)
    rate, valid = helper.ground_heading_yaw_rate(_quaternion(roll, pitch, yaw), omega_w)
    assert valid.item() and rate.item() == pytest.approx(speed, abs=1e-12)


def test_heading_rate_agrees_with_wrapped_heading_finite_difference(monkeypatch):
    helper = _kinematics(monkeypatch)
    q = _quaternion(.23, -.31, math.pi - 1e-7)
    omega_w = torch.tensor([[.3, -.6, .7]], dtype=torch.float64)
    dt = 1e-6
    angle = omega_w.norm(dim=-1) * dt
    dq = torch.cat((torch.cos(angle / 2).unsqueeze(-1),
                    omega_w / omega_w.norm(dim=-1, keepdim=True) * torch.sin(angle / 2).unsqueeze(-1)), dim=-1)
    w1, v1, w2, v2 = dq[:, :1], dq[:, 1:], q[:, :1], q[:, 1:]
    next_q = torch.cat((w1*w2 - (v1*v2).sum(-1, keepdim=True),
                        w1*v2 + w2*v1 + torch.linalg.cross(v1, v2)), dim=-1)
    x0, _, _ = helper.ground_heading_axes(q)
    x1, _, _ = helper.ground_heading_axes(next_q)
    yaw_delta = torch.atan2(x0[:, 0]*x1[:, 1] - x0[:, 1]*x1[:, 0], (x0*x1).sum(-1))
    actual, _ = helper.ground_heading_yaw_rate(q, omega_w)
    assert actual.item() == pytest.approx((yaw_delta / dt).item(), abs=1e-6)


@pytest.mark.parametrize("command", [.4, .7])
def test_pitch_coupling_cannot_match_yaw_reward_or_curriculum_error(monkeypatch, command):
    reward = _reward_module(monkeypatch)
    env = _env([command])
    env.command_manager.get_term = lambda _: NS(cfg=NS(yaw_rate_range=(0., command)))
    data = env.scene["robot"].data
    roll = math.radians(7.)
    data.root_quat_w[:] = _quaternion(roll, 0., 0.).float()
    pitch_dot = -command / math.sin(roll)
    data.root_ang_vel_w[:] = torch.tensor([0., pitch_dot, 0.])
    # Old body-Z score was perfect in this state. Poison it deliberately.
    data.root_ang_vel_b[:] = torch.tensor([0., math.cos(roll)*pitch_dot, command])
    env.clearance[:, [1, 2]] = 1.
    score = reward.yaw_pos_gated_tracking(
        env, "yaw_rate_cmd", EntityCfg("contact_forces", body_names=["FL_WHEEL", "HR_WHEEL"]),
        EntityCfg("robot", body_names=["FR_WHEEL", "HL_WHEEL"]), .091, .20, .20, .1, neutral_std=.3,
    )
    # Preserve the requested Gaussian: zero actual yaw has the same small
    # baseline as standing still, never the full reward for matching command.
    assert score.item() == pytest.approx(math.exp(-(command / .20)**2), abs=1e-7)
    assert env._yaw_active_yaw_score_sum.item() == pytest.approx(score.item())
    assert env._yaw_rate_abs_error_current.item() == pytest.approx(command)
    assert env._yaw_rate_abs_error_sum.item() == pytest.approx(command)
    assert env._yaw_edge_rate_abs_error_sum.item() == pytest.approx(command)
    assert env._yaw_tracking_metric_samples.item() == 1


def test_vertical_body_x_cannot_receive_neutral_or_active_yaw_credit(monkeypatch):
    helper = _kinematics(monkeypatch)
    q = _quaternion(0., math.pi / 2, 0.)
    rate, valid = helper.ground_heading_yaw_rate(q, torch.tensor([[0., 0., .7]], dtype=torch.float64))
    assert not valid.item() and rate.item() == 0.
    x, y, _ = helper.ground_heading_axes(q)
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    reward = _reward_module(monkeypatch)
    env = _env([0., .4])
    env.scene["robot"].data.root_quat_w[:] = q.float()
    env.clearance[:, [1, 2]] = 1.
    score = reward.yaw_pos_gated_tracking(
        env, "yaw_rate_cmd", EntityCfg("contact_forces", body_names=["FL_WHEEL", "HR_WHEEL"]),
        EntityCfg("robot", body_names=["FR_WHEEL", "HL_WHEEL"]), .091, .20, .20, .1,
    )
    assert score.tolist() == [0., 0.]


def test_command_error_uses_same_heading_helper_and_preserves_transfer_default(monkeypatch):
    import sys
    import types
    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)
    baseline = types.ModuleType("yaw_pos_test_package.commands")
    class BaseCommand:
        def _update_metrics(self):
            self.metrics["error_yaw_rate"] = (self.robot.data.root_ang_vel_b[:, 2] - self._command[:, 0]).abs()
    baseline.YawRateCommand = BaseCommand
    baseline.YawRateCommandCfg = type("YawRateCommandCfg", (), {})
    monkeypatch.setitem(sys.modules, baseline.__name__, baseline)
    module = _load(monkeypatch, "yaw_pos_commands", MDP / "yaw_pos_commands.py")
    term = object.__new__(module.YawPosCommand)
    term.cfg = NS(use_ground_heading_rate=True)
    term.robot = _env([.4]).scene["robot"]
    roll = math.radians(7.)
    term.robot.data.root_quat_w[:] = _quaternion(roll, 0., 0.).float()
    term.robot.data.root_ang_vel_w[:] = torch.tensor([0., -.4 / math.sin(roll), 0.])
    term.robot.data.root_ang_vel_b[:, 2] = .4
    term._command = torch.tensor([[.4]])
    term.metrics = {}
    term._update_metrics()
    assert term.metrics["error_yaw_rate"].item() == pytest.approx(.4)
    assert module.YawPosCommandCfg.use_ground_heading_rate is False
    term.cfg.use_ground_heading_rate = False
    term._update_metrics()
    assert term.metrics["error_yaw_rate"].item() == 0.


@pytest.mark.parametrize("separation,expected_weighted", [(.45, 0.), (.40, -.25), (.36, -.81)])
def test_normalized_collapse_at_requested_separations(monkeypatch, separation, expected_weighted):
    reward = _reward_module(monkeypatch)
    env = _env([.4, 0., .1, -.4])
    data = env.scene["robot"].data
    data.body_pos_w[:, 0, 1] = separation / 2
    data.body_pos_w[:, 3, 1] = -separation / 2
    raw_cost = reward.yaw_pos_support_y_collapse_l2(
        env, EntityCfg("robot", body_names=["FL_WHEEL", "HR_WHEEL"]), .45, "yaw_rate_cmd", .1, .05,
    )
    assert (-.25 * raw_cost).tolist() == pytest.approx([expected_weighted, 0., 0., 0.], abs=5e-7)


def test_angular_xy_penalty_is_world_frame_pos_only_and_ignores_yaw(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _env([.4, 0., .1, -.4, .4])
    data = env.scene["robot"].data
    data.root_ang_vel_w[:] = torch.tensor([2., 3., 100.])
    data.root_ang_vel_w[4] = torch.tensor([0., 0., .7])
    data.root_ang_vel_b.fill_(1000.)
    cost = reward.yaw_pos_body_angular_xy_l2(env, "yaw_rate_cmd", .1)
    assert (-.1 * cost).tolist() == pytest.approx([-1.3, 0., 0., 0., 0.])
    assert env._yaw_pos_body_omega_xy_squared_sum.tolist() == [13., 0., 0., 0., 0.]
    assert env._yaw_pos_body_omega_xy_squared_samples.tolist() == [1, 0, 0, 0, 1]


@pytest.mark.parametrize("phase_degrees", [0., 90., 180., 270.])
def test_telemetry_distinguishes_motor_and_ground_rolling_speed(monkeypatch, phase_degrees):
    helper = _kinematics(monkeypatch)
    env = _env([.4])
    robot, data = env.scene["robot"], env.scene["robot"].data
    phase = math.radians(phase_degrees)
    data.body_quat_w = torch.tensor([[[math.cos(phase / 2), 0., -math.sin(phase / 2), 0.]]]).repeat(1, 4, 1)
    data.body_link_lin_vel_w = torch.zeros(1, 4, 3)
    data.body_link_lin_vel_w[0, 0, 0] = -.18
    data.body_link_lin_vel_w[0, 3, 0] = .20
    data.joint_vel[0, 0], data.joint_vel[0, 3] = 0., -1.7
    data.root_quat_w[:] = _quaternion(.12, 0., 0.).float()
    data.root_ang_vel_w[:] = torch.tensor([0., -2., 0.])
    motion = helper.yaw_pos_motion_telemetry(robot, [0, 3], [0, 3])
    assert set(motion) == set(helper.POS_MOTION_METRICS)
    assert motion["support_fl_motor_speed"].item() == 0.
    assert motion["support_hr_motor_speed"].item() == pytest.approx(-1.7)
    assert motion["support_fl_ground_rolling_speed"].item() == pytest.approx(-.18)
    assert motion["support_hr_ground_rolling_speed"].item() == pytest.approx(.20)
    assert abs(motion["true_heading_rate"].item()) < 1e-6
    assert motion["pitch_rate"].item() == pytest.approx(-2.)
    assert motion["abs_pitch_rate"].item() == pytest.approx(2.)


def test_invalid_rolling_axis_is_flagged_in_telemetry(monkeypatch):
    helper = _kinematics(monkeypatch)
    env = _env([.4])
    data = env.scene["robot"].data
    h = 2.**-.5
    data.body_quat_w = torch.tensor([[[h, h, 0., 0.]]]).repeat(1, 4, 1)
    data.body_link_lin_vel_w = torch.ones(1, 4, 3)
    motion = helper.yaw_pos_motion_telemetry(env.scene["robot"], [0, 3], [0, 3])
    assert motion["support_fl_rolling_direction_valid"].item() == 0.
    assert motion["support_fl_ground_rolling_speed"].item() == 0.


def test_reward_logs_pos_motion_without_wheel_drive_credit_or_reset_corruption(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _env([.4, 0.])
    data = env.scene["robot"].data
    data.root_ang_vel_w[:, 2] = torch.tensor([.4, 0.])
    data.body_quat_w = data.root_quat_w[:, None, :].expand(-1, 4, -1).clone()
    data.body_link_lin_vel_w = torch.zeros(2, 4, 3)
    data.body_pos_w[:, 0, 1], data.body_pos_w[:, 3, 1] = .225, -.225
    env.clearance[:, [1, 2]] = 1.
    kwargs = dict(
        support_asset_cfg=EntityCfg("robot", body_names=["FL_WHEEL", "HR_WHEEL"]),
        support_joint_cfg=NS(name="robot", joint_ids=[0, 3]),
    )
    score = reward.yaw_pos_gated_tracking(
        env, "yaw_rate_cmd", EntityCfg("contact_forces", body_names=["FL_WHEEL", "HR_WHEEL"]),
        EntityCfg("robot", body_names=["FR_WHEEL", "HL_WHEEL"]), .091, .20, .20, .1, **kwargs,
    )
    reward.yaw_pos_support_line(env, kwargs["support_asset_cfg"], .20, "yaw_rate_cmd", .1)
    # This round deliberately adds no motor-based reward/gate.
    assert score.tolist() == pytest.approx([1., 1.])
    assert env._yaw_pos_support_fl_motor_speed_sum.tolist() == [0., 0.]
    assert env._yaw_pos_true_heading_rate_samples.tolist() == [1, 0]
    data.root_ang_vel_w.fill_(1000.)
    data.body_pos_w.fill_(1000.)
    assert env._yaw_pos_motion_metrics_current["world_yaw_rate"].tolist() == pytest.approx([.4, 0.])
    assert env._yaw_pos_geometry_metrics_current["support_y_separation"].tolist() == pytest.approx([.45, .45])


@pytest.mark.parametrize("motion_kind,expected_pass", [("pitch_coupling", False), ("world_yaw", True)])
def test_real_curriculum_certifies_heading_rotation_only(monkeypatch, motion_kind, expected_pass):
    import sys
    from test_yaw_curriculum import _load_curriculums_module

    reward = _reward_module(monkeypatch)
    env = _env([.4])
    data = env.scene["robot"].data
    roll = math.radians(7.)
    data.root_quat_w[:] = _quaternion(roll, 0., 0.).float()
    data.root_ang_vel_b[:, 2] = .4
    data.root_ang_vel_w[:] = torch.tensor([0., -.4/math.sin(roll), 0.] if motion_kind == "pitch_coupling" else [0., 0., .4])
    env.clearance[:, [1, 2]] = 1.
    command_term = NS(cfg=NS(yaw_rate_range=(0., .4)))
    env.command_manager.get_term = lambda _: command_term
    support = EntityCfg("contact_forces", body_names=["FL_WHEEL", "HR_WHEEL"])
    lifted = EntityCfg("robot", body_names=["FR_WHEEL", "HL_WHEEL"])
    score = reward.yaw_pos_gated_tracking(env, "yaw_rate_cmd", support, lifted, .091, .20, .20, .1)
    reward.yaw_pos_lift_clearance(env, lifted, support, .091, .20, "yaw_rate_cmd", .1)

    class ConfigManager:
        def __init__(self, configs):
            self.configs = configs
        def get_term_cfg(self, name):
            return self.configs[name]
        def set_term_cfg(self, name, cfg):
            self.configs[name] = cfg

    env.num_envs, env.device, env.step_dt, env.common_step_counter = 1, "cpu", .02, 100
    env.episode_length_buf = torch.ones(1, dtype=torch.long)
    env._yaw_base_height_min = torch.tensor([.49])
    env.termination_manager = NS(get_term=lambda _: torch.zeros(1, dtype=torch.bool))
    env.reward_manager = ConfigManager({
        "lift_clearance": NS(weight=3., params={"target_clearance": .20}),
        "gated_yaw_tracking": NS(weight=8., params={"target_clearance": .20}),
        "balance": NS(weight=2., params={}),
        "neutral_landing_progress": NS(weight=3., params={"target_clearance": .20}),
    })
    env.reward_manager._episode_sums = {"balance": torch.tensor([.02*2.])}
    env.event_manager = ConfigManager({
        "randomize_apply_external_force_torque": NS(params={}), "randomize_actuator_gains": NS(params={}),
        "randomize_push_robot": NS(params={}),
        "randomize_reset_base": NS(params={"pose_range": {}, "velocity_range": {}}),
    })
    baseline = _load_curriculums_module(monkeypatch)
    monkeypatch.setitem(sys.modules, "yaw_pos_test_package.curriculums", baseline)
    wrapper = _load(monkeypatch, "yaw_pos_curriculums", MDP / "yaw_pos_curriculums.py")
    result = wrapper.yaw_pos_task_levels(
        env, [0], "yaw_rate_cmd", (.20,), (.4, .7), (.3, .4), (.3, .35), (.2, .25),
        "lift_clearance", "balance", "gated_yaw_tracking", "torso_contact", .35,
        .85, .80, .75, .65, 1, .85, 1, 0, 0,
    )
    assert bool(result["window_passed"].item()) is expected_pass
    assert bool(result["yaw_limit_advanced"].item()) is expected_pass
    assert result["yaw_score"].item() == pytest.approx(score.item())
    assert result["window_tracking_ratio"].item() == pytest.approx(1. if expected_pass else 0., abs=1e-6)
    for alias, original in (("heading_error", "mean_error_yaw_rate"), ("support_contact", "support_score"),
                            ("lift_progress", "lift_min_progress")):
        assert result[alias] is result[original]
    assert result["body_omega_xy"].item() == pytest.approx(result["body_omega_xy_squared"].sqrt().item())
