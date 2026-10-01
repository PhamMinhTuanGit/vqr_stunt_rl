"""Phase-invariant rolling contracts without launching Isaac Sim."""

import math
from types import SimpleNamespace as NS

import pytest
import torch

from test_yaw_pos_task import MDP, _load, _reward_module, _env


def _kinematics(monkeypatch):
    return _load(monkeypatch, "wheel_contact_kinematics", MDP / "wheel_contact_kinematics.py")


def _multiply(left, right):
    w1, xyz1 = left[..., :1], left[..., 1:]
    w2, xyz2 = right[..., :1], right[..., 1:]
    return torch.cat((w1 * w2 - (xyz1 * xyz2).sum(-1, keepdim=True),
                      w1 * xyz2 + w2 * xyz1 + torch.linalg.cross(xyz1, xyz2)), dim=-1)


@pytest.mark.parametrize("phase_degrees", [0, 90, 180, 270])
@pytest.mark.parametrize("speed", [-0.4, 0., 0.2, 0.8])
@pytest.mark.parametrize("steering_degrees", [0, 37, 90])
def test_no_slip_at_all_wheel_phases(monkeypatch, phase_degrees, speed, steering_degrees):
    helper = _kinematics(monkeypatch)
    phase, steering = math.radians(phase_degrees), math.radians(steering_degrees)
    # URDF joint axis is -Y: positive forward speed implies negative qdot.
    spin = torch.tensor([math.cos(phase / 2), 0., -math.sin(phase / 2), 0.], dtype=torch.float64)
    heading = torch.tensor([math.cos(steering / 2), 0., 0., math.sin(steering / 2)], dtype=torch.float64)
    quaternion = _multiply(heading, spin)
    tangent = torch.tensor([math.cos(steering), math.sin(steering), 0.], dtype=torch.float64)
    axle = torch.tensor([-math.sin(steering), math.cos(steering), 0.], dtype=torch.float64)
    result = helper.wheel_contact_velocities(quaternion, speed * tangent, (speed / 0.091) * axle, 0.091)
    assert result[0].abs().item() < 1e-12
    assert result[1].abs().item() < 1e-12


def test_contact_velocity_equivalence_on_tilted_ground_and_wheel(monkeypatch):
    helper = _kinematics(monkeypatch)
    torch.manual_seed(11)
    orientation = torch.randn(32, 4, dtype=torch.float64)
    orientation /= orientation.norm(dim=-1, keepdim=True)
    axle = helper.rotate_vector(orientation, orientation.new_tensor((0., 1., 0.)))
    normal = torch.tensor([0.2, -0.3, 1.], dtype=torch.float64)
    normal /= normal.norm()
    tangent = torch.linalg.cross(axle, normal.expand_as(axle))
    tangent /= tangent.norm(dim=-1, keepdim=True)
    radial = normal - (normal * axle).sum(-1, keepdim=True) * axle
    radial /= radial.norm(dim=-1, keepdim=True)
    omega = torch.randn(32, 3, dtype=torch.float64)
    velocity = torch.randn(32, 3, dtype=torch.float64)
    contact_velocity = velocity + torch.linalg.cross(omega, -0.091 * radial)
    expected = (contact_velocity * tangent).sum(-1)
    actual = helper.wheel_contact_velocities(orientation, velocity, omega, 0.091, normal)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=1e-12)


def test_reward_and_critic_share_residual_contact_mask_and_order(monkeypatch):
    reward = _reward_module(monkeypatch)
    helper = _kinematics(monkeypatch)
    env = _env([0.4, 0.])
    env.num_envs = 2
    data = env.scene["robot"].data
    phases = torch.tensor([0., 90., 180., 270.]) * math.pi / 180.
    data.body_quat_w = torch.stack((torch.cos(phases / 2), torch.zeros(4),
                                    -torch.sin(phases / 2), torch.zeros(4)), -1).repeat(2, 1, 1)
    data.body_link_lin_vel_w = torch.zeros(2, 4, 3)
    data.body_link_lin_vel_w[..., 0] = 0.2 + torch.tensor([0., 0.1, -0.2, 0.3])
    data.body_link_lin_vel_w[..., 1] = torch.tensor([0.01, 0.02, 0.03, 0.04])
    data.body_ang_vel_w = torch.zeros(2, 4, 3)
    data.body_ang_vel_w[..., 1] = 0.2 / 0.091
    # Poison both CoM velocity and joint velocity to ensure center and full omega are used.
    data.body_lin_vel_w = torch.full((2, 4, 3), 100.)
    data.joint_vel.fill_(100.)
    forces = torch.zeros(2, 4, 3)
    forces[..., 2] = torch.tensor([2., 3., 1., 4.])  # Equality to threshold is off contact.
    class Scene(dict):
        pass
    env.scene = Scene(env.scene)
    env.scene.sensors = {"contact_forces": NS(data=NS(net_forces_w=forces))}
    cfg = dict(sensor_cfg=NS(name="contact_forces", body_ids=[3, 0, 2, 1]),
               body_asset_cfg=NS(name="robot", body_ids=[3, 0, 2, 1]),
               joint_asset_cfg=NS(name="robot", joint_ids=[3, 0, 2, 1]), wheel_radius=0.091)
    observation = helper.rolling_lateral_contact_velocity(env, **cfg)
    assert observation.shape == (2, 8)
    torch.testing.assert_close(observation[0], torch.tensor([0.3, 0.04, 0., 0.01, 0., 0., 0.1, 0.02]),
                               atol=1e-6, rtol=1e-6)
    # Actual command relief from the baseline, unchanged: scale = 1 - abs(command)/reference.
    monkeypatch.setattr(reward, "_yaw_command_penalty_scale", lambda e, *_: 1. - e.command[:, 0].abs())
    score = reward.yaw_pos_rolling_wheel_slip(env, **cfg, command_name="yaw_rate_cmd", yaw_reference=1., deadband=0.1)
    expected = observation.reshape(2, 4, 2)[..., 0].square().sum(1) * torch.tensor([0.6, 1.])
    torch.testing.assert_close(score, expected)
    assert score.tolist() == pytest.approx([0.06, 0.10], abs=1e-6)  # Neutral penalty remains active.
    assert env._yaw_pos_rolling_error_sum.tolist() == pytest.approx([0.4 / 3., 0.], abs=1e-6)
    assert env._yaw_pos_rolling_error_samples.tolist() == [1, 0]
    forces.zero_()
    assert torch.count_nonzero(helper.rolling_lateral_contact_velocity(env, **cfg)) == 0
    assert torch.count_nonzero(reward.yaw_pos_rolling_wheel_slip(
        env, **cfg, command_name="yaw_rate_cmd", yaw_reference=1., deadband=0.1
    )) == 0
    assert env._yaw_pos_rolling_error_samples.tolist() == [1, 0]


def test_no_contact_zeroes_reward_and_does_not_log_a_zero_sample(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _env([0.25])
    monkeypatch.setattr(reward, "contacted_wheel_velocities", lambda *args: (
        torch.ones(1, 4, 2), torch.zeros(1, 4, dtype=torch.bool)
    ))
    result = reward.yaw_pos_rolling_wheel_slip(env, None, None, None, 0.091, "yaw_rate_cmd", 1., 0.1)
    assert result.item() == 0.
    assert env._yaw_pos_rolling_error_samples.item() == 0


def test_parallel_axle_is_finite_and_nonpositive_radius_rejected(monkeypatch):
    helper = _kinematics(monkeypatch)
    h = 2.0**-0.5
    q = torch.tensor([h, h, 0., 0.])  # Local Y -> world Z.
    result = helper.wheel_contact_velocities(q, torch.ones(3), torch.ones(3), 0.091)
    assert torch.isfinite(result).all() and result[0] == 0.
    with pytest.raises(ValueError, match="wheel_radius"):
        helper.wheel_contact_velocities(q, torch.zeros(3), torch.zeros(3), 0.)
