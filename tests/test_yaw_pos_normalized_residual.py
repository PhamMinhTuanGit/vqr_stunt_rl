"""Certificate tolerance, contact geometry and mode checks for the opt-in cost."""

from types import SimpleNamespace as NS

import pytest
import torch

from test_yaw_pos_differential import _rolling_env
from test_yaw_pos_task import EntityCfg, _reward_module


def _configs(env, relative_std=.25, speed_std=.05):
    supports = ["FL_WHEEL", "HR_WHEEL"]
    wheels = ["FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL"]
    support_asset = EntityCfg("robot", body_names=supports)
    support_joints = EntityCfg("robot", joint_names=supports)
    support_joints.joint_ids = [0, 3]
    params = {
        "support_asset_cfg": support_asset,
        "support_joint_cfg": support_joints,
        "support_sensor_cfg": EntityCfg("contact_forces", body_names=supports),
        "command_name": "yaw_rate_cmd", "wheel_radius": .091,
        "relative_std": relative_std, "speed_std": speed_std,
    }
    env.reward_manager = NS(get_term_cfg=lambda name: NS(params=params))
    all_joints = EntityCfg("robot", joint_names=wheels)
    all_joints.joint_ids = [0, 1, 2, 3]
    return {
        "sensor_cfg": EntityCfg("contact_forces", body_names=wheels),
        "body_asset_cfg": EntityCfg("robot", body_names=wheels),
        "joint_asset_cfg": all_joints, "wheel_radius": .091,
        "command_name": "yaw_rate_cmd", "yaw_reference": 1., "deadband": .1,
    }


def _residual(env, values):
    data = env.scene["robot"].data
    data.body_ang_vel_w[:, [0, 3], 1] = (
        data.body_link_lin_vel_w[:, [0, 3], 0] - torch.tensor(values)
    ) / .091
    data.joint_vel[:, [0, 3]] = data.body_ang_vel_w[:, [0, 3], 1]


@pytest.mark.parametrize("factor,expected", [(0., 0.), (.5, 0.), (1., 0.), (1.5, .125), (2., .5), (3., 1.5)])
def test_same_dimensionless_excess_at_relative_and_absolute_tolerances(monkeypatch, factor, expected):
    reward = _reward_module(monkeypatch)
    env = _rolling_env((.2, 1.))
    configs = _configs(env)
    # Targets are +/-0.06 and +/-0.3 m/s, so the certificate tolerances
    # respectively use the 0.05 floor and the 25% relative branch.
    _residual(env, [[factor * .05, 0.], [factor * .075, 0.]])
    result = reward.yaw_pos_rolling_wheel_slip(env, **configs, normalized_excess=True)
    assert result.tolist() == pytest.approx([expected, expected], abs=1.e-6)


def test_uses_current_certificate_tolerance_and_worse_support_wheel(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _rolling_env((1.,))
    configs = _configs(env, relative_std=.5, speed_std=.1)
    _residual(env, [[.15, -.45]])  # Current tolerance 0.15, excesses 0 and 2.
    result = reward.yaw_pos_rolling_wheel_slip(env, **configs, normalized_excess=True)
    assert result.item() == pytest.approx(1.5, abs=1.e-6)


def test_keeps_neutral_raw_penalty_and_command_scaling(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _rolling_env((.2, 0., .1))
    configs = _configs(env)
    _residual(env, [[.1, .15], [.1, .15], [.1, .15]])
    monkeypatch.setattr(reward, "_yaw_command_penalty_scale", lambda *_: torch.full((3,), .3))
    original = reward.yaw_pos_rolling_wheel_slip(env, **configs)
    explicit_original = reward.yaw_pos_rolling_wheel_slip(env, **configs, normalized_excess=False)
    normalized = reward.yaw_pos_rolling_wheel_slip(env, **configs, normalized_excess=True)
    torch.testing.assert_close(original, explicit_original, rtol=0, atol=0)
    assert original.tolist() == pytest.approx([.3 * (.1**2 + .15**2)] * 3)
    torch.testing.assert_close(normalized[1:], original[1:], rtol=0, atol=0)
    assert normalized[0].item() == pytest.approx(.3 * 1.5, abs=1.e-6)


def test_contact_loss_and_degenerate_target_are_not_penalized_as_rolling(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _rolling_env((1., 1.))
    configs = _configs(env)
    _residual(env, [[.3, 0.], [.3, 0.]])
    env.scene.sensors["contact_forces"].data.net_forces_w[0, 0] = 0.
    env.scene["robot"].data.body_pos_w[1, 0, :2] = 0.
    result = reward.yaw_pos_rolling_wheel_slip(env, **configs, normalized_excess=True)
    assert result.tolist() == pytest.approx([0., 0.], abs=1.e-6)


def test_active_excess_uses_supports_and_retains_raw_all_wheel_telemetry(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _rolling_env((1.,))
    configs = _configs(env)
    env.scene["robot"].data.body_ang_vel_w[0, 1, 1] = -1. / .091
    original = reward.yaw_pos_rolling_wheel_slip(env, **configs)
    normalized = reward.yaw_pos_rolling_wheel_slip(env, **configs, normalized_excess=True)
    assert original.item() == pytest.approx(1.)
    assert normalized.item() == pytest.approx(0.)
    assert env._yaw_pos_rolling_error_sum.item() == pytest.approx(.5)
