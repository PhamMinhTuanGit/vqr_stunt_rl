"""Lightweight configuration and tensor checks for four-wheel rotation."""

from types import SimpleNamespace

import gymnasium as gym
import torch

from isaaclab.managers import SceneEntityCfg

import rl_training.tasks.manager_based.locomotion.pivot.config.wheeled.vqr  # noqa: F401
from rl_training.tasks.manager_based.locomotion.pivot.config.wheeled.vqr.four_wheel_rotate_env_cfg import (
    VQRFourWheelRotateEnvCfg,
)
from rl_training.tasks.manager_based.locomotion.pivot.mdp.rewards import lateral_wheel_slip
from rl_training.tasks.manager_based.locomotion.pivot.mdp.terminations import lost_wheel_contact


class _Scene(dict):
    def __init__(self, robot, sensor):
        super().__init__(robot=robot)
        self.sensors = {"contact_forces": sensor}


def _fake_env(wheel_velocity, contact_forces):
    num_envs = wheel_velocity.shape[0]
    identity_quat = torch.zeros((num_envs, 4, 4), dtype=wheel_velocity.dtype)
    identity_quat[..., 0] = 1.0
    robot = SimpleNamespace(
        data=SimpleNamespace(body_lin_vel_w=wheel_velocity, body_quat_w=identity_quat)
    )
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=contact_forces))
    return SimpleNamespace(
        scene=_Scene(robot, sensor),
        episode_length_buf=torch.full((num_envs,), 10),
        step_dt=0.02,
    )


def _entity_cfg(name):
    cfg = SceneEntityCfg(name)
    cfg.body_ids = [0, 1, 2, 3]
    return cfg


def test_config_and_registration():
    cfg = VQRFourWheelRotateEnvCfg()
    ranges = cfg.commands.yaw_rate_cmd.ranges

    assert "Pivot-VQR-FourWheel-Rotate-v0" in gym.registry
    assert ranges.lin_vel_x == (0.0, 0.0)
    assert ranges.lin_vel_y == (0.0, 0.0)
    assert ranges.ang_vel_z == (-0.3, 0.3)
    assert len(cfg.actions.leg_positions.joint_names) == 12
    assert len(cfg.actions.wheel_velocities.joint_names) == 4
    assert cfg.rewards.track_ang_vel_z_exp.weight == 3.0
    assert cfg.rewards.track_lin_vel_xy_exp.weight == 1.0
    assert cfg.rewards.flat_orientation_l2.weight == -1.0
    assert cfg.rewards.lateral_wheel_slip.weight == -1.0
    for gait_reward in (
        "feet_air_time_ang_z_M20",
        "rotation_gait_status",
        "rotation_gait_symmetry",
        "feet_gait",
    ):
        assert not hasattr(cfg.rewards, gait_reward)


def test_lateral_wheel_slip_shape_and_zero_lateral_velocity():
    wheel_velocity = torch.zeros((3, 4, 3))
    wheel_velocity[..., 0] = 2.0  # valid longitudinal rolling is not penalized
    contact_forces = torch.zeros((3, 4, 3))
    contact_forces[..., 2] = 10.0
    env = _fake_env(wheel_velocity, contact_forces)

    slip = lateral_wheel_slip(
        env,
        sensor_cfg=_entity_cfg("contact_forces"),
        asset_cfg=_entity_cfg("robot"),
        threshold=2.0,
    )

    assert slip.shape == (3,)
    torch.testing.assert_close(slip, torch.zeros(3))


def test_persistent_lost_wheel_contact():
    wheel_velocity = torch.zeros((3, 4, 3))
    contact_forces = torch.zeros((3, 4, 3))
    contact_forces[..., 2] = 10.0
    env = _fake_env(wheel_velocity, contact_forces)
    sensor_cfg = _entity_cfg("contact_forces")

    # Randomized PPO episode counters do not bypass the physical reset grace.
    env.episode_length_buf[:] = torch.tensor([73, 327, 19])
    contact_forces.zero_()
    for _ in range(5):
        during_grace = lost_wheel_contact(env, sensor_cfg=sensor_cfg, threshold=2.0)
        torch.testing.assert_close(during_grace, torch.tensor([False, False, False]))
        env.episode_length_buf += 1
    torch.testing.assert_close(env._lost_wheel_contact_age, torch.full((3,), 0.1))
    torch.testing.assert_close(env._lost_wheel_contact_time, torch.zeros(3))

    # One lost-contact step is transient and does not terminate.
    contact_forces[..., 2] = 10.0
    contact_forces[0, 2] = 0.0
    transient = lost_wheel_contact(env, sensor_cfg=sensor_cfg, threshold=2.0)
    torch.testing.assert_close(transient, torch.tensor([False, False, False]))

    # Regained contact immediately clears the accumulated time.
    env.episode_length_buf += 1
    contact_forces[0, 2] = 10.0
    recovered = lost_wheel_contact(env, sensor_cfg=sensor_cfg, threshold=2.0)
    torch.testing.assert_close(recovered, torch.tensor([False, False, False]))
    assert env._lost_wheel_contact_time[0] == 0.0

    # Five continuous 20 ms loss steps terminate only the affected environment.
    contact_forces[1, 2] = 0.0
    for _ in range(4):
        env.episode_length_buf += 1
        not_yet = lost_wheel_contact(env, sensor_cfg=sensor_cfg, threshold=2.0)
        assert not torch.any(not_yet)
    env.episode_length_buf += 1
    persistent = lost_wheel_contact(env, sensor_cfg=sensor_cfg, threshold=2.0)
    torch.testing.assert_close(persistent, torch.tensor([False, True, False]))

    # A genuine per-environment reset restarts both timers only for that environment.
    env.episode_length_buf[1] = 0
    after_reset = lost_wheel_contact(env, sensor_cfg=sensor_cfg, threshold=2.0)
    torch.testing.assert_close(after_reset, torch.tensor([False, False, False]))
    assert env._lost_wheel_contact_time[1] == 0.0
    torch.testing.assert_close(env._lost_wheel_contact_age[1], torch.tensor(0.02))
    assert env._lost_wheel_contact_age[0] > env._lost_wheel_contact_age[1]
