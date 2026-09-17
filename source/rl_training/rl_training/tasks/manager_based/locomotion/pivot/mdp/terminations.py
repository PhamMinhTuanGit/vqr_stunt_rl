# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the VQR pivot tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pivot_base_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Terminate when the explicitly selected torso contacts the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    return torch.linalg.vector_norm(force_history, dim=-1).amax(dim=1).amax(dim=1) > threshold


def pivot_inverted(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate once the torso's local up direction points below the horizon."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b[:, 2] > 0.0


def pivot_tilt_limit(
    env: ManagerBasedRLEnv,
    nominal_roll: float,
    nominal_pitch: float,
    roll_limit: float,
    pitch_limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate only for large roll/pitch deviations from the balance pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    roll_error = math_utils.wrap_to_pi(roll - nominal_roll).abs()
    pitch_error = math_utils.wrap_to_pi(pitch - nominal_pitch).abs()
    return (roll_error > roll_limit) | (pitch_error > pitch_limit)


def pivot_base_height_below(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate after the floating base collapses near the flat ground."""
    asset: Articulation = env.scene[asset_cfg.name]
    relative_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return relative_height < minimum_height


def pivot_drifted_away(
    env: ManagerBasedRLEnv,
    maximum_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when planar displacement from the environment origin is excessive."""
    asset: Articulation = env.scene[asset_cfg.name]
    displacement = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return torch.linalg.vector_norm(displacement, dim=1) > maximum_distance


def lost_wheel_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 2.0,
    grace_period_s: float = 0.1,
    persistence_s: float = 0.1,
) -> torch.Tensor:
    """Terminate after any selected wheel continuously loses contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    in_contact = torch.linalg.vector_norm(current_forces, dim=-1) > threshold
    lost_now = ~torch.all(in_contact, dim=1)

    timer_name = "_lost_wheel_contact_time"
    age_name = "_lost_wheel_contact_age"
    step_name = "_lost_wheel_contact_last_episode_step"
    episode_step = env.episode_length_buf
    if not hasattr(env, timer_name):
        setattr(env, timer_name, torch.zeros_like(lost_now, dtype=torch.float))
        setattr(env, age_name, torch.zeros_like(lost_now, dtype=torch.float))
        setattr(env, step_name, episode_step.clone() - 1)

    lost_time: torch.Tensor = getattr(env, timer_name)
    physical_age: torch.Tensor = getattr(env, age_name)
    last_episode_step: torch.Tensor = getattr(env, step_name)
    reset_env = episode_step < last_episode_step
    new_step = episode_step != last_episode_step

    physical_age[reset_env] = 0.0
    lost_time[reset_env] = 0.0
    in_grace = physical_age < grace_period_s

    lost_time[in_grace | ~lost_now] = 0.0
    accumulate = new_step & ~in_grace & lost_now
    lost_time[accumulate] += env.step_dt
    physical_age[new_step] += env.step_dt
    last_episode_step.copy_(episode_step)

    return (lost_time + 1.0e-6 >= persistence_s) & ~in_grace


def m3_drifted_from_reset(
    env: ManagerBasedRLEnv,
    maximum_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate M3 after excessive XY displacement from its randomized reset pose."""
    if not hasattr(env, "_pivot_reset_root_xy"):
        raise RuntimeError("M3 reset XY buffer is unavailable; reset_four_wheel_standing must run first.")
    asset: Articulation = env.scene[asset_cfg.name]
    displacement = asset.data.root_pos_w[:, :2] - env._pivot_reset_root_xy
    return torch.linalg.vector_norm(displacement, dim=1) > maximum_distance


# ---------------------------------------------------------------------------
# Four-mode pivot terminations (spec section 9): fallen / tilt / drift / grace.
# ---------------------------------------------------------------------------


def _pivot_cache(env: ManagerBasedRLEnv):
    cache = getattr(env.unwrapped, "pivot_cache", None)
    if cache is None:
        raise RuntimeError("Four-mode terminations require env.pivot_cache (I8).")
    return cache


def pivot_fallen(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True once the torso tips fully past the recoverable envelope.

    Fallen = pitch below the balance ceiling (nose-dived) or projected gravity
    indicates the torso is inverted/nearly inverted (|g_z| up with |g_xy| large).
    """
    robot = env.scene["robot"]
    g = robot.data.projected_gravity_b
    inverted = g[:, 2] > 0.0  # gravity pointing along +z body => upside down
    _, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
    nose_dive = pitch < -0.35  # ~-20 deg below the recoverable corridor
    return inverted | nose_dive


def pivot_tilt_exceeded(
    env: ManagerBasedRLEnv,
    roll_limit: float,
    pitch_limit_margin: float = 0.0,
    command_name: str = "pivot_mode",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """|roll| beyond limit or pitch beyond theta*+limit (grace applied by cfg)."""
    from .theta_star import theta_star

    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    omega_z_command = env.command_manager.get_term(command_name).omega_z_command
    roll_bad = math_utils.wrap_to_pi(roll).abs() > roll_limit
    pitch_bad = pitch > theta_star(omega_z_command) + pitch_limit_margin
    return roll_bad | pitch_bad


def pivot_drift_exceeded(
    env: ManagerBasedRLEnv,
    maximum_drift: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Planar drift of the configured support midpoint beyond the budget."""
    from ..config.wheeled.vqr.physical_params import VQR_PHYSICS

    robot: Articulation = env.scene[asset_cfg.name]
    mid = robot.data.body_pos_w[:, asset_cfg.body_ids, :2].mean(dim=1)
    target = env.scene.env_origins[:, :2].clone()
    target[:, 0] -= 0.5 * VQR_PHYSICS.wheelbase
    drift = torch.linalg.vector_norm(mid - target, dim=-1)
    return drift > maximum_drift


def pivot_grace_window(
    env: ManagerBasedRLEnv, grace_s: float, condition: str
) -> torch.Tensor:
    """Time-limited grace: skip termination during the first ``grace_s``."""
    steps = int(grace_s / env.step_dt)
    cond = {
        "fallen": pivot_fallen(env),
        "tilt": pivot_tilt_exceeded(env, roll_limit=0.6),
    }[condition]
    return cond & (env.episode_length_buf > steps)


def pivot_recover_righted(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Recovery success: all four wheels in contact for ~0.5 s."""
    cache = _pivot_cache(env)
    all_down = cache.wheel_contact > 0.5
    held = getattr(env.unwrapped, "_pivot_righted_steps", None)
    if held is None:
        held = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    held = torch.where(
        all_down.all(dim=-1), held + 1, torch.zeros_like(held)
    )
    env.unwrapped._pivot_righted_steps = held
    return held >= int(0.5 / env.step_dt)
