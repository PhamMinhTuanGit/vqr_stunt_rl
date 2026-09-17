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
