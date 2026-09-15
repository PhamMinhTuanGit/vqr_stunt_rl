# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the two-wheel pivot tasks."""

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
