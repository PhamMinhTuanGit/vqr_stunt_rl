# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def wheel_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return one binary contact flag per selected wheel body."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

    return (torch.linalg.vector_norm(force, dim=-1) > threshold).float()


def wheel_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float,
) -> torch.Tensor:
    """Return wheel-bottom clearance relative to each environment origin."""

    robot: Articulation = env.scene[asset_cfg.name]
    wheel_height = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    ground_height = env.scene.env_origins[:, 2].unsqueeze(-1)

    return wheel_height - ground_height - wheel_radius


def wheel_normal_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_scale: float = 100.0,
) -> torch.Tensor:
    """Return normalized upward contact force for each selected wheel."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    fz = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]

    return torch.clamp(fz, min=0.0) / force_scale


def support_wheel_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    wheel_axle_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    body_lateral_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Alignment of support-wheel axle with robot lateral axis.

    Output:
        (N, 2)

    1.0 -> perfectly aligned
    0.0 -> perpendicular
    """

    robot: Articulation = env.scene[asset_cfg.name]

    wheel_quat = robot.data.body_quat_w[:, asset_cfg.body_ids, :]

    # Root/body orientation
    body_quat = robot.data.root_quat_w

    # ---------------------------------------------------
    # Wheel axle -> world
    # ---------------------------------------------------
    axle_local = torch.tensor(
        wheel_axle_axis,
        device=env.device,
        dtype=wheel_quat.dtype,
    ).view(1, 1, 3).expand(wheel_quat.shape[0], wheel_quat.shape[1], 3)

    axle_w = quat_apply(
        wheel_quat.reshape(-1, 4),
        axle_local.reshape(-1, 3),
    ).reshape(wheel_quat.shape[0], wheel_quat.shape[1], 3)

    # ---------------------------------------------------
    # Body lateral axis -> world
    # ---------------------------------------------------
    lateral_local = torch.tensor(
        body_lateral_axis,
        device=env.device,
        dtype=body_quat.dtype,
    ).view(1, 3).expand(body_quat.shape[0], 3)

    body_lateral_w = quat_apply(
        body_quat,
        lateral_local,
    )

    # Normalize
    axle_w = axle_w / (torch.linalg.vector_norm(axle_w, dim=-1, keepdim=True) + 1e-6)
    body_lateral_w = body_lateral_w / (
        torch.linalg.vector_norm(body_lateral_w, dim=-1, keepdim=True) + 1e-6
    )

    # (N, 2)
    alignment = torch.abs(torch.sum(axle_w * body_lateral_w.unsqueeze(1), dim=-1))

    return alignment
