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
import isaaclab.utils.math as math_utils
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


def wheel_contact_state(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return one binary contact value per explicitly resolved wheel body."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return (torch.linalg.vector_norm(forces, dim=-1) > threshold).float()


def wheel_clearance(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float,
) -> torch.Tensor:
    """Return wheel-bottom clearance above the flat ground plane."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    ground_height = env.scene.env_origins[:, 2].unsqueeze(-1)
    return wheel_height - ground_height - wheel_radius


def yaw_rate_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return only the deployable yaw-rate component of a velocity command."""
    return env.command_manager.get_command(command_name)[:, 2:3]


# ---------------------------------------------------------------------------
# Four-mode pivot observations.
# ---------------------------------------------------------------------------


def _pivot_cache(env: ManagerBasedRLEnv):
    """Fetch the environment-owned PivotStateCache (I8) or raise."""
    cache = getattr(env.unwrapped, "pivot_cache", None)
    if cache is None:
        raise RuntimeError(
            "Four-mode pivot observations require env.pivot_cache (set in PivotEnv._pre_physics_step)"
        )
    return cache


def balance_signals(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return [zeta, zeta_dot, xi] relative to selected support wheels."""

    asset: Articulation = env.scene[asset_cfg.name]

    # ---------------------------------------------------------
    # 1. Whole-body CoM
    # ---------------------------------------------------------
    mass = asset.root_physx_view.get_masses().to(asset.device)   # (N, B)

    total_mass = mass.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0e-6)

    com_pos_w = (
        asset.data.body_com_pos_w
        * mass.unsqueeze(-1)
    ).sum(dim=1) / total_mass

    com_vel_w = (
        asset.data.body_com_lin_vel_w
        * mass.unsqueeze(-1)
    ).sum(dim=1) / total_mass

    # ---------------------------------------------------------
    # 2. Support midpoint: selected HL + HR wheel bodies
    # ---------------------------------------------------------
    support_pos_w = asset.data.body_pos_w[
        :, asset_cfg.body_ids
    ].mean(dim=1)

    # ---------------------------------------------------------
    # 3. Heading
    # ---------------------------------------------------------
    quat = asset.data.root_quat_w

    forward_b = torch.zeros(
        env.num_envs,
        3,
        device=env.device,
        dtype=com_pos_w.dtype,
    )
    forward_b[:, 0] = 1.0

    heading_w = math_utils.quat_apply(
        quat,
        forward_b,
    )

    heading_xy = heading_w[:, :2]

    heading_xy = heading_xy / torch.linalg.vector_norm(
        heading_xy,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-6)

    # ---------------------------------------------------------
    # 4. Balance coordinates
    # ---------------------------------------------------------
    rel_pos_xy = (
        com_pos_w[:, :2]
        - support_pos_w[:, :2]
    )

    zeta = torch.sum(
        rel_pos_xy * heading_xy,
        dim=-1,
    )

    zeta_dot = torch.sum(
        com_vel_w[:, :2] * heading_xy,
        dim=-1,
    )

    h = (
        com_pos_w[:, 2]
        - support_pos_w[:, 2]
    ).clamp_min(0.05)

    omega0 = torch.sqrt(
        torch.tensor(
            9.81,
            device=env.device,
            dtype=h.dtype,
        ) / h
    )

    xi = zeta + zeta_dot / omega0

    return torch.stack(
        (zeta, zeta_dot, xi),
        dim=-1,
    )
    
def wheel_contact_flags(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return stateless contact flags for explicitly selected wheel bodies."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return (torch.linalg.vector_norm(forces, dim=-1) > threshold).float()


def pivot_command_obs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The 7-D pivot command: mode one-hot | omega_z* | delta_theta* | tuck*."""
    return env.command_manager.get_command(command_name)


def contact_forces_term(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
):
    sensor = env.scene.sensors[sensor_cfg.name]

    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]

    magnitude = torch.linalg.vector_norm(
        forces,
        dim=-1,
    )

    return torch.relu(
        magnitude - threshold
    )

def com_offset_term(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    
    robot: Articulation = env.scene["robot"]

    mass = asset.root_physx_view.get_masses().to(asset.device)   # (N, B)
    
    total_mass = mass.sum(dim=1, keepdim=True)

    com_w = (
        robot.data.body_com_pos_w * mass.unsqueeze(-1)
    ).sum(dim=1) / total_mass

    offset_w = com_w - robot.data.root_pos_w

    return offset_w


def action_delay_term(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged: actuation delay RNG percentile proxy (10-30 ms budget)."""
    robot = env.scene["robot"]
    delay_steps = getattr(robot, "computed_effort_delay", None)
    if delay_steps is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return delay_steps.float().mean(dim=-1, keepdim=True)
