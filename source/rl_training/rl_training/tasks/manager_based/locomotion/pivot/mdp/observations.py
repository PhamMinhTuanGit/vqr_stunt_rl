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
# Four-mode pivot observations: all readers of the per-step cache (I8).
# ---------------------------------------------------------------------------


def _pivot_cache(env: ManagerBasedRLEnv):
    """Fetch the environment-owned PivotStateCache (I8) or raise."""
    cache = getattr(env.unwrapped, "pivot_cache", None)
    if cache is None:
        raise RuntimeError(
            "Four-mode pivot observations require env.pivot_cache (set in PivotEnv._pre_physics_step)"
        )
    return cache


def balance_signals(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(zeta, zeta_dot, xi) straight from the cache (I1/I8)."""
    cache = _pivot_cache(env)
    return torch.stack([cache.zeta, cache.zeta_dot, cache.xi], dim=-1)


def ema_wheel_torque(env: ManagerBasedRLEnv) -> torch.Tensor:
    """EMA|tau_w| from the cache, scaled by the continuous-torque limit."""
    cache = _pivot_cache(env)
    return cache.ema_wheel_torque / cache.physics.wheel_continuous_torque


def wheel_contact_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Binary contact flag per wheel, FL | FR | HL | HR (cache-provided)."""
    cache = _pivot_cache(env)
    return cache.wheel_contact


def pivot_history_stack(
    env: ManagerBasedRLEnv,
    command_name: str,
    history_length: int = 5,
) -> torch.Tensor:
    """Concatenate the last H frames of the scalar balance core.

    Maintains an env-side ring buffer of the last H-1 (zeta, zeta_dot, xi)
    frames; the current frame appends at read time.
    """
    cache = _pivot_cache(env)
    history = env.unwrapped.pivot_history
    current = torch.stack([cache.zeta, cache.zeta_dot, cache.xi], dim=-1)
    return history.concat(current)


def pivot_command_obs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The 8-dim pivot command: mode one-hot | omega_z* | delta_theta* | tuck*."""
    return env.command_manager.get_command(command_name)


def contact_forces_term(
    env: ManagerBasedRLEnv, threshold: float = 1.0
) -> torch.Tensor:
    """Privileged: per-wheel normal force magnitude above threshold."""
    sensor = env.scene.sensors["contact_forces"]
    magnitude = sensor.data.net_forces_w.norm(dim=-1)
    return torch.relu(magnitude - threshold)


def mu_hat_term(env: ManagerBasedRLEnv, default: float = 1.0) -> torch.Tensor:
    """Privileged: per-env friction estimate used by thexi safety radii."""
    cache = _pivot_cache(env)
    return cache.mu_hat.unsqueeze(-1)


def com_offset_term(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged: upstream CoM offset body frame proxy (privileged)."""
    robot = env.scene["robot"]
    com = robot.data.body_com_pos_w[:, robot.root_idx]  # (N, 3)
    root = robot.data.root_pos_w
    return com - root


def payload_mass_term(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged: body payload mass sum normalized by the SSOT mass."""
    robot = env.scene["robot"]
    return (robot.data.body_mass.sum(dim=-1) / _pivot_cache(env).physics.mass).unsqueeze(-1)


def action_delay_term(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged: actuation delay RNG percentile proxy (10-30 ms budget)."""
    robot = env.scene["robot"]
    delay_steps = getattr(robot, "computed_effort_delay", None)
    if delay_steps is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return delay_steps.float().mean(dim=-1, keepdim=True)

