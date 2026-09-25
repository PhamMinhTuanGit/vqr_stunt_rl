# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, yaw_quat

from .fsm_gates import fsm_gates
from .fsm import select_swing_wheel_contact

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Global curriculum scalar in [0, 1], updated from terrain-level mean.
gait_level: float = 0.0

#Reward mới cho task,làm lại
#Phần phạt task chính:
def custom_yaw_vel_tracking_exp(env: ManagerBasedRLEnv, std:float, command_name: str, 
                            asset_cfg: SceneEntityCfg("robot")) -> torch.Tensor:
    robot: RigidObject = env.scene[asset_cfg.name]
    # Để lấy vận tốc góc yaw thực tế trong Isaaclab
    # 
    actual_yaw_vel = robot.data.root_ang_vel_b[:,2]
    yaw_cmd = env.command_manager.get_command("command_name")
    return torch.exp(-torch.sum(torch.square(actual_yaw_vel - yaw_cmd))/0.25)

#Phần phạt lệnh điều khiển:
def motor_effort_penalty_l2(env: ManagerBasedRLEnv, std:float, command_name: str,
                            asset_cfg: SceneEntityCfg("robot")) -> torch.Tensor:
    action_effort = env.command_manager.get_command("command_name")

    raise NotImplementedError("motor_effort_penalty_l2 is not implemented")

#Phần phạt dáng đứng:












def update_gait_level_from_terrain_mean(terrain_level_mean: float | torch.Tensor) -> float:
    """Update global gait_level from mean terrain level.

    Mapping rule:
    - mean <= 0.0 -> 0.0
    - 0.0 < mean < 3.0 -> 使用 exp 函数映射
    - mean == 3.0 -> 1.0
    - mean >= 3.0 -> 1.0
    """
    global gait_level

    mean_tensor = torch.as_tensor(terrain_level_mean, dtype=torch.float32)
    if mean_tensor.numel() == 0:
        mean_val = 0.0
    else:
        mean_val = float(torch.mean(mean_tensor).item())

    if math.isnan(mean_val) or math.isinf(mean_val):
        mean_val = 0.0

    if mean_val <= 0.0:
        gait_level = 0.0
    elif mean_val < 3.0:
        # exp 映射：mean=0 时接近 0，mean=3 时恰好为 1
        gait_level = math.exp(mean_val - 3.0)
    else:  # mean_val >= 3.0
        gait_level = 1.0

    return gait_level

def get_gait_level_tensor(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return global gait_level as tensor matching environment batch size."""
    return torch.full((env.num_envs,), gait_level, device=env.device)


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques (curriculum-scaled by gait_level)."""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    return reward * get_gait_level_tensor(env)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action rate (curriculum-scaled by gait_level)."""
    reward = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return reward * get_gait_level_tensor(env)

def action_smooth_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize second-order action changes using an env-local action history."""
    cache_name = "_action_smooth_prev_prev_action"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, torch.zeros_like(env.action_manager.action))

    prev_prev_action = getattr(env, cache_name)
    diff = torch.square(env.action_manager.action + prev_prev_action - 2 * env.action_manager.prev_action)
    # Ignore the first two steps of each episode where action history is incomplete.
    diff = diff * (env.action_manager.prev_action != 0)
    diff = diff * (prev_prev_action != 0)
    reward = torch.sum(diff, dim=1)

    setattr(env, cache_name, env.action_manager.prev_action.clone())
    return reward * get_gait_level_tensor(env)

def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact force violations (curriculum-scaled by gait_level)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    reward = torch.sum(violation.clip(min=0.0), dim=1)
    return reward * get_gait_level_tensor(env)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_yaw_rate_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of a scalar body-frame yaw-rate command."""
    asset: RigidObject = env.scene[asset_cfg.name]
    yaw_rate_command = env.command_manager.get_command(command_name)[:, 0]
    yaw_rate_error = torch.square(yaw_rate_command - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-yaw_rate_error / std**2)


def _yaw_wheel_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Return contact flags for the explicitly selected wheel bodies."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.linalg.vector_norm(forces, dim=-1) > threshold


def _yaw_support_load_quality(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    target_force_n: float,
) -> torch.Tensor:
    """Score both wheels' upward normal load on the flat ground in [0, 1]."""
    if target_force_n <= 0.0:
        raise ValueError("target_force_n must be positive.")
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    if forces.ndim != 3 or forces.shape[1:] != (2, 3):
        raise ValueError("Support load requires exactly two wheel force vectors per environment.")
    per_wheel = torch.clamp(forces[..., 2] / target_force_n, min=0.0, max=1.0)
    # The weaker wheel dominates; one loaded wheel still provides a small
    # exploration signal toward recovering the other contact.
    return 0.8 * per_wheel.amin(dim=1) + 0.2 * per_wheel.mean(dim=1)


def yaw_transition_support_load(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    sensor_cfg_mirror: SceneEntityCfg,
    fsm_command_name: str,
    target_force_n: float,
) -> torch.Tensor:
    """Reward load on the selected support diagonal during TRANSITION only."""
    gates = fsm_gates(env, fsm_command_name)
    pos = _yaw_support_load_quality(env, sensor_cfg, target_force_n)
    neg = _yaw_support_load_quality(env, sensor_cfg_mirror, target_force_n)
    reward = torch.where(gates["diag_pos"], pos, neg) * gates["f_trans"]
    _fsm_record_positive_budget(env, gates, "transition_support_load", reward)
    return reward


def _yaw_support_shape(contacts: torch.Tensor) -> torch.Tensor:
    """Give nearly all support credit only when both selected wheels contact."""
    c1, c2 = contacts.to(dtype=torch.float32).unbind(dim=1)
    partial = 0.5 * (c1 + c2)
    both = c1 * c2
    return 0.1 * partial + 0.9 * both


def _yaw_whole_body_com_xy(asset: Articulation) -> torch.Tensor:
    """Return the mass-weighted whole-body CoM projected onto the ground plane."""
    masses = asset.root_physx_view.get_masses().to(asset.device)
    total_mass = masses.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    return (asset.data.body_com_pos_w[..., :2] * masses.unsqueeze(-1)).sum(dim=1) / total_mass


def _yaw_support_geometry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CoM distance to the support line, projection coordinate, and segment length."""
    asset: Articulation = env.scene[asset_cfg.name]
    support_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    if support_xy.shape[1] != 2:
        raise ValueError("Yaw support rewards require exactly two ordered support wheel bodies.")

    start = support_xy[:, 0]
    segment = support_xy[:, 1] - start
    length_sq = torch.sum(segment.square(), dim=-1).clamp_min(1.0e-8)
    length = torch.sqrt(length_sq)
    com_offset = _yaw_whole_body_com_xy(asset) - start
    projection = torch.sum(com_offset * segment, dim=-1) / length_sq
    perpendicular_distance = torch.abs(
        com_offset[:, 0] * segment[:, 1] - com_offset[:, 1] * segment[:, 0]
    ) / length
    return perpendicular_distance, projection, length


def yaw_com_support(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
    asset_cfg_mirror: SceneEntityCfg | None = None,
    fsm_command_name: str | None = None,
) -> torch.Tensor:
    """Reward CoM proximity with a non-flat reciprocal-L1 kernel."""
    if std <= 0.0:
        raise ValueError("std must be positive.")
    distance, _, _ = _yaw_support_geometry(env, asset_cfg)
    score = 1.0 / (1.0 + distance / std)
    if fsm_command_name is None:
        return score
    if asset_cfg_mirror is None:
        raise ValueError("asset_cfg_mirror is required when fsm_command_name is set.")
    mirror_distance, _, _ = _yaw_support_geometry(env, asset_cfg_mirror)
    mirror_score = 1.0 / (1.0 + mirror_distance / std)
    gates = fsm_gates(env, fsm_command_name)
    reward = torch.where(gates["diag_pos"], score, mirror_score) * gates["f_geom"]
    _fsm_record_positive_budget(env, gates, "com_support", reward)
    return reward


def yaw_base_height_tracking(
    env: ManagerBasedRLEnv,
    target_height: float,
    error_scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fsm_command_name: str | None = None,
) -> torch.Tensor:
    """Track TORSO height with non-saturating, normalized Huber shaping.

    The raw score is ``1 - huber(abs(height - target) / error_scale)``.
    It is maximal at the finite target, quadratic nearby, and linear rather
    than exponentially flat when the torso is far from the target.
    """
    if error_scale <= 0.0:
        raise ValueError("error_scale must be positive.")

    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    if not hasattr(env, "_yaw_base_height_min"):
        env._yaw_base_height_min = torch.full_like(base_height, torch.inf)
    env._yaw_base_height_min = torch.minimum(env._yaw_base_height_min, base_height)

    normalized_error = torch.abs(base_height - target_height) / error_scale
    huber = torch.where(
        normalized_error <= 1.0,
        0.5 * normalized_error.square(),
        normalized_error - 0.5,
    )
    score = 1.0 - huber
    if fsm_command_name is None:
        return score
    gates = fsm_gates(env, fsm_command_name)
    reward = score * gates["f_yaw"]
    _fsm_record_positive_budget(env, gates, "base_height", reward)
    return reward


def yaw_low_base_height_l1(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    error_scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return normalized L1 violation below the minimum safe base height."""
    if error_scale <= 0.0:
        raise ValueError("error_scale must be positive.")
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.relu(minimum_height - base_height) / error_scale


def yaw_downward_low_base_velocity_l2(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    height_margin: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize downward base velocity only inside the low-height region."""
    if height_margin <= 0.0:
        raise ValueError("height_margin must be positive.")
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    low_gate = torch.clamp(
        torch.relu(minimum_height - base_height) / height_margin,
        max=1.0,
    )
    downward_speed = torch.relu(-asset.data.root_lin_vel_w[:, 2])
    return low_gate * downward_speed.square()


def yaw_support_span_band_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    minimum_span: float,
    maximum_span: float,
    std: float,
    asset_cfg_mirror: SceneEntityCfg | None = None,
    fsm_command_name: str | None = None,
) -> torch.Tensor:
    """Penalize FL-HR span only outside the configured non-zero-width band."""
    if minimum_span < 0.0 or maximum_span <= minimum_span:
        raise ValueError("Support-span bounds must satisfy 0 <= minimum_span < maximum_span.")
    if std <= 0.0:
        raise ValueError("std must be positive.")

    _, _, span = _yaw_support_geometry(env, asset_cfg)
    outside_distance = torch.relu(minimum_span - span) + torch.relu(span - maximum_span)
    score = (outside_distance / std).square()
    if fsm_command_name is None:
        return score
    if asset_cfg_mirror is None:
        raise ValueError("asset_cfg_mirror is required when fsm_command_name is set.")
    _, _, mirror_span = _yaw_support_geometry(env, asset_cfg_mirror)
    mirror_outside = torch.relu(minimum_span - mirror_span) + torch.relu(mirror_span - maximum_span)
    mirror_score = (mirror_outside / std).square()
    gates = fsm_gates(env, fsm_command_name)
    return torch.where(gates["diag_pos"], score, mirror_score) * gates["f_geom"]


def yaw_planar_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base x/y velocity for rotate-in-place behavior."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(asset.data.root_lin_vel_b[:, :2].square(), dim=1)


def yaw_support_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward only when both selected support wheels contact the ground."""
    return _yaw_wheel_contacts(env, sensor_cfg, threshold).float().prod(dim=1)


def _yaw_lift_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float,
    target_clearance: float,
) -> torch.Tensor:
    """Return independent normalized clearance progress for the selected wheels."""
    if target_clearance <= 0.0:
        raise ValueError("target_clearance must be positive.")

    asset: Articulation = env.scene[asset_cfg.name]
    wheel_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    ground_height = env.scene.env_origins[:, 2].unsqueeze(-1)
    clearance = wheel_height - ground_height - wheel_radius
    return torch.clamp(clearance / target_clearance, min=0.0, max=1.0)


def yaw_lift_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float,
    target_clearance: float,
    asset_cfg_mirror: SceneEntityCfg | None = None,
    fsm_command_name: str | None = None,
    support_sensor_cfg: SceneEntityCfg | None = None,
    support_sensor_cfg_mirror: SceneEntityCfg | None = None,
    support_force_target_n: float = 80.0,
    transition_ungated_fraction: float = 0.35,
) -> torch.Tensor:
    """Shape lifted-wheel clearance with partial credit and a four-wheel penalty.

    Each selected wheel contributes independently. The returned score is in
    ``[-1, 1]``: both wheels on the ground score ``-1``, lifting either wheel
    improves the score, and both wheels must reach the target to score ``1``.
    In FSM TRANSITION, positive credit is partly gated by support load;
    the raw clearance metric and YAW reward are unchanged.
    """
    progress = _yaw_lift_progress(env, asset_cfg, wheel_radius, target_clearance)

    score = 2.0 * progress.mean(dim=1) - 1.0
    if fsm_command_name is None:
        # Keep a separate curriculum metric: the weaker wheel's progress,
        # averaged over the episode. This must not be reconstructed from the
        # signed mean reward because one fully lifted wheel can hide the other.
        min_progress = progress.amin(dim=1)
        if not hasattr(env, "_yaw_lift_min_progress_sum"):
            env._yaw_lift_min_progress_sum = torch.zeros_like(min_progress)
            env._yaw_lift_min_progress_samples = torch.zeros_like(min_progress, dtype=torch.long)
        env._yaw_lift_min_progress_sum += min_progress
        env._yaw_lift_min_progress_samples += 1
        return score
    if asset_cfg_mirror is None:
        raise ValueError("asset_cfg_mirror is required when fsm_command_name is set.")
    if support_sensor_cfg is None or support_sensor_cfg_mirror is None:
        raise ValueError("Both support sensor configs are required when fsm_command_name is set.")
    if not 0.0 < transition_ungated_fraction < 1.0:
        raise ValueError("transition_ungated_fraction must be in (0, 1).")
    mirror_progress = _yaw_lift_progress(
        env,
        asset_cfg_mirror,
        wheel_radius,
        target_clearance,
    )
    mirror_score = 2.0 * mirror_progress.mean(dim=1) - 1.0
    gates = fsm_gates(env, fsm_command_name)
    selected_progress = torch.where(gates["diag_pos"].unsqueeze(1), progress, mirror_progress)
    min_progress = selected_progress.amin(dim=1)
    if not hasattr(env, "_yaw_lift_min_progress_sum"):
        env._yaw_lift_min_progress_sum = torch.zeros_like(min_progress)
        env._yaw_lift_min_progress_samples = torch.zeros_like(min_progress, dtype=torch.long)
    env._yaw_lift_min_progress_sum += min_progress
    env._yaw_lift_min_progress_samples += 1
    selected_score = torch.where(gates["diag_pos"], score, mirror_score)
    support_pos = _yaw_support_load_quality(env, support_sensor_cfg, support_force_target_n)
    support_neg = _yaw_support_load_quality(env, support_sensor_cfg_mirror, support_force_target_n)
    support_quality = torch.where(gates["diag_pos"], support_pos, support_neg)
    lift_gate = transition_ungated_fraction + (1.0 - transition_ungated_fraction) * support_quality
    # Keep the no-lift penalty intact.  Only positive lift credit is reduced
    # when support is weak, so dropping a wheel cannot erase that penalty.
    transition_score = torch.clamp(selected_score, max=0.0) + torch.relu(selected_score) * lift_gate
    reward = transition_score * gates["f_trans"] + selected_score * gates["f_yaw"]
    _fsm_record_positive_budget(env, gates, "lift_clearance", reward)
    return reward


def yaw_com_inside_support_segment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
    asset_cfg_mirror: SceneEntityCfg | None = None,
    fsm_command_name: str | None = None,
) -> torch.Tensor:
    """Reward CoM projection inside the finite FL-HR support segment.

    The score is one anywhere inside the segment and decays smoothly with the
    physical distance past either endpoint.
    """
    _, projection, segment_length = _yaw_support_geometry(env, asset_cfg)
    outside_distance = (
        torch.relu(-projection) + torch.relu(projection - 1.0)
    ) * segment_length
    score = torch.exp(-outside_distance.square() / std**2)
    if fsm_command_name is None:
        return score
    if asset_cfg_mirror is None:
        raise ValueError("asset_cfg_mirror is required when fsm_command_name is set.")
    _, mirror_projection, mirror_segment_length = _yaw_support_geometry(env, asset_cfg_mirror)
    mirror_outside_distance = (
        torch.relu(-mirror_projection) + torch.relu(mirror_projection - 1.0)
    ) * mirror_segment_length
    mirror_score = torch.exp(-mirror_outside_distance.square() / std**2)
    gates = fsm_gates(env, fsm_command_name)
    reward = torch.where(gates["diag_pos"], score, mirror_score) * gates["f_geom"]
    _fsm_record_positive_budget(env, gates, "com_inside_segment", reward)
    return reward


def yaw_balance(
    env: ManagerBasedRLEnv,
    std: float,
    nominal_roll: float = 0.0,
    nominal_pitch: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fsm_command_name: str | None = None,
) -> torch.Tensor:
    """Reward roll and pitch near the configured diagonal-support equilibrium."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    roll_error = math_utils.wrap_to_pi(roll - nominal_roll)
    pitch_error = math_utils.wrap_to_pi(pitch - nominal_pitch)
    score = torch.exp(-(roll_error.square() + pitch_error.square()) / std**2)
    # ``None`` preserves the baseline task's exact calculation.  The FSM
    # task suppresses this otherwise-positive term in SAFE_RECOVERY.
    if fsm_command_name is None:
        return score
    gates = fsm_gates(env, fsm_command_name)
    reward = score * (1.0 - gates["f_safe"])
    _fsm_record_positive_budget(env, gates, "balance", reward)
    return reward


# def yaw_gated_tracking(
#     env: ManagerBasedRLEnv,
#     command_name: str,
#     support_sensor_cfg: SceneEntityCfg,
#     lifted_asset_cfg: SceneEntityCfg,
#     wheel_radius: float,
#     target_clearance: float,
#     std: float,
#     contact_threshold: float = 1.0,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Track yaw only after both support wheels contact and both lifted wheels clear the plane."""
#     support_gate = yaw_support_contact(env, support_sensor_cfg, contact_threshold)
#     clearance_gate = yaw_lift_clearance(
#         env, lifted_asset_cfg, wheel_radius, target_clearance
#     )
#     return support_gate * clearance_gate * track_yaw_rate_exp(env, std, command_name, asset_cfg)
def yaw_gated_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    support_sensor_cfg: SceneEntityCfg,
    lifted_asset_cfg: SceneEntityCfg,
    wheel_radius: float,
    target_clearance: float,
    std: float,
    contact_threshold: float = 1.0,
    clearance_gate_floor: float = 0.25,
    edge_command_fraction: float = 0.80,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track yaw while transitioning toward the two-wheel pose.

    - Both support wheels must remain in contact.
    - Yaw tracking is already rewarded before the lifted wheels reach
      their target clearance.
    - The yaw reward smoothly increases as the lifted wheels rise.

    clearance_gate_floor:
        Fraction of yaw reward available when lifted-wheel clearance is zero.
        0.25 means 25% yaw reward is available from the beginning.
    edge_command_fraction:
        Commands above this fraction of the active yaw limit contribute to the
        edge-tracking curriculum metric.
    """
    if not 0.0 < edge_command_fraction <= 1.0:
        raise ValueError("edge_command_fraction must be in (0, 1].")

    # Hard safety/task-topology gate:
    # 1 only when both FL and HR support wheels are in contact.
    support_gate = yaw_support_contact(
        env,
        support_sensor_cfg,
        contact_threshold,
    )
    # Contact is a task condition, not a standalone positive reward. Accumulate
    # it here for curriculum evaluation without inflating the value target.
    if not hasattr(env, "_yaw_support_score_sum"):
        env._yaw_support_score_sum = torch.zeros_like(support_gate)
        env._yaw_support_score_samples = torch.zeros_like(support_gate, dtype=torch.long)
    env._yaw_support_score_sum += support_gate
    env._yaw_support_score_samples += 1
    if not hasattr(env, "_yaw_gate_open_sum"):
        env._yaw_gate_open_sum = torch.zeros_like(support_gate)
        env._yaw_gate_open_samples = torch.zeros_like(support_gate, dtype=torch.long)
    env._yaw_gate_open_sum += support_gate
    env._yaw_gate_open_samples += 1
    env._yaw_gate_open_current = support_gate

    # Smooth [0, 1] partial-credit progress toward lifting FR and HL. Keep this
    # separate from yaw_lift_clearance, whose signed score penalizes four-wheel
    # stance and therefore is not suitable as a multiplicative gate.
    clearance_gate = _yaw_lift_progress(
        env,
        lifted_asset_cfg,
        wheel_radius,
        target_clearance,
    ).mean(dim=1)

    # Accumulate scale-invariant tracking inputs independently of the shaped
    # exponential score. These are also exposed to play diagnostics.
    asset: RigidObject = env.scene[asset_cfg.name]
    yaw_command = env.command_manager.get_command(command_name)[:, 0]
    yaw_abs_error = torch.abs(yaw_command - asset.data.root_ang_vel_b[:, 2])
    if not hasattr(env, "_yaw_command_abs_sum"):
        env._yaw_command_abs_sum = torch.zeros_like(yaw_command)
        env._yaw_rate_abs_error_sum = torch.zeros_like(yaw_command)
        env._yaw_tracking_metric_samples = torch.zeros_like(yaw_command, dtype=torch.long)
    env._yaw_command_abs_sum += torch.abs(yaw_command)
    env._yaw_rate_abs_error_sum += yaw_abs_error
    env._yaw_tracking_metric_samples += 1
    env._yaw_command_abs_current = torch.abs(yaw_command)
    env._yaw_rate_abs_error_current = yaw_abs_error

    # Aggregate a separate metric near the active command boundary. A global
    # mean can otherwise hide poor behavior close to the target +/-yaw limit.
    command_term = env.command_manager.get_term(command_name)
    yaw_limit = max(abs(float(value)) for value in command_term.cfg.yaw_rate_range)
    edge_mask = torch.abs(yaw_command) >= edge_command_fraction * yaw_limit
    if not hasattr(env, "_yaw_edge_command_abs_sum"):
        env._yaw_edge_command_abs_sum = torch.zeros_like(yaw_command)
        env._yaw_edge_rate_abs_error_sum = torch.zeros_like(yaw_command)
        env._yaw_edge_tracking_samples = torch.zeros_like(yaw_command, dtype=torch.long)
    env._yaw_edge_command_abs_sum += torch.where(edge_mask, torch.abs(yaw_command), 0.0)
    env._yaw_edge_rate_abs_error_sum += torch.where(edge_mask, yaw_abs_error, 0.0)
    env._yaw_edge_tracking_samples += edge_mask.long()

    # Yaw tracking score in [0, 1].
    yaw_tracking = track_yaw_rate_exp(
        env,
        std,
        command_name,
        asset_cfg,
    )

    # Do not require lift completion before yaw becomes useful.
    #
    # clearance = 0  -> 0.25
    # clearance = 0.5 -> 0.625
    # clearance = 1  -> 1.0
    clearance_weight = (
        clearance_gate_floor
        + (1.0 - clearance_gate_floor) * clearance_gate
    )

    return support_gate * clearance_weight * yaw_tracking


def _fsm_masked_accumulate(
    env: ManagerBasedRLEnv,
    value: torch.Tensor,
    mask: torch.Tensor,
    sum_name: str,
    samples_name: str,
) -> None:
    """Accumulate a metric only for the selected FSM state mask."""
    if not hasattr(env, sum_name):
        setattr(env, sum_name, torch.zeros_like(value))
        setattr(env, samples_name, torch.zeros_like(value, dtype=torch.long))
    getattr(env, sum_name).add_(value * mask)
    getattr(env, samples_name).add_(mask.to(dtype=torch.long))


def _fsm_masked_accumulate_pair(
    env: ManagerBasedRLEnv,
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
    first_sum_name: str,
    second_sum_name: str,
    samples_name: str,
) -> None:
    """Accumulate two metrics with one shared, masked sample count."""
    if not hasattr(env, first_sum_name):
        setattr(env, first_sum_name, torch.zeros_like(first))
    if not hasattr(env, second_sum_name):
        setattr(env, second_sum_name, torch.zeros_like(second))
    if not hasattr(env, samples_name):
        setattr(env, samples_name, torch.zeros_like(mask, dtype=torch.long))
    getattr(env, first_sum_name).add_(first * mask)
    getattr(env, second_sum_name).add_(second * mask)
    getattr(env, samples_name).add_(mask.to(dtype=torch.long))


def _fsm_episode_or(env: ManagerBasedRLEnv, name: str, mask: torch.Tensor) -> None:
    """Remember whether an FSM event occurred in the current episode."""
    if not hasattr(env, name):
        setattr(env, name, torch.zeros_like(mask, dtype=torch.bool))
    getattr(env, name).logical_or_(mask.to(dtype=torch.bool))


def _fsm_step_telemetry(env: ManagerBasedRLEnv, gates: dict[str, torch.Tensor]) -> None:
    """Accumulate per-episode FSM occupancy and switching telemetry.

    This runs from the single FSM tracking term, which is evaluated once per
    reward step.  It therefore records time rather than duplicating a reward
    term or adding a manager callback just for diagnostics.
    """
    state = gates["fsm_state"]
    if not hasattr(env, "_yaw_fsm_state_steps"):
        env._yaw_fsm_state_steps = torch.zeros(
            env.num_envs, 7, dtype=torch.long, device=state.device
        )
        env._yaw_fsm_switches = torch.zeros(env.num_envs, dtype=torch.long, device=state.device)
        env._yaw_fsm_episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=state.device)
    env._yaw_fsm_state_steps.scatter_add_(
        1, state.unsqueeze(1), torch.ones_like(state, dtype=torch.long).unsqueeze(1)
    )
    env._yaw_fsm_switches += gates["just_switched"].to(dtype=torch.long)
    env._yaw_fsm_episode_steps += 1
    in_transition = gates["b_trans"]
    if not hasattr(env, "_yaw_fsm_transition_duration_steps"):
        env._yaw_fsm_transition_duration_steps = torch.zeros_like(state, dtype=torch.long)
        env._yaw_fsm_transition_attempts = torch.zeros_like(state, dtype=torch.long)
        env._yaw_fsm_transition_active = torch.zeros_like(in_transition)
    entered = in_transition & (~env._yaw_fsm_transition_active | gates["just_switched"])
    env._yaw_fsm_transition_attempts += entered.to(dtype=torch.long)
    env._yaw_fsm_transition_duration_steps += in_transition.to(dtype=torch.long)
    env._yaw_fsm_transition_active.copy_(in_transition)


_TRANSITION_TELEMETRY_FIELDS = (
    "support_ready", "lift_wheel_1_progress", "lift_wheel_2_progress",
    "clearance_ready", "attitude_ready", "pose_ready", "torso_contact",
)


def _fsm_transition_telemetry(
    env: ManagerBasedRLEnv,
    gates: dict[str, torch.Tensor],
    support_contacts: torch.Tensor,
    lift_progress: torch.Tensor,
    robot: RigidObject,
    command,
) -> None:
    """Sample the active diagonal's transition predicates once per reward step."""
    # CPU reward doubles can omit predicate wiring; production command configs
    # provide all six sensor/body selections together.
    torso_cfg = getattr(command.cfg, "torso_sensor_cfg", None)
    if torso_cfg is None:
        return
    contact_threshold = command.cfg.contact_threshold
    torso_contact = _yaw_wheel_contacts(env, torso_cfg, contact_threshold)[:, 0]
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    attitude_ready = (roll.abs() < command.cfg.pose_angle_limit) & (
        pitch.abs() < command.cfg.pose_angle_limit
    )
    support_ready = support_contacts.all(dim=1)
    clearance_ready = (lift_progress >= command.cfg.clearance_fraction).all(dim=1)
    values = (
        support_ready, lift_progress[:, 0], lift_progress[:, 1],
        clearance_ready, attitude_ready,
        support_ready & clearance_ready & attitude_ready, torso_contact,
    )
    mask = gates["b_trans"]
    for field, value in zip(_TRANSITION_TELEMETRY_FIELDS, values):
        name = f"_yaw_fsm_transition_{field}_sum"
        if not hasattr(env, name):
            setattr(env, name, torch.zeros(env.num_envs, device=mask.device))
        getattr(env, name).add_(value.to(dtype=torch.float32) * mask)


def _fsm_record_positive_budget(
    env: ManagerBasedRLEnv,
    gates: dict[str, torch.Tensor],
    term_name: str,
    value: torch.Tensor,
) -> None:
    """Accumulate actual weighted positive reward by FSM state/direction.

    Reward functions return unweighted values, so the manager's configured
    term weight is applied here before taking the positive part.  This is
    telemetry only; a missing lightweight test-double reward manager simply
    leaves the diagnostic unset and never changes the reward value.
    """
    try:
        weight = float(env.reward_manager.get_term_cfg(term_name).weight)
    except (AttributeError, KeyError, TypeError):
        return
    positive = torch.relu(value * weight)
    state = gates["fsm_state"]
    if not hasattr(env, "_yaw_fsm_positive_budget"):
        env._yaw_fsm_positive_budget = torch.zeros(
            env.num_envs, 7, dtype=positive.dtype, device=positive.device
        )
        env._yaw_fsm_positive_budget_pos = torch.zeros_like(positive)
        env._yaw_fsm_positive_budget_neg = torch.zeros_like(positive)
    env._yaw_fsm_positive_budget.scatter_add_(1, state.unsqueeze(1), positive.unsqueeze(1))
    env._yaw_fsm_positive_budget_pos += positive * gates["diag_pos"].to(positive.dtype)
    env._yaw_fsm_positive_budget_neg += positive * gates["diag_neg"].to(positive.dtype)


def fsm_gated_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    fsm_command_name: str,
    support_sensor_cfg: SceneEntityCfg,
    support_sensor_cfg_mirror: SceneEntityCfg,
    lifted_asset_cfg: SceneEntityCfg,
    lifted_asset_cfg_mirror: SceneEntityCfg,
    wheel_radius: float,
    target_clearance: float,
    std: float,
    contact_threshold: float = 1.0,
    clearance_gate_floor: float = 0.0,
    clearance_gate_floor_decay_s: float = 0.0,
    edge_command_fraction: float = 0.80,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track signed yaw commands during TRANSITION and YAW states.

    Tracking and lift-scaled support contact are available during transition;
    curriculum tracking accumulators update only in ``YAW_POS``/``YAW_NEG``.
    The signed command preserves the direction constraint for either branch.
    """
    if std <= 0.0:
        raise ValueError("std must be positive.")
    if target_clearance <= 0.0:
        raise ValueError("target_clearance must be positive.")
    if not 0.0 <= clearance_gate_floor <= 1.0:
        raise ValueError("clearance_gate_floor must be in [0, 1].")
    if clearance_gate_floor_decay_s < 0.0:
        raise ValueError("clearance_gate_floor_decay_s must be non-negative.")
    if not 0.0 < edge_command_fraction <= 1.0:
        raise ValueError("edge_command_fraction must be in (0, 1].")

    gates = fsm_gates(env, fsm_command_name)
    _fsm_step_telemetry(env, gates)
    support_pos_contact = _yaw_wheel_contacts(env, support_sensor_cfg, contact_threshold)
    support_neg_contact = _yaw_wheel_contacts(env, support_sensor_cfg_mirror, contact_threshold)
    support_pos = support_pos_contact.to(dtype=torch.float32).prod(dim=1)
    support_neg = support_neg_contact.to(dtype=torch.float32).prod(dim=1)
    support_gate = torch.where(gates["diag_pos"], support_pos, support_neg)
    lost_support = gates["b_yaw"] & (support_gate == 0.0)
    if not hasattr(env, "_yaw_fsm_support_loss_run_steps"):
        env._yaw_fsm_support_loss_run_steps = torch.zeros(
            env.num_envs, device=support_gate.device, dtype=torch.long
        )
    env._yaw_fsm_support_loss_run_steps = torch.where(
        lost_support,
        env._yaw_fsm_support_loss_run_steps + 1,
        torch.zeros_like(env._yaw_fsm_support_loss_run_steps),
    )
    for suffix, direction in (("pos", gates["diag_pos"]), ("neg", gates["diag_neg"])):
        name = f"_yaw_fsm_{suffix}_support_loss_max_steps"
        if not hasattr(env, name):
            setattr(env, name, torch.zeros_like(env._yaw_fsm_support_loss_run_steps))
        previous_max = getattr(env, name)
        previous_max.copy_(torch.where(
            direction & lost_support,
            torch.maximum(previous_max, env._yaw_fsm_support_loss_run_steps),
            previous_max,
        ))
    support_contacts = torch.where(
        gates["diag_pos"].unsqueeze(1), support_pos_contact, support_neg_contact
    )
    support_shape = _yaw_support_shape(support_contacts)
    # The opposite support diagonal is exactly the active swing diagonal.
    swing_contact = select_swing_wheel_contact(
        support_pos_contact,
        support_neg_contact,
        gates["support_diagonal"],
    )

    lift_pos = _yaw_lift_progress(env, lifted_asset_cfg, wheel_radius, target_clearance)
    lift_neg = _yaw_lift_progress(env, lifted_asset_cfg_mirror, wheel_radius, target_clearance)
    selected_lift = torch.where(gates["diag_pos"].unsqueeze(1), lift_pos, lift_neg)
    lift_progress = selected_lift.mean(dim=1)

    asset: RigidObject = env.scene[asset_cfg.name]
    _fsm_transition_telemetry(
        env, gates, support_contacts, selected_lift, asset,
        env.command_manager.get_term(fsm_command_name),
    )
    yaw_command = env.command_manager.get_command(command_name)[:, 0]
    yaw_error = torch.abs(yaw_command - asset.data.root_ang_vel_b[:, 2])
    yaw_tracking = torch.exp(-yaw_error.square() / std**2)

    # Metrics are intentionally YAW-only; otherwise transition/FOUR steps
    # dilute the curriculum score and make the scale ladder stall.
    yaw_mask = gates["b_yaw"].to(dtype=yaw_command.dtype)
    # Reuse the legacy yaw-curriculum accumulator names.  This keeps the
    # existing checkpoint/export path valid while changing only the sample
    # mask: FSM tracking statistics are normalized by YAW-state steps.
    _fsm_masked_accumulate(
        env,
        support_gate,
        yaw_mask,
        "_yaw_support_score_sum",
        "_yaw_support_score_samples",
    )
    _fsm_masked_accumulate(
        env,
        support_gate,
        yaw_mask,
        "_yaw_gate_open_sum",
        "_yaw_gate_open_samples",
    )
    _fsm_masked_accumulate_pair(
        env,
        torch.abs(yaw_command),
        yaw_error,
        yaw_mask,
        "_yaw_command_abs_sum",
        "_yaw_rate_abs_error_sum",
        "_yaw_tracking_metric_samples",
    )

    # The legacy accumulators above deliberately remain direction-agnostic:
    # the original yaw curriculum and checkpoint export consume them.  The
    # FSM curriculum additionally needs independent evidence for each
    # diagonal; a strong POS branch must never promote a weak NEG branch.
    yaw_pos_mask = gates["b_yaw"] & gates["diag_pos"]
    yaw_neg_mask = gates["b_yaw"] & gates["diag_neg"]
    for suffix, contacts, direction_mask, wheel_names in (
        ("pos", support_pos_contact, yaw_pos_mask, ("FL", "HR")),
        ("neg", support_neg_contact, yaw_neg_mask, ("FR", "HL")),
    ):
        for index, wheel_name in enumerate(wheel_names):
            _fsm_masked_accumulate(
                env,
                contacts[:, index].to(dtype=yaw_command.dtype),
                direction_mask.to(dtype=yaw_command.dtype),
                f"_yaw_fsm_{suffix}_support_{wheel_name}_sum",
                f"_yaw_fsm_{suffix}_support_{wheel_name}_samples",
            )
    for suffix, direction_mask in (("pos", yaw_pos_mask), ("neg", yaw_neg_mask)):
        mask = direction_mask.to(dtype=yaw_command.dtype)
        _fsm_masked_accumulate(
            env,
            lift_progress,
            mask,
            f"_yaw_fsm_{suffix}_lift_sum",
            f"_yaw_fsm_{suffix}_yaw_samples",
        )
        _fsm_masked_accumulate(
            env,
            support_gate,
            mask,
            f"_yaw_fsm_{suffix}_support_sum",
            f"_yaw_fsm_{suffix}_support_samples",
        )
        _fsm_masked_accumulate_pair(
            env,
            torch.abs(yaw_command),
            yaw_error,
            mask,
            f"_yaw_fsm_{suffix}_command_abs_sum",
            f"_yaw_fsm_{suffix}_yaw_abs_error_sum",
            f"_yaw_fsm_{suffix}_tracking_samples",
        )
        _fsm_masked_accumulate(
            env,
            swing_contact.to(dtype=yaw_command.dtype),
            mask,
            f"_yaw_fsm_{suffix}_swing_contact_sum",
            f"_yaw_fsm_{suffix}_swing_contact_samples",
        )

    # Phase B is an attempted transition that reaches its corresponding YAW
    # state before reset.  Store booleans rather than a step count so a long
    # transition cannot distort the episode success rate.
    _fsm_episode_or(
        env,
        "_yaw_fsm_pos_transition_attempted",
        gates["b_trans"] & gates["diag_pos"],
    )
    _fsm_episode_or(
        env,
        "_yaw_fsm_neg_transition_attempted",
        gates["b_trans"] & gates["diag_neg"],
    )
    _fsm_episode_or(env, "_yaw_fsm_pos_transition_succeeded", yaw_pos_mask)
    _fsm_episode_or(env, "_yaw_fsm_neg_transition_succeeded", yaw_neg_mask)
    env._yaw_support_score_current = support_gate
    env._yaw_gate_open_current = support_gate
    env._yaw_command_abs_current = torch.abs(yaw_command)
    env._yaw_rate_abs_error_current = yaw_error

    command_term = env.command_manager.get_term(command_name)
    yaw_limits = torch.as_tensor(
        command_term.cfg.yaw_rate_range,
        device=yaw_command.device,
        dtype=yaw_command.dtype,
    )
    yaw_limit = yaw_limits.abs().amax()
    edge_mask = (torch.abs(yaw_command) >= edge_command_fraction * yaw_limit) & gates["b_yaw"]
    _fsm_masked_accumulate_pair(
        env,
        torch.abs(yaw_command),
        yaw_error,
        edge_mask.to(dtype=yaw_command.dtype),
        "_yaw_edge_command_abs_sum",
        "_yaw_edge_rate_abs_error_sum",
        "_yaw_edge_tracking_samples",
    )

    # The phase-B floor provides a short exploration bridge at transition
    # entry, then decays to zero so the policy cannot keep collecting reward
    # without lifting.  A zero decay keeps the legacy/static-floor behavior.
    if clearance_gate_floor_decay_s > 0.0:
        floor = clearance_gate_floor * torch.clamp(
            1.0 - gates["state_time"] / clearance_gate_floor_decay_s,
            min=0.0,
            max=1.0,
        )
    else:
        floor = clearance_gate_floor
    clearance_weight = floor + (1.0 - floor) * lift_progress
    state_gate = gates["f_trans"] + gates["f_yaw"]
    tracking = support_gate * clearance_weight * yaw_tracking * state_gate
    # Contact alone gives no bonus while the swing pair is still on the ground.
    support_bonus = 0.25 * support_shape * lift_progress * state_gate
    reward = tracking + support_bonus
    _fsm_record_positive_budget(env, gates, "fsm_gated_tracking", reward)
    return reward


def four_stand_stability(
    env: ManagerBasedRLEnv,
    fsm_command_name: str,
    target_height: float,
    height_std: float = 0.08,
    attitude_std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a stable four-wheel pose with a soft FSM phase gate.

    SAFE_RECOVERY is intentionally absent from the gate: recovery gets only
    its entry cost and baseline penalties, never a positive stability reward.
    """
    if height_std <= 0.0 or attitude_std <= 0.0:
        raise ValueError("height_std and attitude_std must be positive.")

    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    height_score = torch.exp(-(base_height - target_height).square() / height_std**2)
    attitude_score = torch.exp(-(roll.square() + pitch.square()) / attitude_std**2)
    gates = fsm_gates(env, fsm_command_name)
    reward = gates["f_stability"] * height_score * attitude_score
    _fsm_record_positive_budget(env, gates, "four_stand_stability", reward)
    return reward


class ReturnToFourLanding(ManagerTermBase):
    """Reward lowering progress and each regained contact once per return."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_progress = torch.zeros(env.num_envs, 2, device=env.device)
        self.contact_seen = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        self.was_return = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        index = slice(None) if env_ids is None else env_ids
        self.prev_progress[index] = 0.0
        self.contact_seen[index] = False
        self.was_return[index] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        asset_cfg_mirror: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        sensor_cfg_mirror: SceneEntityCfg,
        wheel_radius: float,
        fsm_command_name: str,
        contact_threshold: float = 1.0,
    ) -> torch.Tensor:
        gates = fsm_gates(env, fsm_command_name)
        target_clearance = env.command_manager.get_term(fsm_command_name).cfg.target_clearance
        progress_pos = _yaw_lift_progress(env, asset_cfg, wheel_radius, target_clearance)
        progress_neg = _yaw_lift_progress(env, asset_cfg_mirror, wheel_radius, target_clearance)
        contact_pos = _yaw_wheel_contacts(env, sensor_cfg, contact_threshold)
        contact_neg = _yaw_wheel_contacts(env, sensor_cfg_mirror, contact_threshold)
        progress = torch.where(gates["diag_pos"].unsqueeze(1), progress_pos, progress_neg)
        contact = torch.where(gates["diag_pos"].unsqueeze(1), contact_pos, contact_neg)
        in_return = gates["b_return"]
        continuing = in_return & self.was_return & ~gates["just_switched"]
        lowering = (self.prev_progress - progress).mean(dim=1) * continuing
        # Contacts present at return entry were never lost and earn no bonus.
        first_contacts = (contact & ~self.contact_seen & continuing.unsqueeze(1)).float().mean(dim=1)
        reward = lowering + 2.0 * first_contacts
        self.prev_progress.copy_(torch.where(in_return.unsqueeze(1), progress, torch.zeros_like(progress)))
        self.contact_seen.copy_(torch.where(
            in_return.unsqueeze(1), self.contact_seen | contact, torch.zeros_like(contact)
        ))
        self.was_return.copy_(in_return)
        _fsm_record_positive_budget(env, gates, "return_to_four_landing", reward)
        return reward


def four_stand_ready_bonus(env: ManagerBasedRLEnv, fsm_command_name: str) -> torch.Tensor:
    """Pay once when RETURN_TO_4 exits through four-wheel readiness."""
    gates = fsm_gates(env, fsm_command_name)
    reward = gates["b_return_complete"].to(dtype=torch.float32)
    _fsm_record_positive_budget(env, gates, "four_stand_ready_bonus", reward)
    return reward


class TransitionProgress(ManagerTermBase):
    """Potential-based lift/handoff shaping for TRANSITION and RETURN."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_phi = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.prev_phi.zero_()
        else:
            self.prev_phi[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        asset_cfg_mirror: SceneEntityCfg,
        wheel_radius: float,
        target_clearance: float,
        fsm_command_name: str,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1].")

        gates = fsm_gates(env, fsm_command_name)
        progress_pos = _yaw_lift_progress(
            env, asset_cfg, wheel_radius, target_clearance
        ).mean(dim=1)
        progress_neg = _yaw_lift_progress(
            env, asset_cfg_mirror, wheel_radius, target_clearance
        ).mean(dim=1)
        progress = torch.where(gates["diag_pos"], progress_pos, progress_neg)
        phi = torch.where(gates["b_return"], 1.0 - progress, progress)

        # Always advance the potential, including FOUR/YAW/SAFE states.  This
        # prevents a spurious pulse when the next gated phase begins.
        reward = gamma * phi - self.prev_phi
        self.prev_phi.copy_(phi)
        reward = torch.where(gates["just_switched"], torch.zeros_like(reward), reward)
        reward = reward * (gates["f_trans"] + gates["f_return"])
        _fsm_record_positive_budget(env, gates, "transition_progress", reward)
        return reward


def spin_center_drift(
    env: ManagerBasedRLEnv,
    fsm_command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize accumulated XY drift from the yaw-entry position."""
    asset: Articulation = env.scene[asset_cfg.name]
    current_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    gates = fsm_gates(env, fsm_command_name)
    command = env.command_manager.get_term(fsm_command_name)
    entry_xy = command.yaw_entry_pos
    drift = torch.linalg.vector_norm(current_xy - entry_xy, dim=1)
    for suffix, direction_mask in (
        ("pos", gates["b_yaw"] & gates["diag_pos"]),
        ("neg", gates["b_yaw"] & gates["diag_neg"]),
    ):
        _fsm_masked_accumulate(
            env,
            drift,
            direction_mask.to(dtype=drift.dtype),
            f"_yaw_fsm_{suffix}_drift_sum",
            f"_yaw_fsm_{suffix}_drift_samples",
        )
        # Record a single sample at the ten-second YAW milestone.  This is a
        # radius-at-time metric, not an average that can hide late drift.
        milestone = direction_mask & (gates["state_time"] >= 10.0) & (
            gates["state_time"] < 10.0 + env.step_dt
        )
        _fsm_masked_accumulate(
            env,
            drift,
            milestone.to(dtype=drift.dtype),
            f"_yaw_fsm_{suffix}_drift_10s_sum",
            f"_yaw_fsm_{suffix}_drift_10s_samples",
        )
    return drift * gates["f_yaw"]


def safe_recovery_entry(
    env: ManagerBasedRLEnv,
    fsm_command_name: str,
) -> torch.Tensor:
    """Return one event pulse on entry to SAFE_RECOVERY."""
    gates = fsm_gates(env, fsm_command_name)
    return (gates["b_safe"] & gates["just_switched"]).to(dtype=torch.float32)


def _yaw_command_penalty_scale(
    env: ManagerBasedRLEnv,
    command_name: str,
    yaw_reference: float,
    minimum_scale: float = 0.0,
) -> torch.Tensor:
    """Return command relief against a fixed reference, optionally with a floor."""
    if yaw_reference <= 0.0:
        raise ValueError("yaw_reference must be positive.")
    if not 0.0 <= minimum_scale <= 1.0:
        raise ValueError("minimum_scale must be in [0, 1].")
    yaw_command = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    scale = 1.0 - torch.clamp(yaw_command / yaw_reference, min=0.0, max=1.0)
    return torch.clamp(scale, min=minimum_scale)


def yaw_lateral_wheel_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize contacted wheels' velocity along their local axle direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    velocity_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    orientation_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    velocity_local = math_utils.quat_apply_inverse(orientation_w, velocity_w)
    in_contact = _yaw_wheel_contacts(env, sensor_cfg, threshold)
    return torch.sum(in_contact * velocity_local[..., 1].square(), dim=1)


def yaw_rolling_wheel_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    body_asset_cfg: SceneEntityCfg,
    joint_asset_cfg: SceneEntityCfg,
    wheel_radius: float,
    command_name: str,
    yaw_reference: float,
    threshold: float = 1.0,
    sensor_cfg_mirror: SceneEntityCfg | None = None,
    body_asset_cfg_mirror: SceneEntityCfg | None = None,
    joint_asset_cfg_mirror: SceneEntityCfg | None = None,
    fsm_command_name: str | None = None,
) -> torch.Tensor:
    """Penalize rolling error less aggressively as requested yaw rises."""
    asset: Articulation = env.scene[body_asset_cfg.name]
    velocity_w = asset.data.body_lin_vel_w[:, body_asset_cfg.body_ids]
    orientation_w = asset.data.body_quat_w[:, body_asset_cfg.body_ids]
    forward_velocity = math_utils.quat_apply_inverse(orientation_w, velocity_w)[..., 0]
    wheel_velocity = asset.data.joint_vel[:, joint_asset_cfg.joint_ids]
    rolling_error = forward_velocity + wheel_radius * wheel_velocity
    in_contact = _yaw_wheel_contacts(env, sensor_cfg, threshold)
    penalty = torch.sum(in_contact * rolling_error.square(), dim=1)
    score = _yaw_command_penalty_scale(env, command_name, yaw_reference) * penalty
    if fsm_command_name is None:
        return score
    if sensor_cfg_mirror is None or body_asset_cfg_mirror is None or joint_asset_cfg_mirror is None:
        raise ValueError("rolling-slip mirror configs are required when fsm_command_name is set.")

    mirror_asset: Articulation = env.scene[body_asset_cfg_mirror.name]
    mirror_velocity_w = mirror_asset.data.body_lin_vel_w[:, body_asset_cfg_mirror.body_ids]
    mirror_orientation_w = mirror_asset.data.body_quat_w[:, body_asset_cfg_mirror.body_ids]
    mirror_forward_velocity = math_utils.quat_apply_inverse(
        mirror_orientation_w, mirror_velocity_w
    )[..., 0]
    mirror_wheel_velocity = mirror_asset.data.joint_vel[:, joint_asset_cfg_mirror.joint_ids]
    mirror_error = mirror_forward_velocity + wheel_radius * mirror_wheel_velocity
    mirror_contact = _yaw_wheel_contacts(env, sensor_cfg_mirror, threshold)
    mirror_penalty = torch.sum(mirror_contact * mirror_error.square(), dim=1)
    mirror_score = _yaw_command_penalty_scale(
        env, command_name, yaw_reference
    ) * mirror_penalty
    gates = fsm_gates(env, fsm_command_name)
    return torch.where(gates["diag_pos"], score, mirror_score) * gates["f_yaw"]


def yaw_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize first-order changes in the complete action vector."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )


def yaw_joint_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize velocity of explicitly selected joints without terrain gating."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(asset.data.joint_vel[:, asset_cfg.joint_ids].square(), dim=1)


def yaw_joint_torque_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    yaw_reference: float,
    minimum_scale: float,
) -> torch.Tensor:
    """Penalize torque with command-relative relief during intentional rotation."""
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(asset.data.applied_torque[:, asset_cfg.joint_ids].square(), dim=1)
    scale = _yaw_command_penalty_scale(env, command_name, yaw_reference, minimum_scale)
    return scale * penalty


def yaw_lifted_wheel_spin_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    yaw_reference: float,
    asset_cfg_mirror: SceneEntityCfg | None = None,
    fsm_command_name: str | None = None,
) -> torch.Tensor:
    """Penalize lifted-wheel spin with relief during intentional rotation."""
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(asset.data.joint_vel[:, asset_cfg.joint_ids].square(), dim=1)
    score = _yaw_command_penalty_scale(env, command_name, yaw_reference) * penalty
    if fsm_command_name is None:
        return score
    if asset_cfg_mirror is None:
        raise ValueError("asset_cfg_mirror is required when fsm_command_name is set.")
    mirror_asset: Articulation = env.scene[asset_cfg_mirror.name]
    mirror_penalty = torch.sum(
        mirror_asset.data.joint_vel[:, asset_cfg_mirror.joint_ids].square(), dim=1
    )
    mirror_score = _yaw_command_penalty_scale(
        env, command_name, yaw_reference
    ) * mirror_penalty
    gates = fsm_gates(env, fsm_command_name)
    return torch.where(gates["diag_pos"], score, mirror_score) * gates["f_yaw"]


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward * get_gait_level_tensor(env)


def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] > 0
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward * get_gait_level_tensor(env)


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def feet_air_time(
#     env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
# ) -> torch.Tensor:
#     """Reward long steps taken by the feet using L2-kernel.

#     This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
#     that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
#     the time for which the feet are in the air.

#     If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
#     """
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
#     # no reward for zero command
#     reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
#     # print(torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1), "command norm")
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    # return reward

# def feet_air_time(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     sensor_cfg: SceneEntityCfg,
#     mode_time: float,
#     velocity_threshold: float,
# ) -> torch.Tensor:
#     """Reward longer feet air and contact time."""
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     asset: Articulation = env.scene[asset_cfg.name]
#     if contact_sensor.cfg.track_air_time is False:
#         raise RuntimeError("Activate ContactSensor's track_air_time!")
#     # compute the reward
#     current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
#     current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

#     t_max = torch.max(current_air_time, current_contact_time)
#     t_min = torch.clip(t_max, max=mode_time)
#     stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
#     cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
#     body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
#     reward = torch.where(
#         torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
#         torch.where(t_max < mode_time, t_min, 0),
#         stance_cmd_reward,
#     )
#     return torch.sum(reward, dim=1)


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1)
    # print(last_air_time, "last air time")
    # print(last_contact_time, "last contact time")
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward




def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.5
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # print(contact, "contact")
    reward = torch.sum(contact, dim=-1).float()
    # print(reward, "reward after sum")
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.5
    # print(env.command_manager.get_command(command_name), "env.command_manager.get_command(command_name)")
    # print(reward, "reward after multiply")
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    # foot_velocity_tanh = torch.tanh(
    #     tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    # )
    # reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward = torch.sum(foot_z_target_error, dim=1)
    # print(foot_z_target_error, "foot_z_target_error")
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.2
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def _bernstein_torch(n: int, k: int, t: torch.Tensor) -> torch.Tensor:
    """Bernstein basis B_k^n(t) for tensor t in [0, 1]."""
    coeff = float(math.comb(n, k))
    return coeff * (1.0 - t) ** (n - k) * t**k


def _bezier_curve_torch(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate Bezier curve points for batched parameter t.

    Args:
        control_points: Tensor of shape [m, 2].
        t: Tensor of shape [N, L] in [0, 1].

    Returns:
        Tensor of shape [N, L, 2].
    """
    n = control_points.shape[0] - 1
    out = torch.zeros(*t.shape, 2, device=t.device, dtype=t.dtype)
    for k in range(n + 1):
        out = out + _bernstein_torch(n, k, t).unsqueeze(-1) * control_points[k]
    return out


def _bezier_curve_derivative_torch(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate d/dt of Bezier curve points for batched parameter t."""
    n = control_points.shape[0] - 1
    delta_ctrl = n * (control_points[1:] - control_points[:-1])
    out = torch.zeros(*t.shape, 2, device=t.device, dtype=t.dtype)
    for k in range(n):
        out = out + _bernstein_torch(n - 1, k, t).unsqueeze(-1) * delta_ctrl[k]
    return out


def phase_foot_trajectory_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 0.1,
    command_threshold: float = 0.1,
    cycle_time: float = 0.4,
    phase_offsets: tuple[float, ...] = (0.0, 1.0, 1.0, 0.0),
    gait_span: float = -0.008,
    gait_psi: float = 0.15,
    gait_delta: float = 0.03,
    x_offset: float = 0.0,
    stance_span: float = 0.20,
    stand_ref_z_offset: float = -0.2,
    velocity_weight: float = 0.5,
) -> torch.Tensor:
    """Track MuJoCo-style phase foot trajectory in body frame with exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    num_feet = len(body_ids)

    if num_feet == 0:
        return torch.zeros(env.num_envs, device=env.device)
    if len(phase_offsets) != num_feet:
        raise ValueError(f"phase_offsets length ({len(phase_offsets)}) must match tracked feet ({num_feet}).")

    # Build and cache base-fixed stand references from the first call.
    if (not hasattr(env, "phase_foot_ref_body")) or (env.phase_foot_ref_body.shape[1] != num_feet):
        rel_foot_pos_w = asset.data.body_pos_w[:, body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
        foot_pos_b = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
        for i in range(num_feet):
            foot_pos_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, rel_foot_pos_w[:, i, :])
        ref = foot_pos_b[0].detach().clone()
        ref[:, 2] += stand_ref_z_offset
        env.phase_foot_ref_body = ref.unsqueeze(0)

    stand_ref_body = env.phase_foot_ref_body.to(env.device).expand(env.num_envs, -1, -1)

    # Build phase S in [0, 2).
    phase_time = env.episode_length_buf.float() * env.step_dt
    phase_offsets_t = torch.tensor(phase_offsets, device=env.device, dtype=phase_time.dtype).unsqueeze(0)
    S = torch.remainder((2.0 * phase_time / max(cycle_time, 1e-6)).unsqueeze(1) + phase_offsets_t, 2.0)

    # MuJoCo-like piecewise trajectory in local (q, z).
    tau = float(gait_span)
    psi = float(gait_psi)
    delta = float(gait_delta)
    stance_span = float(stance_span)
    stance_span = min(max(stance_span, 1e-6), 2.0 - 1e-6)

    q = torch.zeros_like(S)
    z = torch.zeros_like(S)
    dq_dS = torch.zeros_like(S)
    dz_dS = torch.zeros_like(S)

    stance_mask = S < stance_span
    if stance_mask.any():
        s_stance = S / stance_span
        q_stance = tau * (1.0 - 2.0 * s_stance)
        z_stance = torch.full_like(S, delta)
        dq_dS_stance = torch.full_like(S, -2.0 * tau / stance_span)
        dz_dS_stance = torch.zeros_like(S)

        q = torch.where(stance_mask, q_stance, q)
        z = torch.where(stance_mask, z_stance, z)
        dq_dS = torch.where(stance_mask, dq_dS_stance, dq_dS)
        dz_dS = torch.where(stance_mask, dz_dS_stance, dz_dS)

    swing_mask = ~stance_mask
    if swing_mask.any():
        t_bezier = torch.clamp((S - stance_span) / (2.0 - stance_span), 0.0, 1.0)
        ctrl = torch.tensor(
            [
                [-tau, 0.0],
                [-0.95 * tau, 0.80 * psi],
                [-0.55 * tau, 1.00 * psi],
                [0.55 * tau, 1.00 * psi],
                [0.95 * tau, 0.80 * psi],
                [tau, 0.0],
            ],
            device=env.device,
            dtype=S.dtype,
        )
        qz_swing = _bezier_curve_torch(ctrl, t_bezier)
        dqz_dt = _bezier_curve_derivative_torch(ctrl, t_bezier)
        dt_dS = 1.0 / (2.0 - stance_span)

        q = torch.where(swing_mask, qz_swing[..., 0], q)
        z = torch.where(swing_mask, qz_swing[..., 1] + delta, z)
        dq_dS = torch.where(swing_mask, dqz_dt[..., 0] * dt_dS, dq_dS)
        dz_dS = torch.where(swing_mask, dqz_dt[..., 1] * dt_dS, dz_dS)

    dS_dt = 2.0 / max(cycle_time, 1e-6)
    dq_dt = dq_dS * dS_dt
    dz_dt = dz_dS * dS_dt

    ref_pos_b = stand_ref_body + torch.stack(
        [q + float(x_offset), torch.zeros_like(q), z],
        dim=-1,
    )
    ref_vel_b = torch.stack(
        [dq_dt, torch.zeros_like(dq_dt), dz_dt],
        dim=-1,
    )

    # Actual foot states in body frame.
    rel_foot_pos_w = asset.data.body_pos_w[:, body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    rel_foot_vel_w = asset.data.body_lin_vel_w[:, body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    foot_pos_b = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
    foot_vel_b = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
    for i in range(num_feet):
        foot_pos_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, rel_foot_pos_w[:, i, :])
        foot_vel_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, rel_foot_vel_w[:, i, :])

    pos_offset = foot_pos_b - ref_pos_b
    vel_offset = foot_vel_b - ref_vel_b

    # Per-dimension (x, y, z) errors over feet for each environment.
    pos_err = torch.sum(torch.square(pos_offset), dim=1)
    vel_err = torch.sum(torch.square(vel_offset), dim=1)

    # Scalar total error for reward computation.
    total_err = torch.sum(pos_err, dim=1) + float(velocity_weight) * torch.sum(vel_err, dim=1)
    reward = torch.exp(-total_err / max(std, 1e-6) ** 2)

    # Command-gating follows full (x, y, yaw) command magnitude.
    command = env.command_manager.get_command(command_name)
    gate = torch.linalg.norm(command[:, :3], dim=1) > command_threshold

    # info for debugging
    pos_offset_xyz_mean = torch.mean(pos_offset, dim=(0, 1))
    vel_offset_xyz_mean = torch.mean(vel_offset, dim=(0, 1))
    # print(
    #     "Offset xyz mean | "
    #     f"pos(x,y,z)=({pos_offset_xyz_mean[0].item():.4f}, {pos_offset_xyz_mean[1].item():.4f}, {pos_offset_xyz_mean[2].item():.4f}) | "
    #     f"vel(x,y,z)=({vel_offset_xyz_mean[0].item():.4f}, {vel_offset_xyz_mean[1].item():.4f}, {vel_offset_xyz_mean[2].item():.4f})"
    # )
    # print("Reward:", reward * gate.float())
    return reward * gate.float() * get_gait_level_tensor(env)

def foot_impact_velocity(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    speed_threshold: float = 0.10,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].float()
    foot_lin_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]

    downward_speed = torch.clamp(-foot_lin_vel[:, :, 2], min=0.0)
    downward_speed = torch.clamp(downward_speed - speed_threshold, min=0.0)

    penalty = torch.sum(first_contact * torch.square(downward_speed), dim=1)
    return penalty * get_gait_level_tensor(env)

# def stand_still_joint_deviation_l1(
#     env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Penalize offsets from the default joint positions when the command is very small."""
    # command = env.command_manager.get_command(command_name)
#     # Penalize motion when command is nearly zero.
#     return joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :], dim=1) < command_threshold)

# def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize joint positions that deviate from the default one."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # compute out of limits constraints
#     angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
#     return torch.sum(torch.abs(angle), dim=1)


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def joint_acc_l2_new(env: ManagerBasedRLEnv) -> torch.Tensor:

# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     # print(torch.sum(diff, dim=1), "smoothness l2")
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward





def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward * get_gait_level_tensor(env)


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_air_time_lin_xy_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Air-time reward gated by planar linear velocity command (x, y)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Core logic unchanged
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    # Gate by planar linear velocity command only
    cmd_lin_xy = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    reward *= cmd_lin_xy > cmd_threshold
    return reward * get_gait_level_tensor(env)

def feet_air_time_x_neg_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Air-time reward gated by negative x velocity command only."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Core logic unchanged
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    # Gate: x command must be negative and exceed magnitude threshold
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    reward *= cmd_x < 0.0

    return reward

def feet_air_time_ang_z_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Air-time reward gated by yaw angular velocity command (z)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Core logic unchanged
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    # Gate by angular velocity command only
    cmd_ang_z = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    reward *= cmd_ang_z > cmd_threshold
    return reward * get_gait_level_tensor(env)

def feet_air_time_including_ang_z(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    return reward

def lin_vel_xy_l2_with_ang_z_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Penalize xy-axis base linear velocity using L2 squared kernel if command is ang_vel_z."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    command = env.command_manager.get_command(command_name)
    reward *= (torch.sum(torch.square(command[:, 2:]), dim=1) > command_threshold) & \
            (torch.sum(torch.square(command[:, :2]), dim=1) < command_threshold)
    # reward *= torch.sum(torch.square(env.command_manager.get_command(command_name)[:, 2:]), dim=1) > command_threshold
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def contact_forces_FL_HR(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    scale: float = 20.0,
) -> torch.Tensor:
    """Reward simultaneous contact of FL and HR wheels at current timestep."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # net_forces_w shape:
    # (num_envs, num_sensor_bodies, 3)
    #
    # Select only FL and HR using body_ids
    # -> (num_envs, 2, 3)
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

    # Force magnitude of FL and HR
    # -> (num_envs, 2)
    force_mag = torch.norm(forces, dim=-1)

    # Only reward contact force above threshold
    excess_force = torch.clamp(force_mag - threshold, min=0.0)

    # Smooth score [0, 1)
    # -> (num_envs, 2)
    contact_score = torch.tanh(excess_force / scale)

    # Require BOTH FL and HR to have contact
    # -> (num_envs,)
    reward = contact_score[:, 0] * contact_score[:, 1]

    return reward * get_gait_level_tensor(env)
