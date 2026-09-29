"""Command-conditioned rewards and metrics for the neutral/positive yaw task."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from .rewards import (
    _yaw_command_penalty_scale,
    _yaw_lift_progress,
    _yaw_support_geometry,
    _yaw_wheel_contacts,
    yaw_support_contact,
)


def yaw_pos_masks(env, command_name: str, deadband: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the shared command, POS mask, and closed-interval neutral mask."""
    command = env.command_manager.get_command(command_name)[:, 0]
    active = command > deadband
    return command, active, command.abs() <= deadband


def _accumulate(env, name: str, value: torch.Tensor, mask: torch.Tensor) -> None:
    sum_name, count_name = f"{name}_sum", f"{name}_samples"
    if not hasattr(env, sum_name):
        setattr(env, sum_name, torch.zeros_like(value))
        setattr(env, count_name, torch.zeros_like(value, dtype=torch.long))
    getattr(env, sum_name).add_(torch.where(mask, value, 0.0))
    getattr(env, count_name).add_(mask.long())


def yaw_pos_com_support(env, asset_cfg: SceneEntityCfg, std: float, command_name: str, deadband: float) -> torch.Tensor:
    _, active, _ = yaw_pos_masks(env, command_name, deadband)
    distance, _, _ = _yaw_support_geometry(env, asset_cfg)
    return torch.where(active, 1.0 / (1.0 + distance / std), 0.0)


def yaw_pos_support_span_band_l2(
    env, asset_cfg: SceneEntityCfg, minimum_span: float, maximum_span: float, std: float,
    command_name: str, deadband: float,
) -> torch.Tensor:
    _, active, _ = yaw_pos_masks(env, command_name, deadband)
    _, _, span = _yaw_support_geometry(env, asset_cfg)
    outside = torch.relu(minimum_span - span) + torch.relu(span - maximum_span)
    return torch.where(active, (outside / std).square(), 0.0)


def yaw_pos_lift_clearance(
    env, asset_cfg: SceneEntityCfg, support_sensor_cfg: SceneEntityCfg, wheel_radius: float,
    target_clearance: float, command_name: str, deadband: float, contact_threshold: float = 1.0,
) -> torch.Tensor:
    _, active, _ = yaw_pos_masks(env, command_name, deadband)
    progress = _yaw_lift_progress(env, asset_cfg, wheel_radius, target_clearance)
    support = yaw_support_contact(env, support_sensor_cfg, contact_threshold)
    minimum = progress.amin(dim=1) * support
    _accumulate(env, "_yaw_lift_min_progress", minimum, active)
    signed = 2.0 * progress.mean(dim=1) - 1.0
    # Preserve the penalty during acquisition; full positive credit needs both supports.
    shaped = signed.clamp(max=0.0) + signed.clamp(min=0.0) * (0.25 + 0.75 * support)
    return torch.where(active, shaped, 0.0)


def yaw_pos_com_inside_segment(env, asset_cfg: SceneEntityCfg, std: float, command_name: str, deadband: float) -> torch.Tensor:
    _, active, _ = yaw_pos_masks(env, command_name, deadband)
    _, projection, length = _yaw_support_geometry(env, asset_cfg)
    outside = (torch.relu(-projection) + torch.relu(projection - 1.0)) * length
    return torch.where(active, torch.exp(-outside.square() / std**2), 0.0)


def _four_contact_score(env, sensor_cfg: SceneEntityCfg, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    contacts = _yaw_wheel_contacts(env, sensor_cfg, threshold).float()
    all_four = contacts.prod(dim=1)
    return 0.2 * contacts.mean(dim=1) + 0.8 * all_four, all_four


def yaw_pos_four_wheel_contact(
    env, sensor_cfg: SceneEntityCfg, command_name: str, deadband: float, threshold: float = 1.0,
) -> torch.Tensor:
    _, _, neutral = yaw_pos_masks(env, command_name, deadband)
    score, all_four = _four_contact_score(env, sensor_cfg, threshold)
    _accumulate(env, "_yaw_pos_neutral_four_contact", all_four, neutral)
    return torch.where(neutral, score, 0.0)


def yaw_pos_neutral_landing_progress(
    env, asset_cfg: SceneEntityCfg, wheel_radius: float, target_clearance: float,
    command_name: str, deadband: float,
) -> torch.Tensor:
    """Credit each FR/HL wheel as it descends from target clearance to ground."""
    _, _, neutral = yaw_pos_masks(env, command_name, deadband)
    lift_progress = _yaw_lift_progress(env, asset_cfg, wheel_radius, target_clearance)
    return torch.where(neutral, 1.0 - lift_progress.mean(dim=1), 0.0)


def yaw_pos_four_stand_pose(
    env, asset_cfg: SceneEntityCfg, command_name: str, deadband: float, std: float = 0.25,
) -> torch.Tensor:
    _, _, neutral = yaw_pos_masks(env, command_name, deadband)
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    default = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    error = (q - default).square().mean(dim=1)
    _accumulate(env, "_yaw_pos_neutral_pose_error", error, neutral)
    return torch.where(neutral, torch.exp(-error / std**2), 0.0)


def yaw_pos_gated_tracking(
    env, command_name: str, support_sensor_cfg: SceneEntityCfg,
    lifted_asset_cfg: SceneEntityCfg, wheel_radius: float, target_clearance: float, std: float,
    deadband: float, contact_threshold: float = 1.0, clearance_gate_floor: float = 0.25,
    edge_command_fraction: float = 0.80, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command, active, neutral = yaw_pos_masks(env, command_name, deadband)
    robot = env.scene[asset_cfg.name]
    yaw_rate = robot.data.root_ang_vel_b[:, 2]
    support = yaw_support_contact(env, support_sensor_cfg, contact_threshold)
    lift = _yaw_lift_progress(env, lifted_asset_cfg, wheel_radius, target_clearance).mean(dim=1)
    clearance_weight = clearance_gate_floor + (1.0 - clearance_gate_floor) * lift
    tracking = torch.exp(-(yaw_rate - command).square() / std**2)
    active_score = support * clearance_weight * tracking
    neutral_score = torch.exp(-yaw_rate.square() / std**2)

    _accumulate(env, "_yaw_support_score", support, active)
    _accumulate(env, "_yaw_gate_open", support, active)
    _accumulate(env, "_yaw_active_yaw_score", active_score, active)
    _accumulate(env, "_yaw_command_abs", command.abs(), active)
    _accumulate(env, "_yaw_rate_abs_error", (yaw_rate - command).abs(), active)
    # Existing curriculum expects one common sample count for command/error.
    env._yaw_tracking_metric_samples = env._yaw_command_abs_samples
    env._yaw_command_abs_current = command.abs()
    env._yaw_rate_abs_error_current = (yaw_rate - command).abs()
    edge = active & (command >= edge_command_fraction * env.command_manager.get_term(command_name).cfg.yaw_rate_range[1])
    _accumulate(env, "_yaw_edge_command_abs", command.abs(), edge)
    _accumulate(env, "_yaw_edge_rate_abs_error", (yaw_rate - command).abs(), edge)
    env._yaw_edge_tracking_samples = env._yaw_edge_command_abs_samples
    _accumulate(env, "_yaw_pos_neutral_abs_yaw_rate", yaw_rate.abs(), neutral)
    planar_speed = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=1)
    _accumulate(env, "_yaw_pos_neutral_planar_speed", planar_speed, neutral)
    env._yaw_pos_neutral_samples = env._yaw_pos_neutral_abs_yaw_rate_samples
    return torch.where(active, active_score, torch.where(neutral, neutral_score, 0.0))


def yaw_pos_lifted_wheel_spin_l2(
    env, asset_cfg: SceneEntityCfg, command_name: str, deadband: float, yaw_reference: float,
) -> torch.Tensor:
    _, active, _ = yaw_pos_masks(env, command_name, deadband)
    robot = env.scene[asset_cfg.name]
    penalty = robot.data.joint_vel[:, asset_cfg.joint_ids].square().sum(dim=1)
    scale = _yaw_command_penalty_scale(env, command_name, yaw_reference)
    return torch.where(active, scale * penalty, 0.0)
