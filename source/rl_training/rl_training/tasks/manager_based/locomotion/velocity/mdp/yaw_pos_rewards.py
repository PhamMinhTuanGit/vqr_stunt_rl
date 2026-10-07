"""Command-conditioned rewards and metrics for the neutral/positive yaw task."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from .rewards import (
    _yaw_command_penalty_scale,
    _yaw_lift_progress,
    _yaw_support_geometry,
    _yaw_whole_body_com_xy,
    _yaw_wheel_contacts,
    yaw_support_contact,
)
from .wheel_contact_kinematics import contacted_wheel_velocities, rotate_vector, wheel_center_positions, wheel_center_velocities
from .yaw_pos_kinematics import (
    differential_rolling_kinematics, ground_heading_axes, ground_heading_yaw_rate, yaw_pos_motion_telemetry,
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


def yaw_pos_base_height_deficit_l2(
    env, command_name: str, minimum_height: float, error_scale: float = 0.05,
    low_speed_yaw: float = 0.5, low_speed_multiplier: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    active_minimum_height: float | None = None, deadband: float = 0.1,
) -> torch.Tensor:
    """Penalize crouching above the safety floor, with optional mode-specific minima.

    The POS configuration uses a constant multiplier at every commanded speed.
    Legacy callers retain the optional extra cost at low requested yaw.
    """
    if error_scale <= 0.0 or low_speed_yaw <= 0.0:
        raise ValueError("Height error_scale and low_speed_yaw must be positive.")
    if low_speed_multiplier < 1.0:
        raise ValueError("low_speed_multiplier must be at least one.")
    data = env.scene[asset_cfg.name].data
    height = data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    command = env.command_manager.get_command(command_name)[:, 0].abs()
    low_speed = (1.0 - command / low_speed_yaw).clamp(0.0, 1.0)
    minimum = torch.full_like(height, minimum_height)
    if active_minimum_height is not None:
        minimum = torch.where(command > deadband, active_minimum_height, minimum)
    deficit = (minimum - height).clamp_min(0.0)
    all_envs = torch.ones_like(command, dtype=torch.bool)
    _accumulate(env, "_yaw_pos_base_height", height, all_envs)
    _accumulate(env, "_yaw_pos_low_speed_base_height", height, command < low_speed_yaw)
    _accumulate(env, "_yaw_pos_low_speed_height_deficit", deficit, command < low_speed_yaw)
    _accumulate(env, "_yaw_pos_active_base_height_deficit", deficit, command > deadband)
    env._yaw_pos_active_base_height_deficit = deficit.detach().clone()
    return (1.0 + (low_speed_multiplier - 1.0) * low_speed) * (deficit / error_scale).square()


def yaw_pos_hipx_deviation_l2(env, asset_cfg: SceneEntityCfg, std: float = 0.15, free_angle: float = 0.0) -> torch.Tensor:
    """Soft cost for HipX abduction in both modes, around the nominal joint pose."""
    if std <= 0.0 or free_angle < 0.0:
        raise ValueError("HipX std must be positive and free_angle nonnegative.")
    data = env.scene[asset_cfg.name].data
    error = data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[:, asset_cfg.joint_ids]
    _accumulate(env, "_yaw_pos_hipx_abs_error", error.abs().mean(dim=1),
                torch.ones(error.shape[0], dtype=torch.bool, device=error.device))
    return ((error.abs() - free_angle).clamp_min(0.) / std).square().mean(dim=1)


def yaw_pos_base_height_tracking(
    env, target_height: float, error_scale: float, command_name: str, deadband: float,
    active_target_height: float | None = None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Mode-dependent finite height target, preserving the existing curriculum height minimum."""
    _, active, _ = yaw_pos_masks(env, command_name, deadband)
    height = env.scene[asset_cfg.name].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    if error_scale <= 0.:
        raise ValueError("Height error_scale must be positive.")
    if not hasattr(env, "_yaw_base_height_min"):
        env._yaw_base_height_min = torch.full_like(height, torch.inf)
    env._yaw_base_height_min = torch.minimum(env._yaw_base_height_min, height)
    target = torch.full_like(height, target_height)
    if active_target_height is not None:
        target = torch.where(active, active_target_height, target)
    error = (height - target).abs() / error_scale
    return 1. - torch.where(error <= 1., .5 * error.square(), error - .5)


def differential_rolling_score(
    target: torch.Tensor, measured: torch.Tensor, rolling_residual: torch.Tensor,
    motor_fraction: torch.Tensor, scale: torch.Tensor, minimum_speed_fraction: float,
    motor_rolling_speed: torch.Tensor | None = None, motor_tracking_weight: float = 1.0,
) -> torch.Tensor:
    """Dense [0, 1] shaping with the worse wheel controlling each component.

    Pseudo-Huber costs grow linearly for large normalized errors. Combine
    tracking, slip and motor deficit in ONE reciprocal, rather than multiplying
    narrow kernels. This retains a tracking slope even with an inactive motor.
    The better wheel cannot compensate for the worse tracking error or for
    the less active motor. A perfect score requires both motors to participate.
    Certification is deliberately independent of this shaping function.
    """
    tracking_cost = (torch.sqrt(1. + ((measured - target) / scale).square()) - 1.).amax(dim=1)
    rolling_cost = (torch.sqrt(1. + (rolling_residual / scale).square()) - 1.).amax(dim=1)
    participation = (motor_fraction / minimum_speed_fraction).clamp(0., 1.).amin(dim=1)
    motor_cost = 0.
    if motor_rolling_speed is not None:
        motor_cost = motor_tracking_weight * (torch.sqrt(1. + ((motor_rolling_speed - target) / scale).square()) - 1.).amax(dim=1)
    return 1. / (1. + tracking_cost + rolling_cost + motor_cost + 2. * (1. - participation))


def yaw_pos_differential_rolling(
    env, support_asset_cfg: SceneEntityCfg, support_joint_cfg: SceneEntityCfg,
    support_sensor_cfg: SceneEntityCfg, command_name: str, deadband: float, wheel_radius: float,
    contact_threshold: float = 1.0, relative_std: float = 0.25, speed_std: float = 0.05,
    minimum_speed_fraction: float = 0.5, settle_time: float = 0.5,
    certification_speed_std: float | None = None, motor_tracking_weight: float = 0.0,
    support_parent_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Shape both-wheel rolling densely; retain the strict settled certificate."""
    if wheel_radius <= 0 or relative_std <= 0 or speed_std <= 0 or settle_time < 0:
        raise ValueError("Differential scales/radius must be positive and settle_time nonnegative.")
    if not 0 < minimum_speed_fraction <= 1:
        raise ValueError("minimum_speed_fraction must be in (0, 1].")
    command, active, _ = yaw_pos_masks(env, command_name, deadband)
    robot = env.scene[support_asset_cfg.name]
    data = robot.data
    target, measured, valid = differential_rolling_kinematics(
        data.body_quat_w[:, support_asset_cfg.body_ids],
        wheel_center_positions(robot, support_asset_cfg.body_ids),
        wheel_center_velocities(robot, support_asset_cfg.body_ids),
        _yaw_whole_body_com_xy(robot), command,
    )
    motor_speed = data.joint_vel[:, support_joint_cfg.joint_ids].abs()
    contacts = _yaw_wheel_contacts(env, support_sensor_cfg, contact_threshold)
    if target.shape[1] != 2 or motor_speed.shape != target.shape or contacts.shape != target.shape:
        raise ValueError("Differential reward requires two ordered support bodies, joints and sensors.")
    _, _, heading_valid = ground_heading_axes(data.root_quat_w)
    valid = valid & heading_valid[:, None] & (target.abs() > 1.0e-4)
    scale = (relative_std * target.abs()).clamp_min(speed_std)
    certificate_scale = (relative_std * target.abs()).clamp_min(
        speed_std if certification_speed_std is None else certification_speed_std
    )
    # Only magnitude participates: qdot sign depends on the joint convention.
    motor_fraction = wheel_radius * motor_speed / target.abs().clamp_min(1.0e-4)
    residual, _ = contacted_wheel_velocities(
        env, support_sensor_cfg, support_asset_cfg, support_joint_cfg, wheel_radius, contact_threshold
    )
    motor_rolling = None
    if motor_tracking_weight > 0.:
        if support_parent_cfg is None:
            raise ValueError("Signed motor tracking requires ordered parent shank bodies.")
        body_ids = support_asset_cfg.body_ids
        signs = robot._yaw_pos_wheel_motor_axis_signs[body_ids]
        axle = rotate_vector(data.body_quat_w[:, body_ids], target.new_tensor((0., 1., 0.)))
        parent_spin = (data.body_ang_vel_w[:, support_parent_cfg.body_ids] * axle).sum(dim=-1)
        motor_rolling = wheel_radius * (signs * data.joint_vel[:, support_joint_cfg.joint_ids] + parent_spin)
    score = differential_rolling_score(target, measured, residual[..., 0], motor_fraction,
                                       scale, minimum_speed_fraction, motor_rolling, motor_tracking_weight)
    score *= (valid & contacts).all(dim=1)
    passing = (
        valid & contacts & (measured * target > 0.)
        & (measured.abs() >= minimum_speed_fraction * target.abs())
        & (motor_fraction >= minimum_speed_fraction)
        & (residual[..., 0].abs() <= certificate_scale)
    ).all(dim=1)
    if not hasattr(env, "_yaw_pos_active_age"):
        env._yaw_pos_active_age = torch.zeros_like(command)
    env._yaw_pos_active_age[env.episode_length_buf <= 1] = 0.
    env._yaw_pos_active_age = torch.where(
        active, env._yaw_pos_active_age + env.step_dt, 0.
    )
    settled = active & (env._yaw_pos_active_age + 1.0e-6 >= settle_time)
    _accumulate(env, "_yaw_pos_differential_pass", passing.float(), settled)
    masses = robot.root_physx_view.get_masses().to(command.device)
    com_velocity = (data.body_com_lin_vel_w * masses[..., None]).sum(1)
    com_velocity /= masses.sum(1).clamp_min(1.0e-6)[:, None]
    metrics = {
        "differential_fl_target": target[:, 0], "differential_hr_target": target[:, 1],
        "differential_fl_measured": measured[:, 0], "differential_hr_measured": measured[:, 1],
        "differential_fl_error": (measured[:, 0] - target[:, 0]).abs(),
        "differential_hr_error": (measured[:, 1] - target[:, 1]).abs(),
        "differential_both_contact": contacts.all(1).float(),
        "differential_one_motor_stopped": (motor_fraction < minimum_speed_fraction).any(1).float(),
        "differential_valid": valid.all(1).float(),
        "active_com_planar_speed": torch.linalg.vector_norm(com_velocity[:, :2], dim=1),
    }
    if motor_rolling is not None:
        metrics.update(differential_fl_motor_error=(motor_rolling[:, 0] - target[:, 0]).abs(),
                       differential_hr_motor_error=(motor_rolling[:, 1] - target[:, 1]).abs())
    for name, value in metrics.items():
        _accumulate(env, f"_yaw_pos_{name}", value, active)
    env._yaw_pos_differential_metrics_current = {name: value.detach().clone() for name, value in metrics.items()}
    env._yaw_pos_differential_metrics_current["differential_pass"] = (settled & passing).detach().clone()
    return torch.where(active, score, 0.)


def yaw_pos_active_translation(
    env, command_name: str, deadband: float, speed_scale: float = 0.10, drift_scale: float = 0.05,
    settle_time: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Settled CoM speed/drift cost; acquisition remains free to recover its balance."""
    if min(speed_scale, drift_scale) <= 0. or settle_time < 0.:
        raise ValueError("Translation scales must be positive and settle_time nonnegative.")
    _, active, _ = yaw_pos_masks(env, command_name, deadband)
    robot = env.scene[asset_cfg.name]
    com = _yaw_whole_body_com_xy(robot)
    masses = robot.root_physx_view.get_masses().to(com.device)
    velocity = (robot.data.body_com_lin_vel_w * masses[..., None]).sum(1) / masses.sum(1).clamp_min(1.e-6)[:, None]
    speed = torch.linalg.vector_norm(velocity[:, :2], dim=-1)
    age = getattr(env, "_yaw_pos_active_age", torch.zeros_like(speed))
    settled = active & (age + 1.e-6 >= settle_time)
    if not hasattr(env, "_yaw_pos_active_anchor"):
        env._yaw_pos_active_anchor = torch.zeros_like(com)
        env._yaw_pos_active_anchored = torch.zeros_like(active)
    reset = (~active) | (env.episode_length_buf <= 1)
    env._yaw_pos_active_anchored[reset] = False
    latch = settled & ~env._yaw_pos_active_anchored
    env._yaw_pos_active_anchor[latch] = com[latch]
    env._yaw_pos_active_anchored |= latch
    drift = torch.linalg.vector_norm(com - env._yaw_pos_active_anchor, dim=-1)
    _accumulate(env, "_yaw_pos_active_com_position_drift", drift, settled)
    env._yaw_pos_active_com_position_drift = torch.where(settled, drift, 0.).detach().clone()
    cost = torch.sqrt(1. + (speed / speed_scale).square()) - 1.
    cost += .5 * (torch.sqrt(1. + (drift / drift_scale).square()) - 1.)
    return torch.where(settled, cost, 0.)


def yaw_pos_neutral_velocity(
    env, command_name: str, deadband: float, std: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if std <= 0:
        raise ValueError("Neutral velocity std must be positive.")
    _, _, neutral = yaw_pos_masks(env, command_name, deadband)
    speed_squared = env.scene[asset_cfg.name].data.root_lin_vel_w[:, :2].square().sum(1)
    return torch.where(neutral, torch.exp(-speed_squared / std**2), 0.)


def yaw_pos_neutral_position(
    env, sensor_cfg: SceneEntityCfg, command_name: str, deadband: float,
    std: float = 0.05, contact_dwell: float = 0.2, contact_threshold: float = 1.0,
    speed_threshold: float = 0.03, drift_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Latch position after landing; never move the anchor to forgive subsequent drift."""
    if std <= 0 or contact_dwell < 0 or speed_threshold < 0 or drift_threshold < 0:
        raise ValueError("Invalid neutral position scales/thresholds.")
    command, _, neutral = yaw_pos_masks(env, command_name, deadband)
    data = env.scene[asset_cfg.name].data
    xy = data.root_pos_w[:, :2]
    if not hasattr(env, "_yaw_pos_neutral_anchor"):
        env._yaw_pos_neutral_anchor = torch.zeros_like(xy)
        env._yaw_pos_neutral_anchored = torch.zeros_like(neutral)
        env._yaw_pos_neutral_contact_age = torch.zeros_like(command)
    # Playback can disable CurriculumManager, so also detect episode resets here.
    reset = env.episode_length_buf <= 1
    env._yaw_pos_neutral_anchored[reset] = False
    env._yaw_pos_neutral_contact_age[reset] = 0.
    all_contact = _yaw_wheel_contacts(env, sensor_cfg, contact_threshold).all(1)
    env._yaw_pos_neutral_anchored &= neutral
    env._yaw_pos_neutral_contact_age = torch.where(
        neutral & all_contact, env._yaw_pos_neutral_contact_age + env.step_dt, 0.
    )
    acquire = neutral & ~env._yaw_pos_neutral_anchored & (env._yaw_pos_neutral_contact_age + 1.0e-6 >= contact_dwell) & all_contact
    env._yaw_pos_neutral_anchor[acquire] = xy[acquire]
    env._yaw_pos_neutral_anchored |= acquire
    held = neutral & env._yaw_pos_neutral_anchored
    drift = torch.linalg.vector_norm(xy - env._yaw_pos_neutral_anchor, dim=1)
    speed = torch.linalg.vector_norm(data.root_lin_vel_w[:, :2], dim=1)
    passing = all_contact & (speed <= speed_threshold) & (drift <= drift_threshold)
    _accumulate(env, "_yaw_pos_neutral_hold_pass", passing.float(), held)
    _accumulate(env, "_yaw_pos_neutral_position_drift", drift, held)
    env._yaw_pos_neutral_position_metrics_current = {
        "anchored": held.detach().clone(), "drift": drift.detach().clone(),
        "planar_speed": speed.detach().clone(), "passed": (held & passing).detach().clone(),
        "anchor_x": env._yaw_pos_neutral_anchor[:, 0].detach().clone(),
        "anchor_y": env._yaw_pos_neutral_anchor[:, 1].detach().clone(),
        "error_x": (xy[:, 0] - env._yaw_pos_neutral_anchor[:, 0]).detach().clone(),
        "error_y": (xy[:, 1] - env._yaw_pos_neutral_anchor[:, 1]).detach().clone(),
        "contact_age": env._yaw_pos_neutral_contact_age.detach().clone(),
    }
    return torch.where(held, torch.exp(-(drift / std).square()), 0.)


def yaw_pos_com_support(env, asset_cfg: SceneEntityCfg, std: float, command_name: str, deadband: float) -> torch.Tensor:
    _, active, _ = yaw_pos_masks(env, command_name, deadband)
    distance, _, _ = _yaw_support_geometry(env, asset_cfg)
    return torch.where(active, 1.0 / (1.0 + distance / std), 0.0)


def _yaw_pos_heading_support_geometry(
    env, asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """FL/HR x offsets from whole-body CoM and line alignment in the yaw-only heading frame."""
    robot = env.scene[asset_cfg.name]
    support_xy = wheel_center_positions(robot, asset_cfg.body_ids)[..., :2]
    if support_xy.shape[1] != 2:
        raise ValueError("POS heading geometry requires ordered FL and HR support wheels.")

    # Isaac Lab stores quaternions as (w, x, y, z). Ignore roll/pitch so the
    # diagnostic measures horizontal support geometry relative to body heading.
    w, x, y, z = robot.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    heading_x = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    heading_y = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)

    offsets = support_xy - _yaw_whole_body_com_xy(robot).unsqueeze(1)
    support_x = (offsets * heading_x.unsqueeze(1)).sum(dim=-1)
    support_x_rms = torch.linalg.vector_norm(support_x, dim=1) / 2.0**0.5
    support_line = support_xy[:, 1] - support_xy[:, 0]
    alignment = torch.abs((support_line * heading_y).sum(dim=-1)) / torch.linalg.vector_norm(
        support_line, dim=-1
    ).clamp_min(1.0e-8)
    alignment = alignment.clamp(max=1.0)
    return support_x[:, 0], support_x[:, 1], support_x_rms, alignment


def yaw_pos_heading_support(
    env, asset_cfg: SceneEntityCfg, scale: float, command_name: str, deadband: float,
) -> torch.Tensor:
    """Reward both support wheels approaching the body-y line through CoM in POS mode."""
    if scale <= 0.0:
        raise ValueError("scale must be positive.")
    _, active, _ = yaw_pos_masks(env, command_name, deadband)
    fl_x, hr_x, support_x_rms, alignment = _yaw_pos_heading_support_geometry(env, asset_cfg)
    for name, value in (
        ("fl_x_from_com", fl_x),
        ("hr_x_from_com", hr_x),
        ("support_x_rms", support_x_rms),
        ("support_line_body_y_alignment", alignment),
    ):
        _accumulate(env, f"_yaw_pos_heading_{name}", value, active)
    env._yaw_pos_heading_metrics_current = torch.stack((fl_x, hr_x, support_x_rms, alignment), dim=-1).detach()
    env._yaw_pos_heading_active_current = active.clone()
    # Keep a constant gradient through the observed 0.20–0.25 m plateau.
    score = 1.0 - support_x_rms / scale
    return torch.where(active, score, 0.0)


def _yaw_pos_support_line_geometry(
    env, asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ground-heading-X errors and heading-Y separation about whole-body CoM.

    Heading axes are horizontal, so body roll/pitch cannot use the vertical
    CoM-to-wheel offset to improve alignment. Do not use Euclidean span.
    Wheel order is FL, HR in the POS configuration.
    """
    robot = env.scene[asset_cfg.name]
    support_positions = wheel_center_positions(robot, asset_cfg.body_ids)
    if support_positions.shape[1] != 2:
        raise ValueError("POS support line geometry requires exactly two ordered support wheels.")
    masses = robot.root_physx_view.get_masses().to(robot.device)
    com = (robot.data.body_com_pos_w * masses.unsqueeze(-1)).sum(dim=1) / masses.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    heading_x, heading_y, _ = ground_heading_axes(robot.data.root_quat_w)
    offsets = support_positions - com.unsqueeze(1)
    line_errors = (offsets * heading_x.unsqueeze(1)).sum(dim=-1).abs()
    support_y = (offsets * heading_y.unsqueeze(1)).sum(dim=-1)
    separation = (support_y[:, 0] - support_y[:, 1]).abs()
    return line_errors[:, 0], line_errors[:, 1], separation


def yaw_pos_support_line(
    env, asset_cfg: SceneEntityCfg, sigma: float, command_name: str, deadband: float,
) -> torch.Tensor:
    """Reward the worse distance to the horizontal heading-Y line through CoM."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    _, active, _ = yaw_pos_masks(env, command_name, deadband)
    error1, error2, separation = _yaw_pos_support_line_geometry(env, asset_cfg)
    error = torch.maximum(error1, error2)
    for name, value in (
        ("support_line_error", error), ("wheel1_line_error", error1),
        ("wheel2_line_error", error2), ("support_y_separation", separation),
    ):
        _accumulate(env, f"_yaw_pos_{name}", value, active)
    env._yaw_pos_geometry_metrics_current = {
        "support_line_error": error.detach().clone(),
        "wheel1_line_error": error1.detach().clone(),
        "wheel2_line_error": error2.detach().clone(),
        "support_y_separation": separation.detach().clone(),
    }
    _, _, heading_valid = ground_heading_axes(env.scene[asset_cfg.name].data.root_quat_w)
    return torch.where(active & heading_valid, torch.exp(-(error / sigma).square()), 0.0)


def yaw_pos_support_y_collapse_l2(
    env, asset_cfg: SceneEntityCfg, minimum_separation: float, command_name: str, deadband: float,
    separation_scale: float = 0.05,
) -> torch.Tensor:
    """Normalized squared heading-Y minimum separation cost, with no upper bound."""
    if minimum_separation < 0.0:
        raise ValueError("minimum_separation must be nonnegative.")
    if separation_scale <= 0.0:
        raise ValueError("separation_scale must be positive.")
    _, active, _ = yaw_pos_masks(env, command_name, deadband)
    _, _, separation = _yaw_pos_support_line_geometry(env, asset_cfg)
    return torch.where(active, (torch.relu(minimum_separation - separation) / separation_scale).square(), 0.0)


def yaw_pos_body_angular_xy_l2(
    env, command_name: str, deadband: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """POS-only world-X/Y angular velocity cost; no command relief."""
    _, active, _ = yaw_pos_masks(env, command_name, deadband)
    cost = env.scene[asset_cfg.name].data.root_ang_vel_w[:, :2].square().sum(dim=1)
    _accumulate(env, "_yaw_pos_body_omega_xy_squared", cost, active)
    return torch.where(active, cost, 0.)


def yaw_pos_rolling_wheel_slip(
    env, sensor_cfg: SceneEntityCfg, body_asset_cfg: SceneEntityCfg, joint_asset_cfg: SceneEntityCfg,
    wheel_radius: float, command_name: str, yaw_reference: float, deadband: float, threshold: float = 1.0,
    minimum_scale: float = 0.0,
) -> torch.Tensor:
    """Phase-invariant contact slip with an optional floor on command relief."""
    velocities, in_contact = contacted_wheel_velocities(
        env, sensor_cfg, body_asset_cfg, joint_asset_cfg, wheel_radius, threshold
    )
    rolling = velocities[..., 0]
    penalty = (in_contact * rolling.square()).sum(dim=1)
    _, active, _ = yaw_pos_masks(env, command_name, deadband)
    contact_count = in_contact.sum(dim=1)
    mean_error = (in_contact * rolling.abs()).sum(dim=1) / contact_count.clamp_min(1)
    _accumulate(env, "_yaw_pos_rolling_error", mean_error, active & (contact_count > 0))
    return _yaw_command_penalty_scale(env, command_name, yaw_reference, minimum_scale) * penalty


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
    neutral_std: float | None = None,
    support_asset_cfg: SceneEntityCfg | None = None,
    support_joint_cfg: SceneEntityCfg | None = None,
    tracking_relative_std: float | None = None, tracking_min_std: float = 0.04,
) -> torch.Tensor:
    command, active, neutral = yaw_pos_masks(env, command_name, deadband)
    robot = env.scene[asset_cfg.name]
    yaw_rate, heading_valid = ground_heading_yaw_rate(robot.data.root_quat_w, robot.data.root_ang_vel_w)
    support = yaw_support_contact(env, support_sensor_cfg, contact_threshold)
    lift = _yaw_lift_progress(env, lifted_asset_cfg, wheel_radius, target_clearance).mean(dim=1)
    clearance_weight = clearance_gate_floor + (1.0 - clearance_gate_floor) * lift
    if std <= 0. or tracking_min_std <= 0. or (tracking_relative_std is not None and tracking_relative_std <= 0.):
        raise ValueError("Yaw tracking std values must be positive.")
    reward_std = std if tracking_relative_std is None else (command.abs() * tracking_relative_std).clamp_min(tracking_min_std)
    tracking = torch.exp(-(yaw_rate - command).square() / reward_std**2) * heading_valid
    active_score = support * clearance_weight * tracking
    # The existing promotion score retains its fixed std; only PPO shaping changes.
    certification_score = support * clearance_weight * torch.exp(-(yaw_rate - command).square() / std**2) * heading_valid
    neutral_score = torch.exp(-yaw_rate.square() / (std if neutral_std is None else neutral_std) ** 2) * heading_valid

    if support_asset_cfg is not None and support_joint_cfg is not None:
        motion = yaw_pos_motion_telemetry(robot, support_asset_cfg.body_ids, support_joint_cfg.joint_ids)
        for name, value in motion.items():
            _accumulate(env, f"_yaw_pos_{name}", value, active)
        # Freeze the reward's pre-reset state for per-step rollout traces.
        env._yaw_pos_motion_metrics_current = {name: value.detach().clone() for name, value in motion.items()}

    _accumulate(env, "_yaw_support_score", support, active)
    _accumulate(env, "_yaw_gate_open", support, active)
    _accumulate(env, "_yaw_active_yaw_score", certification_score, active)
    _accumulate(env, "_yaw_pos_active_yaw_tracking_error", (yaw_rate - command).abs(), active)
    env._yaw_pos_active_yaw_tracking_error = (yaw_rate - command).abs().detach().clone()
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
    # Optional recorder hook: capture physics state before ManagerBasedRLEnv's
    # automatic reset. Training does not install this callback.
    audit_callback = getattr(env, "_yaw_pos_rollout_audit_callback", None)
    if audit_callback is not None:
        audit_callback(env)
    return torch.where(active, active_score, torch.where(neutral, neutral_score, 0.0))


def yaw_pos_lifted_wheel_spin_l2(
    env, asset_cfg: SceneEntityCfg, command_name: str, deadband: float, yaw_reference: float,
) -> torch.Tensor:
    _, active, _ = yaw_pos_masks(env, command_name, deadband)
    robot = env.scene[asset_cfg.name]
    penalty = robot.data.joint_vel[:, asset_cfg.joint_ids].square().sum(dim=1)
    scale = _yaw_command_penalty_scale(env, command_name, yaw_reference)
    return torch.where(active, scale * penalty, 0.0)
