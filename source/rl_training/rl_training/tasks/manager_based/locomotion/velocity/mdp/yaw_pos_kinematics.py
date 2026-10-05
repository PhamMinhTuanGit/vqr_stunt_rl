"""Ground-heading kinematics and wheel telemetry for the POS task.

Quaternions use Isaac Lab's (w, x, y, z) convention. Angular velocities
are in world coordinates. These helpers do not depend on Isaac Sim.
"""

import torch

from .wheel_contact_kinematics import rotate_vector


POS_MOTION_METRICS = (
    "true_heading_rate", "heading_rate_valid", "world_yaw_rate",
    "world_angular_x", "world_angular_y", "abs_world_angular_x", "abs_world_angular_y",
    "roll_rate", "pitch_rate", "abs_roll_rate", "abs_pitch_rate",
    "support_fl_motor_speed", "support_hr_motor_speed",
    "support_fl_ground_rolling_speed", "support_hr_ground_rolling_speed",
    "support_fl_rolling_direction_valid", "support_hr_rolling_direction_valid",
)

POS_DIFFERENTIAL_METRICS = (
    "differential_fl_target", "differential_hr_target",
    "differential_fl_measured", "differential_hr_measured",
    "differential_fl_error", "differential_hr_error",
    "differential_both_contact", "differential_one_motor_stopped",
    "differential_valid", "active_com_planar_speed",
    "differential_pass", "neutral_hold_pass", "neutral_position_drift",
    "differential_fl_signed_ratio", "differential_hr_signed_ratio",
    "differential_fl_wrong_sign_pct", "differential_hr_wrong_sign_pct", "differential_any_wrong_sign_pct",
    "differential_fl_overspeed", "differential_hr_overspeed",
    "differential_fl_overspeed_pct", "differential_hr_overspeed_pct",
)


def differential_rolling_kinematics(
    wheel_quaternions: torch.Tensor, wheel_positions: torch.Tensor,
    wheel_center_velocities: torch.Tensor, com_xy: torch.Tensor, yaw_command: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project desired ground-plane rotation and actual center motion onto each axle's tangent.

    Local +Y is the physical wheel axle. Reversing its convention reverses
    both signed projections, leaving error/reward unchanged. No motor-axis
    sign or left/right speed convention is assumed.
    """
    axle = rotate_vector(wheel_quaternions, wheel_quaternions.new_tensor((0., 1., 0.)))
    normal = axle.new_tensor((0., 0., 1.)).expand_as(axle)
    tangent = torch.linalg.cross(axle, normal, dim=-1)
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1)
    tangent = tangent / tangent_norm.clamp_min(1.0e-6).unsqueeze(-1)
    offset_xy = wheel_positions[..., :2] - com_xy.unsqueeze(1)
    target_velocity = torch.stack(
        (-offset_xy[..., 1], offset_xy[..., 0], torch.zeros_like(offset_xy[..., 0])), dim=-1
    ) * yaw_command[:, None, None]
    target = (target_velocity * tangent).sum(dim=-1)
    measured = (wheel_center_velocities * tangent).sum(dim=-1)
    valid = (tangent_norm > 1.0e-6) & torch.isfinite(target) & torch.isfinite(measured)
    return torch.where(valid, target, 0.), torch.where(valid, measured, 0.), valid


def _heading_projection(quaternion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    body_x = rotate_vector(quaternion, quaternion.new_tensor((1., 0., 0.)))
    horizontal_squared = body_x[..., :2].square().sum(dim=-1)
    # A vertical body-X has no ground heading. Never certify its zero fallback.
    valid = horizontal_squared > 1.0e-8
    return body_x, horizontal_squared, valid


def ground_heading_axes(quaternion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return orthonormal horizontal heading-X/Y and a projection-valid mask."""
    body_x, horizontal_squared, valid = _heading_projection(quaternion)
    heading_x = torch.cat((body_x[..., :2], torch.zeros_like(body_x[..., :1])), dim=-1)
    heading_x = heading_x / horizontal_squared.sqrt().clamp_min(1.0e-4).unsqueeze(-1)
    heading_x = torch.where(valid.unsqueeze(-1), heading_x, heading_x.new_tensor((1., 0., 0.)))
    heading_y = torch.stack((-heading_x[..., 1], heading_x[..., 0], torch.zeros_like(horizontal_squared)), dim=-1)
    return heading_x, heading_y, valid


def ground_heading_yaw_rate(
    quaternion: torch.Tensor, angular_velocity_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derivative of atan2(body-X_y, body-X_x), including tilt coupling.

    For b = R(q) e_x, b_dot = omega_world x b, so heading_dot is
    (b_x*b_dot_y - b_y*b_dot_x) / (b_x**2 + b_y**2).
    This is the actual ground-heading rate, rather than body-Z or world-Z
    angular velocity. A singular projection returns zero with valid=False.
    """
    body_x, horizontal_squared, valid = _heading_projection(quaternion)
    derivative = torch.linalg.cross(angular_velocity_w, body_x, dim=-1)
    rate = (body_x[..., 0] * derivative[..., 1] - body_x[..., 1] * derivative[..., 0])
    rate = rate / horizontal_squared.clamp_min(1.0e-8)
    return torch.where(valid, rate, 0.), valid


def yaw_pos_motion_telemetry(robot, support_body_ids, support_joint_ids) -> dict[str, torch.Tensor]:
    """Motor qdot (rad/s), center rolling speed (m/s), and body rates (rad/s).

    Motor speed is the actual joint velocity, while ground rolling speed is
    wheel-link center velocity projected onto the phase-invariant tangent.
    Neither quantity alone proves which actuator supplied the mechanical work.
    Support body and joint IDs must both be ordered FL, HR.
    """
    data = robot.data
    quaternion, omega = data.root_quat_w, data.root_ang_vel_w
    heading_rate, valid = ground_heading_yaw_rate(quaternion, omega)
    conjugate = torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)
    omega_b = rotate_vector(conjugate, omega)
    w, x, y, z = quaternion.unbind(dim=-1)
    roll = torch.atan2(2. * (w * x + y * z), 1. - 2. * (x.square() + y.square()))
    sin_pitch = (2. * (w * y - z * x)).clamp(-1., 1.)
    roll_rate = omega_b[..., 0] + heading_rate * sin_pitch
    pitch_rate = omega_b[..., 1] * torch.cos(roll) - omega_b[..., 2] * torch.sin(roll)
    wheel_quaternions = data.body_quat_w[:, support_body_ids]
    motor_speeds = data.joint_vel[:, support_joint_ids]
    if wheel_quaternions.shape[1] != 2 or motor_speeds.shape[1] != 2:
        raise ValueError("POS telemetry requires exactly two support bodies and motor joints ordered FL, HR.")
    axle = rotate_vector(wheel_quaternions, wheel_quaternions.new_tensor((0., 1., 0.)))
    normal = axle.new_tensor((0., 0., 1.)).expand_as(axle)
    tangent = torch.linalg.cross(axle, normal, dim=-1)
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1)
    tangent = tangent / tangent_norm.clamp_min(1.0e-6).unsqueeze(-1)
    rolling_valid = tangent_norm > 1.0e-6
    ground_speeds = (data.body_link_lin_vel_w[:, support_body_ids] * tangent).sum(dim=-1)
    ground_speeds = torch.where(rolling_valid, ground_speeds, 0.)
    return {
        "true_heading_rate": heading_rate, "heading_rate_valid": valid.float(),
        "world_yaw_rate": omega[..., 2],
        "world_angular_x": omega[..., 0], "world_angular_y": omega[..., 1],
        "abs_world_angular_x": omega[..., 0].abs(), "abs_world_angular_y": omega[..., 1].abs(),
        "roll_rate": roll_rate, "pitch_rate": pitch_rate,
        "abs_roll_rate": roll_rate.abs(), "abs_pitch_rate": pitch_rate.abs(),
        "support_fl_motor_speed": motor_speeds[:, 0], "support_hr_motor_speed": motor_speeds[:, 1],
        "support_fl_ground_rolling_speed": ground_speeds[:, 0],
        "support_hr_ground_rolling_speed": ground_speeds[:, 1],
        "support_fl_rolling_direction_valid": rolling_valid[:, 0].float(),
        "support_hr_rolling_direction_valid": rolling_valid[:, 1].float(),
    }
