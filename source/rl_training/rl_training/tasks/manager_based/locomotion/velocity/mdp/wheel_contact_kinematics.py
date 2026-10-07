"""Wheel contact kinematics independent of the rotating wheel phase."""

from __future__ import annotations

import torch


def rotate_vector(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by unit Isaac Lab quaternions in (w, x, y, z) order."""
    vector = vector.expand_as(quaternion[..., 1:])
    twice_cross = 2.0 * torch.linalg.cross(quaternion[..., 1:], vector, dim=-1)
    return vector + quaternion[..., :1] * twice_cross + torch.linalg.cross(
        quaternion[..., 1:], twice_cross, dim=-1
    )


def wheel_center_positions(asset, body_ids) -> torch.Tensor:
    """Collision-center positions; legacy assets without POS geometry use link origins."""
    positions = asset.data.body_pos_w[:, body_ids]
    offsets = getattr(asset, "_yaw_pos_wheel_center_offsets", None)
    if offsets is None:
        return positions
    return positions + rotate_vector(asset.data.body_quat_w[:, body_ids], offsets[body_ids])


def wheel_center_velocities(asset, body_ids) -> torch.Tensor:
    velocities = asset.data.body_link_lin_vel_w[:, body_ids]
    offsets = getattr(asset, "_yaw_pos_wheel_center_offsets", None)
    if offsets is None:
        return velocities
    offset_w = rotate_vector(asset.data.body_quat_w[:, body_ids], offsets[body_ids])
    return velocities + torch.linalg.cross(asset.data.body_ang_vel_w[:, body_ids], offset_w, dim=-1)


def wheel_ground_clearance(asset, body_ids, wheel_radius: float, ground_height) -> torch.Tensor:
    """Vertical tire extent includes axle tilt and finite cylinder width for POS assets."""
    centers = wheel_center_positions(asset, body_ids)
    widths = getattr(asset, "_yaw_pos_wheel_half_widths", None)
    extent = wheel_radius
    if widths is not None:
        axle = rotate_vector(asset.data.body_quat_w[:, body_ids], centers.new_tensor((0., 1., 0.)))
        axial_z = axle[..., 2].abs().clamp_max(1.)
        extent = wheel_radius * (1. - axial_z.square()).clamp_min(0.).sqrt() + widths[body_ids] * axial_z
    return centers[..., 2] - ground_height - extent


def wheel_contact_velocities(
    orientation_w: torch.Tensor,
    center_velocity_w: torch.Tensor,
    angular_velocity_w: torch.Tensor,
    wheel_radius: float,
    ground_normal_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return [..., 2] rolling residual and lateral center velocity in m/s.

    The VQR axle is local +Y, which does not rotate with wheel phase. With
    a = axle_world and t = normalize(a x ground_normal), the rolling residual
    is v_center dot t - radius * (omega_world dot a). This equals the rolling
    component of contact-point velocity for a circular wheel. The measured
    rigid-body angular velocity includes wheel spin and parent-leg motion;
    no joint-axis sign assumption or rotating local-X projection is needed.

    An axle parallel to the ground normal has no defined rolling direction;
    its rolling channel is zero rather than dividing by a vanishing norm.
    """
    if wheel_radius <= 0.0:
        raise ValueError("wheel_radius must be positive.")
    local_axle = orientation_w.new_tensor((0.0, 1.0, 0.0))
    axle = rotate_vector(orientation_w, local_axle)
    axle = axle / torch.linalg.vector_norm(axle, dim=-1, keepdim=True).clamp_min(1.0e-8)
    normal = orientation_w.new_tensor((0.0, 0.0, 1.0)) if ground_normal_w is None else ground_normal_w
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1.0e-8)
    tangent = torch.linalg.cross(axle, normal.expand_as(axle), dim=-1)
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    tangent = tangent / tangent_norm.clamp_min(1.0e-8)
    rolling = (center_velocity_w * tangent).sum(dim=-1) - wheel_radius * (
        angular_velocity_w * axle
    ).sum(dim=-1)
    rolling = torch.where(tangent_norm.squeeze(-1) > 1.0e-6, rolling, 0.0)
    lateral = (center_velocity_w * axle).sum(dim=-1)
    return torch.stack((rolling, lateral), dim=-1)


def contacted_wheel_velocities(
    env, sensor_cfg, body_asset_cfg, joint_asset_cfg, wheel_radius: float, threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared reward/critic path, returning unmasked velocities and contact flags.

    Keep the joint selection for configuration compatibility and order-length
    validation; actual wheel angular velocity is read from the rigid body.
    Start from link velocity rather than CoM velocity and add the POS collider
    offset contribution (Isaac Lab's body_lin_vel_w aliases CoM velocity).
    """
    if len({len(body_asset_cfg.body_ids), len(joint_asset_cfg.joint_ids), len(sensor_cfg.body_ids)}) != 1:
        raise ValueError("Wheel body, joint, and contact-sensor selections must have equal length.")
    asset = env.scene[body_asset_cfg.name]
    data = asset.data
    sensor = env.scene.sensors[sensor_cfg.name]
    velocities = wheel_contact_velocities(
        data.body_quat_w[:, body_asset_cfg.body_ids],
        wheel_center_velocities(asset, body_asset_cfg.body_ids),
        data.body_ang_vel_w[:, body_asset_cfg.body_ids],
        wheel_radius,
    )
    in_contact = torch.linalg.vector_norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1) > threshold
    return velocities, in_contact


def rolling_lateral_contact_velocity(
    env, sensor_cfg, body_asset_cfg, joint_asset_cfg, wheel_radius: float, threshold: float = 1.0,
) -> torch.Tensor:
    """POS critic channels [rolling, lateral] per wheel, zero off contact."""
    velocities, in_contact = contacted_wheel_velocities(
        env, sensor_cfg, body_asset_cfg, joint_asset_cfg, wheel_radius, threshold
    )
    return (velocities * in_contact.unsqueeze(-1)).reshape(env.num_envs, -1)
