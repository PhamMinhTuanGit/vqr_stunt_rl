# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent math shared by the four-mode pivot task.

Keeping these operations free of Isaac Lab types makes the reward and
termination definitions use the same formulas and lets their boundary cases
be tested without launching the simulator.
"""

from __future__ import annotations

import torch


GROUND = 0
REAR_UP = 1
BALANCE = 2
LAND = 3


def smoothstep01(value: torch.Tensor) -> torch.Tensor:
    """C1-continuous interpolation on ``[0, 1]``."""
    value = value.clamp(0.0, 1.0)
    return value.square() * (3.0 - 2.0 * value)


def pose_phase_from_time(
    episode_time: torch.Tensor,
    ground_duration: float,
    rear_up_duration: float,
    balance_duration: float,
    land_duration: float,
) -> torch.Tensor:
    """Return standing=0 / balance=1 phase for the complete mode schedule.

    The phase is zero throughout GROUND, rises smoothly in REAR_UP, stays one
    in BALANCE, and falls smoothly in LAND.  Its value therefore agrees on
    both sides of every mode boundary.
    """
    rear_start = ground_duration
    balance_start = rear_start + rear_up_duration
    land_start = balance_start + balance_duration

    rear_phase = smoothstep01((episode_time - rear_start) / rear_up_duration)
    land_phase = 1.0 - smoothstep01((episode_time - land_start) / land_duration)

    phase = torch.zeros_like(episode_time)
    phase = torch.where(episode_time >= rear_start, rear_phase, phase)
    phase = torch.where(episode_time >= balance_start, torch.ones_like(phase), phase)
    phase = torch.where(episode_time >= land_start, land_phase, phase)
    return phase.clamp(0.0, 1.0)


def blend_joint_prior(
    standing: torch.Tensor,
    balance: torch.Tensor,
    pose_phase: torch.Tensor,
) -> torch.Tensor:
    """Blend two joint targets using the already scheduled smooth phase."""
    phase = pose_phase.clamp(0.0, 1.0).unsqueeze(-1)
    return standing + phase * (balance - standing)


def mode_planar_drift(
    mode: torch.Tensor,
    body_center_xy: torch.Tensor,
    body_anchor_xy: torch.Tensor,
    support_midpoint_xy: torch.Tensor,
    support_anchor_xy: torch.Tensor,
) -> torch.Tensor:
    """GROUND/LAND body-center drift; REAR_UP/BALANCE HL-HR drift."""
    body_drift = torch.linalg.vector_norm(body_center_xy - body_anchor_xy, dim=-1)
    support_drift = torch.linalg.vector_norm(
        support_midpoint_xy - support_anchor_xy, dim=-1
    )
    rear_supported_mode = (mode == REAR_UP) | (mode == BALANCE)
    return torch.where(rear_supported_mode, support_drift, body_drift)


def mode_contact_score(mode: torch.Tensor, wheel_contact: torch.Tensor) -> torch.Tensor:
    """Strict support score for ordered ``[FL, FR, HL, HR]`` contacts."""
    if wheel_contact.shape[-1] != 4:
        raise ValueError("wheel_contact must be ordered [FL, FR, HL, HR]")
    contact = wheel_contact.to(dtype=torch.float32)
    four_wheel = contact.prod(dim=-1)
    rear_support = contact[..., 2:].prod(dim=-1)
    rear_supported_mode = (mode == REAR_UP) | (mode == BALANCE)
    return torch.where(rear_supported_mode, rear_support, four_wheel)


def supported_front_lift_score(wheel_contact: torch.Tensor) -> torch.Tensor:
    """Front-air score that is zero unless both HL and HR support the robot."""
    if wheel_contact.shape[-1] != 4:
        raise ValueError("wheel_contact must be ordered [FL, FR, HL, HR]")
    contact = wheel_contact.to(dtype=torch.float32)
    rear_support = contact[..., 2:].prod(dim=-1)
    front_air = 1.0 - contact[..., :2].mean(dim=-1)
    return rear_support * front_air


def force_budget_excess(normal_force: torch.Tensor, budget: float) -> torch.Tensor:
    """Normalized peak force excess; ordinary support below budget is free."""
    peak = normal_force.abs().amax(dim=-1)
    return torch.relu(peak - budget) / budget


def balance_coordinates(
    com_pos_xy: torch.Tensor,
    com_vel_xy: torch.Tensor,
    support_pos_xy: torch.Tensor,
    support_vel_xy: torch.Tensor,
    forward_w: torch.Tensor,
    angular_velocity_w: torch.Tensor,
    omega0: float,
) -> torch.Tensor:
    """Return ``[zeta, zeta_dot, xi]`` about the moving HL-HR midpoint.

    ``zeta_dot`` differentiates both the relative CoM position and the rotating
    heading axis.  ``omega0`` is the audited rigid-body value, not a per-frame
    point-mass estimate.
    """
    forward_xy = forward_w[..., :2]
    forward_xy_norm = torch.linalg.vector_norm(
        forward_xy, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    heading_xy = forward_xy / forward_xy_norm
    forward_dot_w = torch.linalg.cross(angular_velocity_w, forward_w, dim=-1)
    forward_dot_xy = forward_dot_w[..., :2]
    # Derivative of normalized planar projection u=f_xy/||f_xy||.
    heading_dot = (
        forward_dot_xy
        - heading_xy * torch.sum(heading_xy * forward_dot_xy, dim=-1, keepdim=True)
    ) / forward_xy_norm
    rel_pos = com_pos_xy - support_pos_xy
    rel_vel = com_vel_xy - support_vel_xy
    zeta = torch.sum(rel_pos * heading_xy, dim=-1)
    zeta_dot = torch.sum(rel_vel * heading_xy + rel_pos * heading_dot, dim=-1)
    xi = zeta + zeta_dot / omega0
    return torch.stack((zeta, zeta_dot, xi), dim=-1)
