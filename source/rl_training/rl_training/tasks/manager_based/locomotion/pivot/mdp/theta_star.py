# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""theta*(omega_z) feed-forward (invariant I1): never track raw pitch.

theta* = 32.58 deg + 0.0931 deg * omega_z^2, all constants from the SSOT
``physical_params.py``; this module only assembles tensor expressions.
"""

from __future__ import annotations

import torch

from ..config.wheeled.vqr.physical_params import VQR_PHYSICS


def theta_star(omega_z: torch.Tensor) -> torch.Tensor:
    """theta*(omega_z) = theta*_0 + c * omega_z^2 [rad]."""
    return (
        VQR_PHYSICS.theta_star_0
        + VQR_PHYSICS.theta_star_spin_coeff * omega_z.square()
    )


def theta_star_residual(current_pitch: torch.Tensor, omega_z: torch.Tensor) -> torch.Tensor:
    """Pitch error relative to the spin-adjusted setpoint (rad)."""
    return current_pitch - theta_star(omega_z)


def delta_theta_command(command_delta_theta: torch.Tensor) -> torch.Tensor:
    """Clamp the commanded delta-theta to the +/-8 deg window (section 7)."""
    return command_delta_theta.clamp(
        -VQR_PHYSICS.delta_theta_command_limit,
        VQR_PHYSICS.delta_theta_command_limit,
    )


def tilted_setpoint(omega_z: torch.Tensor, delta_theta_cmd: torch.Tensor) -> torch.Tensor:
    """theta_set = theta*(omega_z) + delta, delta in +/-8 deg, biased forward."""
    bias = VQR_PHYSICS.pitch_setpoint_forward_bias
    return theta_star(omega_z) + delta_theta_command(delta_theta_cmd) + bias
