# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Safety gate logic shared between training and deployment (invariant I9).

Pure functions over plain tensors; deliberately NO Isaac Lab imports so the
same gate runs on the robot.  Mode ids: 0=GROUND, 1=REAR_UP, 2=BALANCE,
3=LAND; RECOVER is handled by the separate pi_recover stack.
"""

from __future__ import annotations

import torch

# Mode ids (shared with commands.py)
GROUND = 0
REAR_UP = 1
BALANCE = 2
LAND = 3
NUM_MODES = 4


class RateLimiter:
    """First-order slew-rate limiter on the commanded yaw rate."""

    def __init__(
        self,
        velocity_limit: float,
        dt: float,
        step: float | None = None,
        num_envs: int = 1,
        device: torch.device | str = "cpu",
    ):
        self.max_delta = velocity_limit * dt if step is None else step * dt
        self.state = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.state.zero_()
        else:
            self.state[env_ids] = 0.0

    def __call__(self, target: torch.Tensor) -> torch.Tensor:
        delta = (target - self.state).clamp(-self.max_delta, self.max_delta)
        self.state = self.state + delta
        return self.state.clone()


def xi_safe_radius(mu_hat: torch.Tensor, physics) -> torch.Tensor:
    """xi_safe = 0.50 * mu_hat * h (section 3)."""
    return physics.xi_safe_fraction * mu_hat * physics.com_height_balance


def xi_max_radius(mu_hat: torch.Tensor, physics) -> torch.Tensor:
    """xi_max = 0.85 * mu_hat * h (section 3)."""
    return physics.xi_max_fraction * mu_hat * physics.com_height_balance


def clamp_omega_z(
    omega_z_cmd: torch.Tensor,
    mode: torch.Tensor,
    ground_limit: float,
    balance_limit: float,
) -> torch.Tensor:
    """Clamp |omega_z*| per mode (invariant I10: +/-3 GROUND, +/-6 BALANCE)."""
    ground = torch.as_tensor(ground_limit, dtype=omega_z_cmd.dtype, device=omega_z_cmd.device)
    balance = torch.as_tensor(balance_limit, dtype=omega_z_cmd.dtype, device=omega_z_cmd.device)
    limit = torch.where(mode == GROUND, ground, balance)
    return omega_z_cmd.clamp(-limit, limit)


def anchor_disable_mask(xi: torch.Tensor, xi_safe: torch.Tensor) -> torch.Tensor:
    """I4: beyond xi_safe the anchor and omega_z tracking must BOTH turn off."""
    return xi.abs() > xi_safe


def backflip_excess(theta: torch.Tensor, theta_star: torch.Tensor) -> torch.Tensor:
    """I5: positive when pitch exceeds theta*(omega_z) by more than zero."""
    return torch.relu(theta - theta_star)


def mode_transition(
    mode: torch.Tensor,
    xi: torch.Tensor,
    xi_safe: torch.Tensor,
    xi_max: torch.Tensor,
    ema_wheel_torque: torch.Tensor,
    thermal_limit: float,
) -> torch.Tensor:
    """Recovery-ladder mode arbitration (section 8, tiers L0-L3).

    Returns the arbitrated mode per env; the environment applies it and may
    impose its own grace windows on top.

    L0: EMA|tau_w| > thermal_limit            -> demote to GROUND, extend legs
    L1: |xi| beyond xi_safe inside REAR_UP     -> back to GROUND
    L2: xi_safe < |xi| < xi_max in BALANCE     -> drive support point under CoM
    L3: xi > xi_max                            -> controlled forward topple (LAND)
    """
    abs_xi = xi.abs()
    thermal_breach = ema_wheel_torque > thermal_limit
    l1 = (mode == REAR_UP) & (abs_xi > xi_safe)
    l2 = (mode == BALANCE) & (abs_xi > xi_safe) & (abs_xi < xi_max)
    l3 = xi > xi_max

    new_mode = mode.clone()
    new_mode = torch.where(l1, torch.full_like(new_mode, GROUND), new_mode)
    new_mode = torch.where(l2 & (~l3), torch.full_like(new_mode, REAR_UP), new_mode)
    new_mode = torch.where(l3, torch.full_like(new_mode, LAND), new_mode)
    new_mode = torch.where(thermal_breach & (new_mode != GROUND), torch.full_like(new_mode, GROUND), new_mode)
    return new_mode
