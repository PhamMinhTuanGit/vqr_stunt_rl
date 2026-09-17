# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""PivotStateCache: every shared quantity computed once per control step (I8).

The environment owns one instance and calls :meth:`update` in
``_pre_physics_step``; observation, reward, and termination terms only read
from the cache.  Nothing in this module imports Isaac Lab except via duck
typing: all inputs are plain torch tensors.
"""

from __future__ import annotations

import torch

from ..config.wheeled.vqr.physical_params import VQR_PHYSICS
from .balance_provider import SimBalanceState
from .hard_state_buffer import HardStateBuffer
from .thermal_tracker import ThermalTracker


class PivotStateCache:
    """Per-step cache of zeta/zeta_dot/xi, EMA torque, contacts, and modes."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str = "cpu",
        mu_hat: float = 1.0,
    ):
        self.physics = VQR_PHYSICS
        self.num_envs = num_envs
        self.device = torch.device(device)

        # -- balance signals (section 3) --------------------------------------
        self.mu_hat = torch.full((num_envs,), float(mu_hat), device=self.device)
        self.zeta = torch.zeros(num_envs, device=self.device)
        self.zeta_dot = torch.zeros(num_envs, device=self.device)
        self.xi = torch.zeros(num_envs, device=self.device)

        # -- dynamic setpoints -------------------------------------------------
        self.theta_star = torch.full(
            (num_envs,), self.physics.theta_star_0, device=self.device
        )
        self.omega_z = torch.zeros(num_envs, device=self.device)
        self.pitch = torch.zeros(num_envs, device=self.device)
        self.roll = torch.zeros(num_envs, device=self.device)

        # -- thermal / contact ---------------------------------------------------
        self.ema_wheel_torque = torch.zeros(num_envs, 4, device=self.device)
        self.wheel_contact = torch.zeros(num_envs, 4, device=self.device)

        # -- sub-trackers ----------------------------------------------------------
        self.thermal = ThermalTracker(num_envs, device=device)
        self.hard_states = HardStateBuffer(num_envs, device=device)
        self.balance = SimBalanceState(num_envs, device=device)

        # Pending reset bookkeeping handled by the environment.
        self._updated = False

    # -- derived safety thresholds -------------------------------------------
    @property
    def xi_safe(self) -> torch.Tensor:
        """Per-env safe capture-point radius xi_safe = 0.50 * mu_hat * h."""
        return self.physics.xi_safe_fraction * self.mu_hat * self.physics.com_height_balance

    @property
    def xi_max(self) -> torch.Tensor:
        """Per-env hard capture-point radius xi_max = 0.85 * mu_hat * h."""
        return self.physics.xi_max_fraction * self.mu_hat * self.physics.com_height_balance

    # -- core update -------------------------------------------------------------
    def update(
        self,
        com_pos_w: torch.Tensor,
        com_vel_w: torch.Tensor,
        rear_contact_pos_w: torch.Tensor,
        heading_w: torch.Tensor,
        proj_gravity_b: torch.Tensor,
        wheel_torques: torch.Tensor,
        wheel_contact_flags: torch.Tensor,
        omega_z: torch.Tensor | None = None,
        omega_z_command: torch.Tensor | None = None,
    ) -> None:
        """Refresh every cached quantity for the current control step.

        Args:
            com_pos_w: CoM position in world frame, ``(N, 3)``.
            com_vel_w: CoM velocity in world frame, ``(N, 3)``.
            rear_contact_pos_w: midpoint of the two rear wheel contacts, ``(N, 3)``.
            heading_w: horizontal unit heading vector, ``(N, 3)``.
            proj_gravity_b: gravity unit vector in the body frame, ``(N, 3)``.
            wheel_torques: applied wheel torques, ``(N, 4)``.
            wheel_contact_flags: binary wheel contact states, ``(N, 4)``.
            omega_z: measured base yaw rate; defaults to zero.
            omega_z_command: commanded yaw rate; also defaults to zero.
        """
        # Balance state (zeta, zeta_dot) and capture point xi (I1).
        self.balance.update(com_pos_w, com_vel_w, rear_contact_pos_w, heading_w)
        state = self.balance.balance_state()
        self.zeta = state.zeta
        self.zeta_dot = state.zeta_dot
        self.xi = self.zeta + self.zeta_dot / self.physics.omega0

        # Attitude from projected gravity (pitch forward positive).
        self.pitch = torch.atan2(proj_gravity_b[:, 1] * 0.0 + (-proj_gravity_b[:, 0]), -proj_gravity_b[:, 2])
        self.roll = torch.atan2(-proj_gravity_b[:, 1], -proj_gravity_b[:, 2])

        # Yaw rate and the spinning setpoint theta*(omega_z_command).
        self.omega_z = torch.zeros_like(self.zeta) if omega_z is None else omega_z
        cmd = self.omega_z if omega_z_command is None else omega_z_command
        self.theta_star = self.physics.theta_star_0 + self.physics.theta_star_spin_coeff * cmd.square()

        # Thermal proxy and contact flags.
        self.ema_wheel_torque = self.thermal.update(wheel_torques)
        self.wheel_contact = wheel_contact_flags

        self._updated = True

    # -- housekeeping ---------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.thermal.reset(env_ids)
        if env_ids is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_ids] = True
        for tensor in (self.zeta, self.zeta_dot, self.xi, self.pitch, self.roll, self.omega_z):
            tensor[mask] = 0.0
        self.ema_wheel_torque[mask] = 0.0
        self.wheel_contact[mask] =  0.0
        self.theta_star[mask] = self.physics.theta_star_0

    def require_updated(self) -> None:
        if not self._updated:
            raise RuntimeError(
                "PivotStateCache.update() must run in _pre_physics_step before MDP terms read it (I8)."
            )
