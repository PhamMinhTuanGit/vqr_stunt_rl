# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""EstimatedBalanceState: deployable balance estimate (IMU + kin + contact).

Enabled from curriculum stage S4 onward; before that the cache is fed by the
simulation-ground-truth :class:`SimBalanceState`.
"""

from __future__ import annotations

import torch

from ..config.wheeled.vqr.physical_params import VQR_PHYSICS
from .balance_provider import BalanceState


class EstimatedBalanceState:
    """Pitch from projected gravity, zeta from pendulum geometry, filtered rate."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str = "cpu",
        dt: float | None = None,
        filter_tau: float = 0.06,
    ):
        self._dt = VQR_PHYSICS.control_dt if dt is None else dt
        self._alpha = self._dt / (filter_tau + self._dt)
        self._zeta_prev = torch.zeros(num_envs, device=device)
        self._zeta_dot = torch.zeros(num_envs, device=device)
        self._initialised = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._state = BalanceState(
            zeta=torch.zeros(num_envs, device=device),
            zeta_dot=torch.zeros(num_envs, device=device),
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._zeta_prev.zero_()
            self._zeta_dot.zero_()
            self._initialised[:] = False
            self._state.zeta.zero_()
            self._state.zeta_dot.zero_()
        else:
            self._zeta_prev[env_ids] = 0.0
            self._zeta_dot[env_ids] = 0.0
            self._initialised[env_ids] = False
            self._state.zeta[env_ids] = 0.0
            self._state.zeta_dot[env_ids] = 0.0

    def update(self, projected_gravity_b: torch.Tensor, com_offset_body: torch.Tensor) -> BalanceState:
        """Estimate (zeta, zeta_dot) from body-frame gravity and CoM offset."""
        # Pitch: forward tilt moves gravity toward +x in the body frame.
        pitch = torch.atan2(projected_gravity_b[:, 0], -projected_gravity_b[:, 2])
        zeta = VQR_PHYSICS.com_height_balance * torch.sin(pitch) + com_offset_body[:, 0]

        rate = (zeta - self._zeta_prev) / self._dt
        zeta_dot = self._zeta_dot + self._alpha * (rate - self._zeta_dot)
        # No valid finite difference on the first update after a reset.
        zeta_dot = torch.where(self._initialised, zeta_dot, torch.zeros_like(zeta_dot))

        self._zeta_prev = zeta
        self._zeta_dot = zeta_dot
        self._initialised[:] = True
        self._state = BalanceState(zeta=zeta, zeta_dot=zeta_dot)
        return self._state

    def balance_state(self) -> BalanceState:
        return self._state
