# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Exponential moving average of the wheel torque magnitude (thermal proxy)."""

from __future__ import annotations

import torch

from ..config.wheeled.vqr.physical_params import VQR_PHYSICS


class ThermalTracker:
    """EMA of |tau_wheel| with the 2 s continuous-duty time constant.

    Section 2: EMA|tau_w| must stay below 3 Nm for the balance pose to be
    sustainable indefinitely; the same signal feeds the L0 recovery trigger.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str = "cpu",
        time_constant: float | None = None,
        dt: float | None = None,
    ):
        self._dt = VQR_PHYSICS.control_dt if dt is None else dt
        tau = VQR_PHYSICS.thermal_time_constant if time_constant is None else time_constant
        self._alpha = self._dt / (tau + self._dt)
        self._ema = torch.zeros(num_envs, 4, device=device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._ema.zero_()
        else:
            self._ema[env_ids] = 0.0

    def update(self, wheel_torques: torch.Tensor) -> torch.Tensor:
        """Push the latest |tau_w| per wheel and return the EMA value."""
        self._ema = (1.0 - self._alpha) * self._ema + self._alpha * wheel_torques.abs()
        return self._ema

    @property
    def value(self) -> torch.Tensor:
        """Current EMA|tau_w| with shape ``(N, 4)``."""
        return self._ema
