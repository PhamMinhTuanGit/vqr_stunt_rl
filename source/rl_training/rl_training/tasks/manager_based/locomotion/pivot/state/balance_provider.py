# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Balance-state sources: simulation ground truth and the shared Protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass
class BalanceState:
    """Longitudinal pendulum state relative to the rear contact line."""

    zeta: torch.Tensor  # CoM offset ahead of the rear contact midpoint [m]
    zeta_dot: torch.Tensor  # its horizontal rate along the heading [m/s]


@runtime_checkable
class BalanceStateProvider(Protocol):
    """Anything that can yield ``(zeta, zeta_dot)`` once per control step."""

    def balance_state(self) -> BalanceState: ...


class SimBalanceState:
    """Ground-truth ``zeta`` from simulator root states."""

    def __init__(self, num_envs: int, device: torch.device | str = "cpu"):
        self._num_envs = num_envs
        self._device = torch.device(device)
        self._state = BalanceState(
            zeta=torch.zeros(num_envs, device=self._device),
            zeta_dot=torch.zeros(num_envs, device=self._device),
        )

    def update(
        self,
        com_pos_w: torch.Tensor,
        com_vel_w: torch.Tensor,
        rear_contact_pos_w: torch.Tensor,
        heading_w: torch.Tensor,
    ) -> BalanceState:
        """Project (CoM - rear contact) onto the horizontal heading."""
        delta = com_pos_w - rear_contact_pos_w
        delta = torch.stack(
            [delta[:, 0], delta[:, 1], torch.zeros_like(delta[:, 2])], dim=-1
        )
        vel = torch.stack(
            [com_vel_w[:, 0], com_vel_w[:, 1], torch.zeros_like(com_vel_w[:, 2])], dim=-1
        )
        self._state = BalanceState(
            zeta=torch.sum(delta * heading_w, dim=-1),
            zeta_dot=torch.sum(vel * heading_w, dim=-1),
        )
        return self._state

    def balance_state(self) -> BalanceState:
        return self._state
