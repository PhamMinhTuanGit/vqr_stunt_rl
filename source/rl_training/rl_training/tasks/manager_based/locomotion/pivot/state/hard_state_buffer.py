# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Ring buffer of recent hard states used for the 15% reset bucket.

Section 9: 15% of resets replay a state drawn from the last T - 0.3 s of
training so the policy repeatedly faces the near-tipover states it actually
visited instead of only freshly sampled ones.
"""

from __future__ import annotations

import torch

from ..config.wheeled.vqr.physical_params import VQR_PHYSICS


class HardStateBuffer:
    """Fixed-horizon ring buffer over arbitrary per-env tensor snapshots."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str = "cpu",
        window: float | None = None,
        dt: float | None = None,
    ):
        dt = VQR_PHYSICS.control_dt if dt is None else dt
        horizon = VQR_PHYSICS.hard_state_window if window is None else window
        self._length = max(1, int(round(horizon / dt)))
        self._index = -1  # nothing stored yet
        self._filled = 0
        self._states: dict[str, torch.Tensor] = {}
        self._num_envs = num_envs
        self._device = device

    @property
    def capacity(self) -> int:
        return self._length

    @property
    def filled(self) -> int:
        return min(self._filled, self._length)

    def update(self, **snapshots: torch.Tensor) -> None:
        """Store one per-env snapshot keyed by name (all ``(N, ...)``)."""
        if not snapshots:
            return
        if not self._states:
            self._states = {
                name: torch.zeros(self._length, *tensor.shape, device=tensor.device, dtype=tensor.dtype)
                for name, tensor in snapshots.items()
            }
        for name, tensor in snapshots.items():
            slot = (self._filled % self._length) if self._index < 0 else (self._index + 1) % self._length
            self._states[name][slot] = tensor
        self._index = (self._index + 1) % self._length
        self._filled += 1

    def sample(self, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """Uniformly sample one buffered frame; caller slices env ids."""
        if self.filled == 0:
            raise RuntimeError("HardStateBuffer.sample called before any update()")
        slot = int(torch.randint(0, self.filled, (1,), generator=generator).item())
        return {name: states[slot] for name, states in self._states.items()}

    def reset(self) -> None:
        self._states = {}
        self._index = -1
        self._filled = 0
