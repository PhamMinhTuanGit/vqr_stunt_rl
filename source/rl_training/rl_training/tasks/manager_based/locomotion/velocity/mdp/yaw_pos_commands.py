"""Neutral/positive yaw commands for Flat-VQR-Wheel-Yaw-POS only."""

from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.utils import configclass

from .commands import YawRateCommand, YawRateCommandCfg


class YawPosCommand(YawRateCommand):
    """Sample exact neutral or positive yaw and allow a runtime command override."""

    cfg: "YawPosCommandCfg"

    def __init__(self, cfg: "YawPosCommandCfg", env):
        if not 0.0 <= cfg.neutral_probability <= 1.0:
            raise ValueError("neutral_probability must be in [0, 1].")
        if not 0.0 < cfg.deadband < cfg.yaw_rate_range[1]:
            raise ValueError("Expected 0 < deadband < positive yaw limit.")
        super().__init__(cfg, env)
        self._external_control = cfg.external_control

    def _resample_command(self, env_ids: Sequence[int]):
        if self._external_control:
            return
        count = self._command[env_ids, 0].numel()
        if count == 0:
            return
        neutral = torch.rand(count, device=self.device) < self.cfg.neutral_probability
        limit = self.cfg.yaw_rate_range[1]
        active = self.cfg.deadband + (limit - self.cfg.deadband) * torch.rand(count, device=self.device)
        active = active.clamp_min(self.cfg.deadband + 1.0e-6)
        self._command[env_ids, 0] = torch.where(neutral, 0.0, active)

    def set_external_command(self, yaw_cmd: float | torch.Tensor, env_ids: Sequence[int] | slice = slice(None)) -> None:
        """Write joystick/runtime yaw into the same buffer observed by policy and rewards."""
        values = torch.as_tensor(yaw_cmd, dtype=self._command.dtype, device=self.device).flatten()
        count = self._command[env_ids, 0].numel()
        if values.numel() not in (1, count):
            raise ValueError(f"Expected one yaw command or {count} commands, got {values.numel()}.")
        if not torch.isfinite(values).all():
            raise ValueError("Yaw command must be finite.")
        values = values.expand(count).clamp(0.0, self.cfg.yaw_rate_range[1])
        self._command[env_ids, 0] = torch.where(values <= self.cfg.deadband, 0.0, values)
        self._external_control = True

    def use_training_sampler(self) -> None:
        """Return control to the scheduled within-episode sampler."""
        self._external_control = False


@configclass
class YawPosCommandCfg(YawRateCommandCfg):
    class_type: type = YawPosCommand
    yaw_rate_range: tuple[float, float] = (0.0, 0.25)
    deadband: float = 0.1
    neutral_probability: float = 0.30
    external_control: bool = False
