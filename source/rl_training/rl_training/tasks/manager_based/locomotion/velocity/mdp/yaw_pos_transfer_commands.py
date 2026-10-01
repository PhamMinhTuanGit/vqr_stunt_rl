"""Transfer-only neutral sampling across the closed yaw deadband."""

from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.utils import configclass

from .yaw_pos_commands import YawPosCommand, YawPosCommandCfg


class YawPosTransferCommand(YawPosCommand):
    """Keep the POS sampler; replace half its neutral zeros with uniform yaw."""

    cfg: "YawPosTransferCommandCfg"

    def __init__(self, cfg: "YawPosTransferCommandCfg", env):
        if not 0.0 <= cfg.neutral_uniform_fraction <= 1.0:
            raise ValueError("neutral_uniform_fraction must be in [0, 1].")
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        if self._external_control:
            return
        super()._resample_command(env_ids)
        values = self._command[env_ids, 0]
        if values.numel() == 0:
            return
        uniform_neutral = (values <= self.cfg.deadband) & (
            torch.rand_like(values) < self.cfg.neutral_uniform_fraction
        )
        deadband_values = torch.empty_like(values).uniform_(0.0, self.cfg.deadband)
        self._command[env_ids, 0] = torch.where(uniform_neutral, deadband_values, values)


@configclass
class YawPosTransferCommandCfg(YawPosCommandCfg):
    class_type: type = YawPosTransferCommand
    # Half of the existing 30% neutral mass: 15% zero + 15% uniform deadband.
    neutral_uniform_fraction: float = 0.50
