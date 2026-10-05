"""Per-joint velocity noise for the POS experiment, before observation scaling."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg


def joint_velocity_noise(data: torch.Tensor, cfg: YawPosJointVelocityNoiseCfg) -> torch.Tensor:
    if data.shape[-1] != len(cfg.joint_names):
        raise ValueError("POS joint velocity noise must match the configured joint order.")
    bounds = data.new_tensor([
        cfg.wheel_noise if name.endswith("_WHEEL") else cfg.leg_noise
        for name in cfg.joint_names
    ])
    return data + (2.0 * torch.rand_like(data) - 1.0) * bounds


@configclass
class YawPosJointVelocityNoiseCfg(NoiseCfg):
    func = joint_velocity_noise
    joint_names: tuple[str, ...] = ()
    leg_noise: float = 3.0
    wheel_noise: float = 0.5
