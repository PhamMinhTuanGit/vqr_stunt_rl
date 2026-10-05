"""Telemetry-only wrapper: the POS neutral reward value is unchanged."""

import torch
from isaaclab.managers import SceneEntityCfg

from .yaw_pos_rewards import yaw_pos_neutral_position, yaw_pos_masks


def skill_neutral_position(
    env, sensor_cfg: SceneEntityCfg, command_name: str, deadband: float,
    std: float = 0.05, contact_dwell: float = 0.2, contact_threshold: float = 1.0,
    speed_threshold: float = 0.03, drift_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    value = yaw_pos_neutral_position(
        env, sensor_cfg, command_name, deadband, std, contact_dwell,
        contact_threshold, speed_threshold, drift_threshold, asset_cfg,
    )
    _, _, neutral = yaw_pos_masks(env, command_name, deadband)
    if not hasattr(env, "_yaw_pos_skill_anchor_sum"):
        env._yaw_pos_skill_anchor_sum = torch.zeros(env.num_envs, device=env.device)
        env._yaw_pos_skill_neutral_samples = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._yaw_pos_skill_anchor_sum.add_((neutral & env._yaw_pos_neutral_anchored).float())
    env._yaw_pos_skill_neutral_samples.add_(neutral.long())
    return value
