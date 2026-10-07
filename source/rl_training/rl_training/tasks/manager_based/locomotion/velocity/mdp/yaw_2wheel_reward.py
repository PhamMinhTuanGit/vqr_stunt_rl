# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, yaw_quat

from .fsm_gates import fsm_gates
from .fsm import select_swing_wheel_contact
from .wheel_contact_kinematics import wheel_center_positions, wheel_ground_clearance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



##### TASK REWARD ########
def custom_yaw_vel_tracking_exp(env: ManagerBasedRLEnv, std:float, command_name: str, 
                            asset_cfg:SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: RigidObject = env.scene[asset_cfg.name]
    # Để lấy vận tốc góc yaw thực tế trong Isaaclab
    # 
    actual_yaw_vel = robot.data.root_ang_vel_b[:,2]
    yaw_cmd = env.command_manager.get_command("command_name")
    return torch.exp(-torch.sum(torch.square(actual_yaw_vel - yaw_cmd))/0.25)

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward



#### HARDWARE PENALTY REWARDS #######
def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques (curriculum-scaled by gait_level)."""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    return reward 

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action rate (curriculum-scaled by gait_level)."""
    reward = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return reward 

def action_smooth_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize second-order action changes using an env-local action history."""
    cache_name = "_action_smooth_prev_prev_action"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, torch.zeros_like(env.action_manager.action))

    prev_prev_action = getattr(env, cache_name)
    diff = torch.square(env.action_manager.action + prev_prev_action - 2 * env.action_manager.prev_action)
    # Ignore the first two steps of each episode where action history is incomplete.
    diff = diff * (env.action_manager.prev_action != 0)
    diff = diff * (prev_prev_action != 0)
    reward = torch.sum(diff, dim=1)

    setattr(env, cache_name, env.action_manager.prev_action.clone())
    return reward 