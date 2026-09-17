
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
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _wheel_contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Resolve contact only through the body IDs selected by ``sensor_cfg``."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.linalg.vector_norm(forces, dim=-1) > threshold


def pivot_support_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward the fraction of selected support wheels currently in contact."""
    return _wheel_contact_state(env, sensor_cfg, threshold).float().mean(dim=1)


def pivot_lifted_wheels(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float,
    minimum_clearance: float,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward lifted wheels for both no contact and positive ground clearance."""
    in_contact = _wheel_contact_state(env, sensor_cfg, threshold)
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    ground_height = env.scene.env_origins[:, 2].unsqueeze(-1)
    clearance = wheel_height - ground_height - wheel_radius
    clearance_score = torch.clamp(clearance / minimum_clearance, min=0.0, max=1.0)
    return 0.5 * (~in_contact).float().mean(dim=1) + 0.5 * clearance_score.mean(dim=1)


def pivot_balance(
    env: ManagerBasedRLEnv,
    nominal_roll: float,
    nominal_pitch: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward roll and pitch near the configured two-wheel equilibrium."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    roll_error = math_utils.wrap_to_pi(roll - nominal_roll)
    pitch_error = math_utils.wrap_to_pi(pitch - nominal_pitch)
    return torch.exp(-(roll_error.square() + pitch_error.square()) / std**2)


def pivot_angular_stability(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body-frame roll and pitch rates."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(asset.data.root_ang_vel_b[:, :2].square(), dim=1)


def pivot_planar_drift(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body-frame planar base velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(asset.data.root_lin_vel_b[:, :2].square(), dim=1)


def pivot_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize changes in the complete 16-dimensional action."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def pivot_leg_effort_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize applied effort only on the explicitly selected leg joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(asset.data.applied_torque[:, asset_cfg.joint_ids].square(), dim=1)


def pivot_track_yaw_rate(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward body-frame yaw rate tracking with an exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    yaw_rate_command = env.command_manager.get_command(command_name)[:, 2]
    error = asset.data.root_ang_vel_b[:, 2] - yaw_rate_command
    return torch.exp(-error.square() / std**2)


def lateral_wheel_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float = 2.0,
) -> torch.Tensor:
    """Penalize contacted wheels' velocity along their local axle direction.

    The VQR wheel joints all rotate about local Y (the URDF axes are
    ``(0, -1, 0)``), so wheel-frame Y is the lateral/axle direction.  Forward
    rolling in wheel-frame X is intentionally not penalized.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_velocity_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    wheel_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    wheel_velocity_local = math_utils.quat_apply_inverse(wheel_quat_w, wheel_velocity_w)

    in_contact = _wheel_contact_state(env, sensor_cfg, threshold)
    lateral_velocity = wheel_velocity_local[..., 1]
    return torch.sum(in_contact * lateral_velocity.square(), dim=1)


def _scalar_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return a scalar command as a one-dimensional environment batch."""
    command = env.command_manager.get_command(command_name)
    if command.shape[1] != 1:
        raise ValueError(f"Command '{command_name}' must be scalar, received shape {tuple(command.shape)}.")
    return command[:, 0]


def _wheel_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float,
) -> torch.Tensor:
    """Return selected wheel-bottom heights relative to the local ground plane."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    ground_height = env.scene.env_origins[:, 2].unsqueeze(-1)
    return wheel_height - ground_height - wheel_radius


def m3_four_wheel_support(
    env: ManagerBasedRLEnv,
    lift_command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward four-wheel contact while the lift command is near zero."""
    lift = _scalar_command(env, lift_command_name)
    contact_score = _wheel_contact_state(env, sensor_cfg, threshold).float().mean(dim=1)
    return (1.0 - lift) * contact_score


def m3_diagonal_support(
    env: ManagerBasedRLEnv,
    lift_command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward FL-HR support in proportion to the commanded lift progress."""
    lift = _scalar_command(env, lift_command_name)
    support_score = _wheel_contact_state(env, sensor_cfg, threshold).float().mean(dim=1)
    return lift * support_score


def m3_lifted_diagonal(
    env: ManagerBasedRLEnv,
    lift_command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float,
    target_clearance: float,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward both FR and HL being clear, using the weaker wheel's score."""
    lift = _scalar_command(env, lift_command_name)
    in_contact = _wheel_contact_state(env, sensor_cfg, threshold)
    clearance = _wheel_clearance(env, asset_cfg, wheel_radius)
    clearance_score = torch.clamp(clearance / target_clearance, min=0.0, max=1.0)
    per_wheel_score = 0.5 * (~in_contact).float() + 0.5 * clearance_score
    return lift * per_wheel_score.amin(dim=1)


def m3_track_yaw_rate(
    env: ManagerBasedRLEnv,
    yaw_command_name: str,
    lift_command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track yaw only as the policy is asked to enter two-wheel support."""
    lift = _scalar_command(env, lift_command_name)
    return lift * pivot_track_yaw_rate(env, yaw_command_name, std, asset_cfg)


def m3_xy_displacement_dead_zone(
    env: ManagerBasedRLEnv,
    dead_zone: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared XY displacement beyond a radius around the reset pose."""
    if not hasattr(env, "_pivot_reset_root_xy"):
        raise RuntimeError("M3 reset XY buffer is unavailable; reset_four_wheel_standing must run first.")
    asset: Articulation = env.scene[asset_cfg.name]
    displacement = torch.linalg.vector_norm(
        asset.data.root_pos_w[:, :2] - env._pivot_reset_root_xy,
        dim=1,
    )
    return torch.clamp(displacement - dead_zone, min=0.0).square()


# Global curriculum scalar in [0, 1], updated from terrain-level mean.
gait_level: float = 0.0

def update_gait_level_from_terrain_mean(terrain_level_mean: float | torch.Tensor) -> float:
    """Update global gait_level from mean terrain level.

    Mapping rule:
    - mean <= 0.0 -> 0.0
    - 0.0 < mean < 3.0 -> 使用 exp 函数映射
    - mean == 3.0 -> 1.0
    - mean >= 3.0 -> 1.0
    """
    global gait_level

    mean_tensor = torch.as_tensor(terrain_level_mean, dtype=torch.float32)
    if mean_tensor.numel() == 0:
        mean_val = 0.0
    else:
        mean_val = float(torch.mean(mean_tensor).item())

    if math.isnan(mean_val) or math.isinf(mean_val):
        mean_val = 0.0

    if mean_val <= 0.0:
        gait_level = 0.0
    elif mean_val < 3.0:
        # exp 映射：mean=0 时接近 0，mean=3 时恰好为 1
        gait_level = math.exp(mean_val - 3.0)
    else:  # mean_val >= 3.0
        gait_level = 1.0

    return gait_level

def get_gait_level_tensor(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return global gait_level as tensor matching environment batch size."""
    return torch.full((env.num_envs,), gait_level, device=env.device)


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


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques (curriculum-scaled by gait_level)."""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    return reward * get_gait_level_tensor(env)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action rate (curriculum-scaled by gait_level)."""
    reward = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return reward * get_gait_level_tensor(env)

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
    return reward * get_gait_level_tensor(env)

def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact force violations (curriculum-scaled by gait_level)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    reward = torch.sum(violation.clip(min=0.0), dim=1)
    return reward * get_gait_level_tensor(env)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward * get_gait_level_tensor(env)


def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] > 0
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward * get_gait_level_tensor(env)


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def feet_air_time(
#     env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
# ) -> torch.Tensor:
#     """Reward long steps taken by the feet using L2-kernel.

#     This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
#     that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
#     the time for which the feet are in the air.

#     If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
#     """
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
#     # no reward for zero command
#     reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
#     # print(torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1), "command norm")
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    # return reward

# def feet_air_time(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     sensor_cfg: SceneEntityCfg,
#     mode_time: float,
#     velocity_threshold: float,
# ) -> torch.Tensor:
#     """Reward longer feet air and contact time."""
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     asset: Articulation = env.scene[asset_cfg.name]
#     if contact_sensor.cfg.track_air_time is False:
#         raise RuntimeError("Activate ContactSensor's track_air_time!")
#     # compute the reward
#     current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
#     current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

#     t_max = torch.max(current_air_time, current_contact_time)
#     t_min = torch.clip(t_max, max=mode_time)
#     stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
#     cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
#     body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
#     reward = torch.where(
#         torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
#         torch.where(t_max < mode_time, t_min, 0),
#         stance_cmd_reward,
#     )
#     return torch.sum(reward, dim=1)


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1)
    # print(last_air_time, "last air time")
    # print(last_contact_time, "last contact time")
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward




def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.5
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # print(contact, "contact")
    reward = torch.sum(contact, dim=-1).float()
    # print(reward, "reward after sum")
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.5
    # print(env.command_manager.get_command(command_name), "env.command_manager.get_command(command_name)")
    # print(reward, "reward after multiply")
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    # foot_velocity_tanh = torch.tanh(
    #     tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    # )
    # reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward = torch.sum(foot_z_target_error, dim=1)
    # print(foot_z_target_error, "foot_z_target_error")
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.2
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def _bernstein_torch(n: int, k: int, t: torch.Tensor) -> torch.Tensor:
    """Bernstein basis B_k^n(t) for tensor t in [0, 1]."""
    coeff = float(math.comb(n, k))
    return coeff * (1.0 - t) ** (n - k) * t**k


def _bezier_curve_torch(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate Bezier curve points for batched parameter t.

    Args:
        control_points: Tensor of shape [m, 2].
        t: Tensor of shape [N, L] in [0, 1].

    Returns:
        Tensor of shape [N, L, 2].
    """
    n = control_points.shape[0] - 1
    out = torch.zeros(*t.shape, 2, device=t.device, dtype=t.dtype)
    for k in range(n + 1):
        out = out + _bernstein_torch(n, k, t).unsqueeze(-1) * control_points[k]
    return out


def _bezier_curve_derivative_torch(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate d/dt of Bezier curve points for batched parameter t."""
    n = control_points.shape[0] - 1
    delta_ctrl = n * (control_points[1:] - control_points[:-1])
    out = torch.zeros(*t.shape, 2, device=t.device, dtype=t.dtype)
    for k in range(n):
        out = out + _bernstein_torch(n - 1, k, t).unsqueeze(-1) * delta_ctrl[k]
    return out


def phase_foot_trajectory_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 0.1,
    command_threshold: float = 0.1,
    cycle_time: float = 0.4,
    phase_offsets: tuple[float, ...] = (0.0, 1.0, 1.0, 0.0),
    gait_span: float = -0.008,
    gait_psi: float = 0.15,
    gait_delta: float = 0.03,
    x_offset: float = 0.0,
    stance_span: float = 0.20,
    stand_ref_z_offset: float = -0.2,
    velocity_weight: float = 0.5,
) -> torch.Tensor:
    """Track MuJoCo-style phase foot trajectory in body frame with exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    num_feet = len(body_ids)

    if num_feet == 0:
        return torch.zeros(env.num_envs, device=env.device)
    if len(phase_offsets) != num_feet:
        raise ValueError(f"phase_offsets length ({len(phase_offsets)}) must match tracked feet ({num_feet}).")

    # Build and cache base-fixed stand references from the first call.
    if (not hasattr(env, "phase_foot_ref_body")) or (env.phase_foot_ref_body.shape[1] != num_feet):
        rel_foot_pos_w = asset.data.body_pos_w[:, body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
        foot_pos_b = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
        for i in range(num_feet):
            foot_pos_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, rel_foot_pos_w[:, i, :])
        ref = foot_pos_b[0].detach().clone()
        ref[:, 2] += stand_ref_z_offset
        env.phase_foot_ref_body = ref.unsqueeze(0)

    stand_ref_body = env.phase_foot_ref_body.to(env.device).expand(env.num_envs, -1, -1)

    # Build phase S in [0, 2).
    phase_time = env.episode_length_buf.float() * env.step_dt
    phase_offsets_t = torch.tensor(phase_offsets, device=env.device, dtype=phase_time.dtype).unsqueeze(0)
    S = torch.remainder((2.0 * phase_time / max(cycle_time, 1e-6)).unsqueeze(1) + phase_offsets_t, 2.0)

    # MuJoCo-like piecewise trajectory in local (q, z).
    tau = float(gait_span)
    psi = float(gait_psi)
    delta = float(gait_delta)
    stance_span = float(stance_span)
    stance_span = min(max(stance_span, 1e-6), 2.0 - 1e-6)

    q = torch.zeros_like(S)
    z = torch.zeros_like(S)
    dq_dS = torch.zeros_like(S)
    dz_dS = torch.zeros_like(S)

    stance_mask = S < stance_span
    if stance_mask.any():
        s_stance = S / stance_span
        q_stance = tau * (1.0 - 2.0 * s_stance)
        z_stance = torch.full_like(S, delta)
        dq_dS_stance = torch.full_like(S, -2.0 * tau / stance_span)
        dz_dS_stance = torch.zeros_like(S)

        q = torch.where(stance_mask, q_stance, q)
        z = torch.where(stance_mask, z_stance, z)
        dq_dS = torch.where(stance_mask, dq_dS_stance, dq_dS)
        dz_dS = torch.where(stance_mask, dz_dS_stance, dz_dS)

    swing_mask = ~stance_mask
    if swing_mask.any():
        t_bezier = torch.clamp((S - stance_span) / (2.0 - stance_span), 0.0, 1.0)
        ctrl = torch.tensor(
            [
                [-tau, 0.0],
                [-0.95 * tau, 0.80 * psi],
                [-0.55 * tau, 1.00 * psi],
                [0.55 * tau, 1.00 * psi],
                [0.95 * tau, 0.80 * psi],
                [tau, 0.0],
            ],
            device=env.device,
            dtype=S.dtype,
        )
        qz_swing = _bezier_curve_torch(ctrl, t_bezier)
        dqz_dt = _bezier_curve_derivative_torch(ctrl, t_bezier)
        dt_dS = 1.0 / (2.0 - stance_span)

        q = torch.where(swing_mask, qz_swing[..., 0], q)
        z = torch.where(swing_mask, qz_swing[..., 1] + delta, z)
        dq_dS = torch.where(swing_mask, dqz_dt[..., 0] * dt_dS, dq_dS)
        dz_dS = torch.where(swing_mask, dqz_dt[..., 1] * dt_dS, dz_dS)

    dS_dt = 2.0 / max(cycle_time, 1e-6)
    dq_dt = dq_dS * dS_dt
    dz_dt = dz_dS * dS_dt

    ref_pos_b = stand_ref_body + torch.stack(
        [q + float(x_offset), torch.zeros_like(q), z],
        dim=-1,
    )
    ref_vel_b = torch.stack(
        [dq_dt, torch.zeros_like(dq_dt), dz_dt],
        dim=-1,
    )

    # Actual foot states in body frame.
    rel_foot_pos_w = asset.data.body_pos_w[:, body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    rel_foot_vel_w = asset.data.body_lin_vel_w[:, body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    foot_pos_b = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
    foot_vel_b = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
    for i in range(num_feet):
        foot_pos_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, rel_foot_pos_w[:, i, :])
        foot_vel_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, rel_foot_vel_w[:, i, :])

    pos_offset = foot_pos_b - ref_pos_b
    vel_offset = foot_vel_b - ref_vel_b

    # Per-dimension (x, y, z) errors over feet for each environment.
    pos_err = torch.sum(torch.square(pos_offset), dim=1)
    vel_err = torch.sum(torch.square(vel_offset), dim=1)

    # Scalar total error for reward computation.
    total_err = torch.sum(pos_err, dim=1) + float(velocity_weight) * torch.sum(vel_err, dim=1)
    reward = torch.exp(-total_err / max(std, 1e-6) ** 2)

    # Command-gating follows full (x, y, yaw) command magnitude.
    command = env.command_manager.get_command(command_name)
    gate = torch.linalg.norm(command[:, :3], dim=1) > command_threshold

    # info for debugging
    pos_offset_xyz_mean = torch.mean(pos_offset, dim=(0, 1))
    vel_offset_xyz_mean = torch.mean(vel_offset, dim=(0, 1))
    # print(
    #     "Offset xyz mean | "
    #     f"pos(x,y,z)=({pos_offset_xyz_mean[0].item():.4f}, {pos_offset_xyz_mean[1].item():.4f}, {pos_offset_xyz_mean[2].item():.4f}) | "
    #     f"vel(x,y,z)=({vel_offset_xyz_mean[0].item():.4f}, {vel_offset_xyz_mean[1].item():.4f}, {vel_offset_xyz_mean[2].item():.4f})"
    # )
    # print("Reward:", reward * gate.float())
    return reward * gate.float() * get_gait_level_tensor(env)

def foot_impact_velocity(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    speed_threshold: float = 0.10,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].float()
    foot_lin_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]

    downward_speed = torch.clamp(-foot_lin_vel[:, :, 2], min=0.0)
    downward_speed = torch.clamp(downward_speed - speed_threshold, min=0.0)

    penalty = torch.sum(first_contact * torch.square(downward_speed), dim=1)
    return penalty * get_gait_level_tensor(env)

# def stand_still_joint_deviation_l1(
#     env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Penalize offsets from the default joint positions when the command is very small."""
    # command = env.command_manager.get_command(command_name)
#     # Penalize motion when command is nearly zero.
#     return joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :], dim=1) < command_threshold)

# def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize joint positions that deviate from the default one."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # compute out of limits constraints
#     angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
#     return torch.sum(torch.abs(angle), dim=1)


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def joint_acc_l2_new(env: ManagerBasedRLEnv) -> torch.Tensor:

# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     # print(torch.sum(diff, dim=1), "smoothness l2")
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward





def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward * get_gait_level_tensor(env)


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_air_time_lin_xy_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Air-time reward gated by planar linear velocity command (x, y)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Core logic unchanged
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    # Gate by planar linear velocity command only
    cmd_lin_xy = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    reward *= cmd_lin_xy > cmd_threshold
    return reward * get_gait_level_tensor(env)

def feet_air_time_x_neg_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Air-time reward gated by negative x velocity command only."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Core logic unchanged
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    # Gate: x command must be negative and exceed magnitude threshold
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    reward *= cmd_x < 0.0

    return reward

def feet_air_time_ang_z_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Air-time reward gated by yaw angular velocity command (z)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Core logic unchanged
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    # Gate by angular velocity command only
    cmd_ang_z = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    reward *= cmd_ang_z > cmd_threshold
    return reward * get_gait_level_tensor(env)

def feet_air_time_including_ang_z(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    return reward

def lin_vel_xy_l2_with_ang_z_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Penalize xy-axis base linear velocity using L2 squared kernel if command is ang_vel_z."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    command = env.command_manager.get_command(command_name)
    reward *= (torch.sum(torch.square(command[:, 2:]), dim=1) > command_threshold) & \
            (torch.sum(torch.square(command[:, :2]), dim=1) < command_threshold)
    # reward *= torch.sum(torch.square(env.command_manager.get_command(command_name)[:, 2:]), dim=1) > command_threshold
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

#Phần reward cho task pivot 
# mdp/rewards.py



def _pitch(env, asset_cfg=SceneEntityCfg("robot")):
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    return torch.atan2(-g[:, 0], -g[:, 2])          # 0 = nằm ngang, + = ngửa mũi lên


# ---------- 1. giữ tư thế wheelie ----------
def pivot_pitch_tracking(env, std: float, command_name: str = "pivot"):
    tgt = env.command_manager.get_command(command_name)[:, 1]
    return torch.exp(-torch.square(_pitch(env) - tgt) / std**2)


def front_wheels_height(env, target_h: float, asset_cfg: SceneEntityCfg):
    """asset_cfg.body_names = ['fl_wheel', 'fr_wheel']"""
    asset = env.scene[asset_cfg.name]
    h = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(-1)
    return torch.clamp(h.mean(dim=1) / target_h, max=1.0)


def rear_only_contact(env, front_cfg: SceneEntityCfg, rear_cfg: SceneEntityCfg, thr: float = 1.0):
    cs = env.scene.sensors
    f = cs[front_cfg.name].data.net_forces_w_history[:, :, front_cfg.body_ids].norm(dim=-1).amax(dim=1)
    r = cs[rear_cfg.name].data.net_forces_w_history[:, :, rear_cfg.body_ids].norm(dim=-1).amax(dim=1)
    return (r > thr).all(dim=1).float() - (f > thr).any(dim=1).float()


# ---------- 2. xoay tại chỗ ----------
def pivot_yaw_tracking(env, std: float, command_name: str = "pivot"):
    tgt = env.command_manager.get_command(command_name)[:, 0]
    w   = env.scene["robot"].data.root_ang_vel_b[:, 2]
    return torch.exp(-torch.square(w - tgt) / std**2)


def stay_in_place(env, std: float, asset_cfg=SceneEntityCfg("robot")):
    """'Tại chỗ' = phạt theo VỊ TRÍ world, KHÔNG dùng track_lin_vel (xem mục 6)."""
    p = env.scene[asset_cfg.name].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return torch.exp(-torch.sum(torch.square(p), dim=1) / std**2)


def flat_roll_only(env, asset_cfg=SceneEntityCfg("robot")):
    """Thay cho flat_orientation_l2: CHỈ phạt roll, KHÔNG phạt pitch."""
    return torch.square(env.scene[asset_cfg.name].data.projected_gravity_b[:, 1])


# ---------- 3. động tác biểu diễn ----------
def gesture_tracking(env, std: float, amp: float, command_name: str, asset_cfg: SceneEntityCfg):
    """asset_cfg.joint_names = khớp hipy/knee của 2 chân trước (thứ tự: fl_hipy, fr_hipy, fl_knee, fr_knee)."""
    asset = env.scene[asset_cfg.name]
    cmd   = env.command_manager.get_command(command_name)
    s, c  = cmd[:, 2:3], cmd[:, 3:4]
    q0    = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    n     = len(asset_cfg.joint_ids) // 2
    # 2 chân trước lệch pha 180° -> vẫy so le cho đẹp
    ref   = q0 + amp * torch.cat([s, -s], dim=1).repeat(1, n)
    q     = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.exp(-torch.sum(torch.square(q - ref), dim=1) / std**2)


def front_wheel_spin(env, target_speed: float, asset_cfg: SceneEntityCfg):
    """Cho 2 bánh trước quay tít khi lơ lửng — hiệu ứng biểu diễn (weight DƯƠNG)."""
    v = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    return torch.exp(-torch.square(v.abs().mean(dim=1) - target_speed) / (0.3 * target_speed) ** 2)


def pitch_collapsed(env, min_pitch: float, grace_s: float):
    grace = int(grace_s / env.step_dt)
    return (_pitch(env) < min_pitch) & (env.episode_length_buf > grace)

def drifted_away(env, max_dist: float):
    p = env.scene["robot"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return p.norm(dim=1) > max_dist


# ---------------------------------------------------------------------------
# Four-mode pivot rewards (spec section 7): mode-gated, cache-fed (I8).
# ---------------------------------------------------------------------------

GROUND = 0
REAR_UP = 1
BALANCE = 2
LAND = 3


def _pivot_balance_core(env: ManagerBasedRLEnv) -> tuple:
    """(zeta, zeta_dot, xi) straight from the cache (invariants I1/I8)."""
    from .observations import _pivot_cache

    cache = _pivot_cache(env)
    return cache.zeta, cache.zeta_dot, cache.xi


def _mode_gate(env, command_name: str) -> torch.Tensor:
    """Current mode id per env from the 8-dim pivot command (section 7)."""
    cmd = env.command_manager.get_command(command_name)
    return cmd[:, :4].argmax(dim=-1)  # one-hot slot holding the mode


def mode_gated(term_func, weights: dict):
    """Wrap ``term_func`` with per-mode weights; unlisted modes contribute 0.

    The wrapped signature matches Isaac Lab reward terms: the first two
    positional args (env, cfg params...) pass through untouched, and the
    factory closes over ``command_name`` so cfg params stay keyword-safe.
    """
    weight_values = torch.tensor(list(weights.values()))

    def wrapped(env, *args, **kwargs):
        mode = _mode_gate(env, kwargs["command_name"])
        out = term_func(env, *args, **kwargs)
        gate = torch.zeros_like(mode, dtype=out.dtype)
        for mode_id, weight in weights.items():
            gate = torch.where(
                mode == mode_id, torch.full_like(gate, weight, dtype=out.dtype), gate
            )
        return out * gate if out.dim() == 1 else out * gate.unsqueeze(-1)
    return wrapped


def capture_point_reward(env: ManagerBasedRLEnv, std: float = 0.12) -> torch.Tensor:
    """exp(-xi^2 / sigma^2): the balance variable IS xi (I1)."""
    _, _, xi = _pivot_balance_core(env)
    return torch.exp(-xi.square() / std.square())


def omega_z_tracking(
    env: ManagerBasedRLEnv, command_name: str, std: float = 0.5
) -> torch.Tensor:
    """Yaw-rate tracking; gated OFF past xi_safe together with anchor (I4)."""
    from .observations import _pivot_cache

    cache = _pivot_cache(env)
    cmd = env.command_manager.get_command(command_name)
    target = cmd[:, 4]
    gate = (cache.xi.abs() <= cache.xi_safe).to(target.dtype)
    return gate * torch.exp(-(target - cache.omega_z).square() / std.square())


def anchor_midpoint_reward(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Anchor drift of the rear-axle midpoint; OFF past xi_safe or LAND (I4)."""
    from .observations import _pivot_cache

    cache = _pivot_cache(env)
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    mode = cmd[:, :4].argmax(dim=-1)
    # Rear-axle midpoint wheel link indices resolved at env init.
    rear_ids = getattr(env.unwrapped, "pivot_rear_wheel_ids", None)
    if rear_ids is None:
        rear_ids = torch.tensor([2, 3], device=robot.device)
    mid = robot.data.body_pos_w[:, rear_ids, :2].mean(dim=1)
    drift = (mid - env.scene.env_origins[:, :2]).norm(dim=-1)
    gate = (cache.xi.abs() <= cache.xi_safe) & (~_is_land(env, command_name))
    gate = gate.to(drift.dtype)
    return gate * drift


def _is_land(env, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return cmd[:, 3] > 0.5


def front_wheels_off_ground(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fraction of the front wheel pair clear of the ground."""
    from .observations import _pivot_cache

    cache = _pivot_cache(env)
    front = cache.wheel_contact[:, :2]
    return (front < 0.5).float().mean(dim=-1)


def roll_flat_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """|roll| from the cache; roll is always driven toward zero."""
    from .observations import _pivot_cache

    cache = _pivot_cache(env)
    return cache.roll.abs()


def tuck_tracking(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Bám tuck*: exp(-||realized tuck - tuck*||^2 / std^2) in BALANCE."""
    from .observations import _pivot_cache

    cache = _pivot_cache(env)
    cmd = env.command_manager.get_command(command_name)
    tuck_cmd = cmd[:, 7]
    realized = getattr(env.unwrapped, "pivot_realized_tuck", None)
    if realized is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.exp(-torch.square(realized - tuck_cmd) / 0.25)


def thermal_excess_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """max(0, EMA|tau_w| - 3 Nm)^2 averaged over wheels; cache-fed."""
    from .observations import _pivot_cache

    cache = _pivot_cache(env)
    excess = torch.relu(cache.ema_wheel_torque - cache.physics.wheel_continuous_torque)
    return excess.square().mean(dim=-1)


def self_collision_analytic(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Analytic capsule self-collision violation (I6): reward-only."""
    from .observations import _pivot_cache

    cache = _pivot_cache(env)
    clearance = getattr(env.unwrapped, "pivot_clearance", None)
    if clearance is None:
        return torch.zeros(env.num_envs, device=env.device)
    target = getattr(cache, "min_clearance_target", 0.02)
    return torch.relu(target - clearance).square()


def landing_peak_force(env: ManagerBasedRLEnv) -> torch.Tensor:
    """max F_n over wheels normalised by the 645 N landing budget."""
    from .observations import _pivot_cache

    cache = _pivot_cache(env)
    sensor = env.scene.sensors["contact_forces"]
    fn_peak = sensor.data.net_forces_w[:, :, 2].abs().amax(dim=-1)
    return fn_peak / cache.physics.fn_peak_budget


def backflip_excess_penalty(
    env: ManagerBasedRLEnv, bias_rad: float = 0.0
) -> torch.Tensor:
    """I5: pitch strictly beyond theta*(omega_z) (+ optional forward bias)."""
    from .observations import _pivot_cache

    cache = _pivot_cache(env)
    return torch.relu(cache.pitch - (cache.theta_star + bias_rad))
