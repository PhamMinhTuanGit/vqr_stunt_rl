# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terrain curriculum with synchronized global gait_level update."""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)

    mean_level = torch.mean(terrain.terrain_levels.float())
    # Local import avoids module-load circular dependency.
    from .rewards import update_gait_level_from_terrain_mean

    update_gait_level_from_terrain_mean(mean_level)
    return mean_level


def gait_level_curve(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Return current global gait_level for logging in curriculum curves."""
    from .rewards import gait_level

    return torch.tensor(gait_level, device=env.device)


def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def yaw_task_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    yaw_rate_levels: Sequence[float],
    clearance_levels: Sequence[float],
    dr_levels: Sequence[float],
    support_reward_name: str,
    lift_reward_name: str,
    balance_reward_name: str,
    yaw_reward_name: str,
    support_threshold: float,
    lift_threshold: float,
    balance_threshold: float,
    yaw_threshold: float,
    min_evaluated_episodes: int,
    required_success_rate: float,
) -> dict[str, torch.Tensor]:
    """Progress the complete yaw task only after sustained task success.

    A single shared stage synchronizes yaw-command range, target lift clearance,
    and reset/interval domain-randomization intensity. This avoids making one
    axis harder while the others remain at an unrelated difficulty.
    """
    num_levels = len(yaw_rate_levels)
    if num_levels == 0 or len(clearance_levels) != num_levels or len(dr_levels) != num_levels:
        raise ValueError("yaw_rate_levels, clearance_levels, and dr_levels must have equal non-zero lengths.")

    if not hasattr(env, "_yaw_task_curriculum_stage"):
        env._yaw_task_curriculum_stage = 0
        env._yaw_task_curriculum_evaluated = 0
        env._yaw_task_curriculum_successes = 0
        env._yaw_task_curriculum_last_success_rate = 0.0

    if isinstance(env_ids, slice):
        selected_env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        selected_env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    episode_steps = env.episode_length_buf[selected_env_ids]
    valid = episode_steps > 0
    if torch.any(valid):
        completed_env_ids = selected_env_ids[valid]
        episode_duration = episode_steps[valid].float() * env.step_dt

        def normalized_score(reward_name: str) -> torch.Tensor:
            reward_weight = env.reward_manager.get_term_cfg(reward_name).weight
            if reward_weight <= 0.0:
                raise ValueError(f"Curriculum reward '{reward_name}' must have a positive weight.")
            weighted_sum = env.reward_manager._episode_sums[reward_name][completed_env_ids]
            return weighted_sum / (episode_duration * reward_weight)

        support_score = normalized_score(support_reward_name)
        lift_score = normalized_score(lift_reward_name)
        balance_score = normalized_score(balance_reward_name)
        yaw_score = normalized_score(yaw_reward_name)
        successful = (
            (support_score >= support_threshold)
            & (lift_score >= lift_threshold)
            & (balance_score >= balance_threshold)
            & (yaw_score >= yaw_threshold)
        )

        env._yaw_task_curriculum_evaluated += len(completed_env_ids)
        env._yaw_task_curriculum_successes += int(successful.sum().item())

    if env._yaw_task_curriculum_evaluated >= min_evaluated_episodes:
        success_rate = env._yaw_task_curriculum_successes / env._yaw_task_curriculum_evaluated
        env._yaw_task_curriculum_last_success_rate = success_rate
        if success_rate >= required_success_rate and env._yaw_task_curriculum_stage < num_levels - 1:
            env._yaw_task_curriculum_stage += 1
        env._yaw_task_curriculum_evaluated = 0
        env._yaw_task_curriculum_successes = 0

    stage = env._yaw_task_curriculum_stage
    yaw_limit = float(yaw_rate_levels[stage])
    target_clearance = float(clearance_levels[stage])
    dr_scale = float(dr_levels[stage])

    # Command curriculum.
    command_term = env.command_manager.get_term(command_name)
    command_term.cfg.yaw_rate_range = (-yaw_limit, yaw_limit)

    # Clearance curriculum: both the signed shaping reward and the yaw gate must
    # use the same target at every stage.
    for reward_name in (lift_reward_name, yaw_reward_name):
        reward_cfg = env.reward_manager.get_term_cfg(reward_name)
        reward_cfg.params["target_clearance"] = target_clearance
        env.reward_manager.set_term_cfg(reward_name, reward_cfg)

    # Reset/interval DR curriculum. Startup-only material/mass randomization is
    # intentionally left unchanged; the progressively scaled terms below are
    # safe to update online and cover disturbances, actuator gains and reset
    # state perturbations.
    force_limit = 10.0 * dr_scale
    force_cfg = env.event_manager.get_term_cfg("randomize_apply_external_force_torque")
    force_cfg.params["force_range"] = (-force_limit, force_limit)
    force_cfg.params["torque_range"] = (-force_limit, force_limit)
    env.event_manager.set_term_cfg("randomize_apply_external_force_torque", force_cfg)

    gain_delta = 0.15 * dr_scale
    gain_cfg = env.event_manager.get_term_cfg("randomize_actuator_gains")
    gain_cfg.params["stiffness_distribution_params"] = (1.0 - gain_delta, 1.0 + gain_delta)
    gain_cfg.params["damping_distribution_params"] = (1.0 - gain_delta, 1.0 + gain_delta)
    env.event_manager.set_term_cfg("randomize_actuator_gains", gain_cfg)

    push_limit = 0.5 * dr_scale
    push_cfg = env.event_manager.get_term_cfg("randomize_push_robot")
    push_cfg.params["velocity_range"] = {
        "x": (-push_limit, push_limit),
        "y": (-push_limit, push_limit),
    }
    env.event_manager.set_term_cfg("randomize_push_robot", push_cfg)

    reset_cfg = env.event_manager.get_term_cfg("randomize_reset_base")
    reset_cfg.params["pose_range"].update(
        {
            "roll": (-0.3 * dr_scale, 0.3 * dr_scale),
            "pitch": (-0.3 * dr_scale, 0.3 * dr_scale),
        }
    )
    reset_cfg.params["velocity_range"].update(
        {
            "x": (-0.2 * dr_scale, 0.2 * dr_scale),
            "y": (-0.2 * dr_scale, 0.2 * dr_scale),
            "z": (-0.2 * dr_scale, 0.2 * dr_scale),
            "roll": (-0.05 * dr_scale, 0.05 * dr_scale),
            "pitch": (-0.05 * dr_scale, 0.05 * dr_scale),
        }
    )
    env.event_manager.set_term_cfg("randomize_reset_base", reset_cfg)

    return {
        "stage": torch.tensor(stage, device=env.device),
        "yaw_limit": torch.tensor(yaw_limit, device=env.device),
        "target_clearance": torch.tensor(target_clearance, device=env.device),
        "dr_scale": torch.tensor(dr_scale, device=env.device),
        "window_success_rate": torch.tensor(env._yaw_task_curriculum_last_success_rate, device=env.device),
    }
