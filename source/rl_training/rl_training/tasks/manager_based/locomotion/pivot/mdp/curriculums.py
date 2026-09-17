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

def pivot_yaw_curriculum(env, env_ids, command_name: str, max_yaw: float, step: float):
    term = env.command_manager.get_term(command_name)
    ok = env.episode_length_buf[env_ids].float().mean() > 0.8 * env.max_episode_length
    if ok:
        hi = min(term.cfg.ang_vel_z[1] + step, max_yaw)
        term.cfg.ang_vel_z = (-hi, hi)
    return torch.tensor(term.cfg.ang_vel_z[1], device=env.device)


def pivot_yaw_rate_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    levels: Sequence[float],
    support_reward_name: str,
    lifted_reward_name: str,
    balance_reward_name: str,
    yaw_reward_name: str,
    support_threshold: float,
    lifted_threshold: float,
    balance_threshold: float,
    yaw_threshold: float,
    min_evaluated_episodes: int,
    required_success_rate: float,
) -> dict[str, torch.Tensor]:
    """Increase yaw range after a window passes all four M2 performance checks."""
    command_term = env.command_manager.get_term(command_name)

    if not hasattr(env, "_pivot_yaw_curriculum_stage"):
        current_limit = float(command_term.cfg.ranges.ang_vel_z[1])
        stage = min(range(len(levels)), key=lambda index: abs(float(levels[index]) - current_limit))
        env._pivot_yaw_curriculum_stage = stage
        env._pivot_yaw_curriculum_evaluated = 0
        env._pivot_yaw_curriculum_successes = 0
        env._pivot_yaw_curriculum_last_success_rate = 0.0

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
            weighted_sum = env.reward_manager._episode_sums[reward_name][completed_env_ids]
            return weighted_sum / (episode_duration * reward_weight)

        support_score = normalized_score(support_reward_name)
        lifted_score = normalized_score(lifted_reward_name)
        balance_score = normalized_score(balance_reward_name)
        yaw_score = normalized_score(yaw_reward_name)
        successful = (
            (support_score >= support_threshold)
            & (lifted_score >= lifted_threshold)
            & (balance_score >= balance_threshold)
            & (yaw_score >= yaw_threshold)
        )

        env._pivot_yaw_curriculum_evaluated += len(completed_env_ids)
        env._pivot_yaw_curriculum_successes += int(successful.sum().item())

    if env._pivot_yaw_curriculum_evaluated >= min_evaluated_episodes:
        success_rate = env._pivot_yaw_curriculum_successes / env._pivot_yaw_curriculum_evaluated
        env._pivot_yaw_curriculum_last_success_rate = success_rate
        if success_rate >= required_success_rate and env._pivot_yaw_curriculum_stage < len(levels) - 1:
            env._pivot_yaw_curriculum_stage += 1
        env._pivot_yaw_curriculum_evaluated = 0
        env._pivot_yaw_curriculum_successes = 0

    yaw_limit = float(levels[env._pivot_yaw_curriculum_stage])
    command_term.cfg.ranges.ang_vel_z = (-yaw_limit, yaw_limit)
    return {
        "stage": torch.tensor(env._pivot_yaw_curriculum_stage, device=env.device),
        "yaw_limit": torch.tensor(yaw_limit, device=env.device),
        "window_success_rate": torch.tensor(
            env._pivot_yaw_curriculum_last_success_rate, device=env.device
        ),
    }


# ---------------------------------------------------------------------------
# Four-mode pivot curriculum: stages S0-S6 (spec section 9).
# ---------------------------------------------------------------------------

GROUND = 0
REAR_UP = 1
BALANCE = 2
LAND = 3

# Gate criteria per stage; the env reads these from reward-manager averages.
PIVOT_STAGE_GATES = {
    0: {"reward": "yaw_ground_tracking", "threshold": 0.8},
    1: {"reward": "transition_success", "threshold": 0.90},
    2: {"reward": "capture_point_balance", "threshold": 0.7},
    3: {"metric": "abs_delta_theta", "threshold": 4.0},  # deg
    4: None,  # S4 opens tuck* and EstimatedBalanceState; no reward gate
    5: None,  # S5 forces low friction (mu -> 0.3) on 30% envs; no gate
    6: None,  # S6 is full DR + hard-state buffer; terminal stage
}
PIVOT_MAX_STAGE = 6


def pivot_stages(env, env_ids, command_name: str):
    """Advance the global four-mode stage once its gate is satisfied."""
    stage = int(getattr(env.unwrapped, "_pivot_stage", 0))
    gate = PIVOT_STAGE_GATES.get(stage)
    if gate is None or stage >= PIVOT_MAX_STAGE:
        return {"stage": torch.tensor(stage, device=env.device)}

    if "reward" in gate:
        mean_reward = float(
            env.reward_manager.get_term(gate["reward"]).mean().item()
        )
        passing = mean_reward > gate["threshold"]
    else:
        value = float(
            env.command_manager.get_term(command_name)
            .metrics[gate["metric"]]
            .mean()
            .item()
        )
        passing = value < gate["threshold"]
    if passing:
        stage += 1
        env.unwrapped._pivot_stage = stage

    return {
        "stage": torch.tensor(stage, device=env.device),
        "passing": torch.tensor(passing, device=env.device),
    }


def pivot_stage_adjustments(env) -> dict:
    """Per-stage MDP adjustments applied by PivotEnv._pre_physics_step (S4-S6)."""
    stage = int(getattr(env.unwrapped, "_pivot_stage", 0))
    return {
        "tuck_enabled": stage >= 4,
        "estimated_balance": stage >= 4,
        "full_dr": stage >= 6,
        "hard_state_resets": stage >= 6,
    }
