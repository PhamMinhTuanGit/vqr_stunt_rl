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


def yaw_tracking_ratio(
    mean_abs_yaw_error: torch.Tensor,
    mean_abs_yaw_command: torch.Tensor,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Return command-scale-invariant yaw tracking progress.

    A policy that outputs zero yaw has equal mean absolute command and error,
    hence ratio zero at every symmetric command limit.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    return torch.where(
        mean_abs_yaw_command > epsilon,
        1.0 - mean_abs_yaw_error / mean_abs_yaw_command.clamp_min(epsilon),
        torch.zeros_like(mean_abs_yaw_command),
    )


def validate_yaw_curriculum_levels(
    clearance_levels: Sequence[float],
    yaw_rate_levels: Sequence[float],
    dr_scale_levels: Sequence[float],
    tracking_ratio_thresholds: Sequence[float],
    edge_tracking_ratio_thresholds: Sequence[float],
) -> None:
    """Validate the stage tables used by :func:`yaw_task_levels`."""
    if not clearance_levels or any(level <= 0.0 for level in clearance_levels):
        raise ValueError("clearance_levels must contain positive values.")
    yaw_level_count = len(yaw_rate_levels)
    if yaw_level_count == 0 or not (
        len(dr_scale_levels)
        == len(tracking_ratio_thresholds)
        == len(edge_tracking_ratio_thresholds)
        == yaw_level_count
    ):
        raise ValueError(
            "yaw_rate_levels, dr_scale_levels, tracking_ratio_thresholds, and "
            "edge_tracking_ratio_thresholds must have equal non-zero lengths."
        )
    if yaw_rate_levels[0] <= 0.0 or any(
        current <= previous for previous, current in zip(yaw_rate_levels, yaw_rate_levels[1:])
    ):
        raise ValueError("yaw_rate_levels must be positive and strictly increasing.")
    if any(not 0.0 <= value <= 1.0 for value in dr_scale_levels):
        raise ValueError("dr_scale_levels values must be in [0, 1].")
    if any(current < previous for previous, current in zip(dr_scale_levels, dr_scale_levels[1:])):
        raise ValueError("dr_scale_levels must be non-decreasing.")
    for name, values in (
        ("tracking_ratio_thresholds", tracking_ratio_thresholds),
        ("edge_tracking_ratio_thresholds", edge_tracking_ratio_thresholds),
    ):
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{name} values must be in [0, 1].")


def yaw_curriculum_window_passes(
    success_rate: float,
    tracking_ratio: float,
    edge_tracking_ratio: float,
    required_success_rate: float,
    tracking_ratio_threshold: float,
    edge_tracking_ratio_threshold: float,
) -> bool:
    """Return whether a completed yaw-stage evaluation window passes all gates."""
    return (
        success_rate >= required_success_rate
        and tracking_ratio >= tracking_ratio_threshold
        and edge_tracking_ratio >= edge_tracking_ratio_threshold
    )


def yaw_task_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    clearance_levels: Sequence[float],
    yaw_rate_levels: Sequence[float],
    dr_scale_levels: Sequence[float],
    tracking_ratio_thresholds: Sequence[float],
    edge_tracking_ratio_thresholds: Sequence[float],
    lift_reward_name: str,
    balance_reward_name: str,
    yaw_reward_name: str,
    torso_contact_termination_name: str,
    minimum_base_height: float,
    support_threshold: float,
    lift_progress_threshold: float,
    balance_threshold: float,
    yaw_threshold: float,
    min_evaluated_episodes: int,
    required_success_rate: float,
    required_consecutive_windows: int,
    min_clearance_stage_steps: int,
    min_yaw_stage_steps: int,
) -> dict[str, torch.Tensor]:
    """Learn the two-wheel pose first, then jointly increase yaw and online DR.

    Clearance and yaw use independent stage indices. Once the final clearance
    is stable, each yaw promotion also increases reset/interval DR. Startup-only
    material, mass, inertia and CoM randomization is intentionally unaffected.
    """
    validate_yaw_curriculum_levels(
        clearance_levels,
        yaw_rate_levels,
        dr_scale_levels,
        tracking_ratio_thresholds,
        edge_tracking_ratio_thresholds,
    )
    if min_evaluated_episodes <= 0:
        raise ValueError("min_evaluated_episodes must be positive.")
    if required_consecutive_windows <= 0:
        raise ValueError("required_consecutive_windows must be positive.")
    if min_clearance_stage_steps < 0 or min_yaw_stage_steps < 0:
        raise ValueError("Minimum stage durations must be non-negative.")
    if not 0.0 <= required_success_rate <= 1.0:
        raise ValueError("required_success_rate must be in [0, 1].")

    current_step = int(env.common_step_counter)
    state_defaults = {
        "_yaw_task_curriculum_stage": 0,
        "_yaw_task_curriculum_yaw_stage": 0,
        "_yaw_task_curriculum_stage_start_step": current_step,
        "_yaw_task_curriculum_consecutive_passes": 0,
        "_yaw_task_curriculum_evaluated": 0,
        "_yaw_task_curriculum_successes": 0,
        "_yaw_task_curriculum_yaw_score_sum": 0.0,
        "_yaw_task_curriculum_yaw_score_samples": 0,
        "_yaw_task_curriculum_command_abs_sum": 0.0,
        "_yaw_task_curriculum_yaw_abs_error_sum": 0.0,
        "_yaw_task_curriculum_tracking_samples": 0,
        "_yaw_task_curriculum_edge_command_abs_sum": 0.0,
        "_yaw_task_curriculum_edge_yaw_abs_error_sum": 0.0,
        "_yaw_task_curriculum_edge_tracking_samples": 0,
        "_yaw_task_curriculum_last_success_rate": 0.0,
        "_yaw_task_curriculum_last_support_score": 0.0,
        "_yaw_task_curriculum_last_lift_progress": 0.0,
        "_yaw_task_curriculum_last_balance_score": 0.0,
        "_yaw_task_curriculum_last_yaw_score": 0.0,
        "_yaw_task_curriculum_last_min_base_height": 0.0,
        "_yaw_task_curriculum_last_torso_contact_rate": 0.0,
        "_yaw_task_curriculum_last_batch_success_rate": 0.0,
        "_yaw_task_curriculum_last_window_evaluated": 0,
        "_yaw_task_curriculum_last_window_successes": 0,
        "_yaw_task_curriculum_last_window_yaw_score": 0.0,
        "_yaw_task_curriculum_last_mean_abs_yaw_cmd": 0.0,
        "_yaw_task_curriculum_last_mean_yaw_rate_error": 0.0,
        "_yaw_task_curriculum_last_tracking_ratio": 0.0,
        "_yaw_task_curriculum_last_mean_abs_edge_yaw_cmd": 0.0,
        "_yaw_task_curriculum_last_mean_edge_yaw_rate_error": 0.0,
        "_yaw_task_curriculum_last_edge_tracking_ratio": 0.0,
        "_yaw_task_curriculum_last_window_mean_abs_yaw_cmd": 0.0,
        "_yaw_task_curriculum_last_window_mean_yaw_rate_error": 0.0,
        "_yaw_task_curriculum_last_window_tracking_ratio": 0.0,
        "_yaw_task_curriculum_last_window_mean_abs_edge_yaw_cmd": 0.0,
        "_yaw_task_curriculum_last_window_mean_edge_yaw_rate_error": 0.0,
        "_yaw_task_curriculum_last_window_edge_tracking_ratio": 0.0,
        "_yaw_task_curriculum_stage_advanced": 0.0,
        "_yaw_task_curriculum_yaw_limit_advanced": 0.0,
        "_yaw_task_curriculum_last_window_passed": 0.0,
        "_yaw_task_curriculum_last_gate_open_rate": 0.0,
        "_yaw_task_curriculum_last_fail_support": 0.0,
        "_yaw_task_curriculum_last_fail_lift": 0.0,
        "_yaw_task_curriculum_last_fail_balance": 0.0,
        "_yaw_task_curriculum_last_fail_yaw": 0.0,
        "_yaw_task_curriculum_last_fail_base_height": 0.0,
        "_yaw_task_curriculum_last_fail_torso_contact": 0.0,
    }
    for name, default in state_defaults.items():
        if not hasattr(env, name):
            setattr(env, name, default)

    clearance_stage_count = len(clearance_levels)
    yaw_stage_count = len(yaw_rate_levels)
    env._yaw_task_curriculum_stage = max(
        0, min(int(env._yaw_task_curriculum_stage), clearance_stage_count - 1)
    )
    env._yaw_task_curriculum_yaw_stage = max(
        0, min(int(env._yaw_task_curriculum_yaw_stage), yaw_stage_count - 1)
    )

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

        if hasattr(env, "_yaw_support_score_sum"):
            support_sample_count = env._yaw_support_score_samples[completed_env_ids].clamp_min(1)
            support_score = env._yaw_support_score_sum[completed_env_ids] / support_sample_count
        else:
            support_score = torch.zeros_like(episode_duration)
        if hasattr(env, "_yaw_gate_open_sum"):
            gate_sample_count = env._yaw_gate_open_samples[completed_env_ids].clamp_min(1)
            gate_open_rate = env._yaw_gate_open_sum[completed_env_ids] / gate_sample_count
        else:
            gate_open_rate = torch.zeros_like(episode_duration)
        if hasattr(env, "_yaw_lift_min_progress_sum"):
            lift_sample_count = env._yaw_lift_min_progress_samples[completed_env_ids].clamp_min(1)
            lift_progress_score = (
                env._yaw_lift_min_progress_sum[completed_env_ids] / lift_sample_count
            )
        else:
            # The initial reset happens before the reward has ever been evaluated.
            lift_progress_score = torch.zeros_like(support_score)
        balance_score = normalized_score(balance_reward_name)
        yaw_score = normalized_score(yaw_reward_name)
        if hasattr(env, "_yaw_base_height_min"):
            minimum_episode_height = env._yaw_base_height_min[completed_env_ids]
        else:
            minimum_episode_height = torch.zeros_like(support_score)
        torso_contact = env.termination_manager.get_term(torso_contact_termination_name)[completed_env_ids]
        if hasattr(env, "_yaw_command_abs_sum"):
            tracking_samples = env._yaw_tracking_metric_samples[completed_env_ids].sum().clamp_min(1)
            command_abs_sum = env._yaw_command_abs_sum[completed_env_ids].sum()
            yaw_abs_error_sum = env._yaw_rate_abs_error_sum[completed_env_ids].sum()
            mean_abs_yaw_command = command_abs_sum / tracking_samples
            mean_yaw_rate_error = yaw_abs_error_sum / tracking_samples
            tracking_ratio = yaw_tracking_ratio(mean_yaw_rate_error, mean_abs_yaw_command)
        else:
            tracking_samples = torch.tensor(1, device=env.device)
            command_abs_sum = torch.tensor(0.0, device=env.device)
            yaw_abs_error_sum = torch.tensor(0.0, device=env.device)
            mean_abs_yaw_command = torch.tensor(0.0, device=env.device)
            mean_yaw_rate_error = torch.tensor(0.0, device=env.device)
            tracking_ratio = torch.tensor(0.0, device=env.device)
        if hasattr(env, "_yaw_edge_command_abs_sum"):
            edge_tracking_samples = env._yaw_edge_tracking_samples[completed_env_ids].sum()
            edge_command_abs_sum = env._yaw_edge_command_abs_sum[completed_env_ids].sum()
            edge_yaw_abs_error_sum = env._yaw_edge_rate_abs_error_sum[completed_env_ids].sum()
            edge_sample_denominator = edge_tracking_samples.clamp_min(1)
            mean_abs_edge_yaw_command = edge_command_abs_sum / edge_sample_denominator
            mean_edge_yaw_rate_error = edge_yaw_abs_error_sum / edge_sample_denominator
            edge_tracking_ratio = yaw_tracking_ratio(
                mean_edge_yaw_rate_error,
                mean_abs_edge_yaw_command,
            )
        else:
            edge_tracking_samples = torch.tensor(0, device=env.device)
            edge_command_abs_sum = torch.tensor(0.0, device=env.device)
            edge_yaw_abs_error_sum = torch.tensor(0.0, device=env.device)
            mean_abs_edge_yaw_command = torch.tensor(0.0, device=env.device)
            mean_edge_yaw_rate_error = torch.tensor(0.0, device=env.device)
            edge_tracking_ratio = torch.tensor(0.0, device=env.device)
        successful = (
            (support_score >= support_threshold)
            & (lift_progress_score >= lift_progress_threshold)
            & (balance_score >= balance_threshold)
            & (yaw_score >= yaw_threshold)
            & (minimum_episode_height >= minimum_base_height)
            & ~torso_contact
        )

        env._yaw_task_curriculum_last_support_score = float(support_score.mean().item())
        env._yaw_task_curriculum_last_lift_progress = float(lift_progress_score.mean().item())
        env._yaw_task_curriculum_last_balance_score = float(balance_score.mean().item())
        env._yaw_task_curriculum_last_yaw_score = float(yaw_score.mean().item())
        env._yaw_task_curriculum_last_min_base_height = float(minimum_episode_height.mean().item())
        env._yaw_task_curriculum_last_torso_contact_rate = float(torso_contact.float().mean().item())
        env._yaw_task_curriculum_last_batch_success_rate = float(successful.float().mean().item())
        env._yaw_task_curriculum_last_mean_abs_yaw_cmd = float(mean_abs_yaw_command.item())
        env._yaw_task_curriculum_last_mean_yaw_rate_error = float(mean_yaw_rate_error.item())
        env._yaw_task_curriculum_last_tracking_ratio = float(tracking_ratio.item())
        env._yaw_task_curriculum_last_mean_abs_edge_yaw_cmd = float(mean_abs_edge_yaw_command.item())
        env._yaw_task_curriculum_last_mean_edge_yaw_rate_error = float(mean_edge_yaw_rate_error.item())
        env._yaw_task_curriculum_last_edge_tracking_ratio = float(edge_tracking_ratio.item())
        env._yaw_task_curriculum_last_gate_open_rate = float(gate_open_rate.mean().item())
        env._yaw_task_curriculum_last_fail_support = float((support_score < support_threshold).float().mean().item())
        env._yaw_task_curriculum_last_fail_lift = float(
            (lift_progress_score < lift_progress_threshold).float().mean().item()
        )
        env._yaw_task_curriculum_last_fail_balance = float(
            (balance_score < balance_threshold).float().mean().item()
        )
        env._yaw_task_curriculum_last_fail_yaw = float((yaw_score < yaw_threshold).float().mean().item())
        env._yaw_task_curriculum_last_fail_base_height = float(
            (minimum_episode_height < minimum_base_height).float().mean().item()
        )
        env._yaw_task_curriculum_last_fail_torso_contact = float(torso_contact.float().mean().item())

        env._yaw_task_curriculum_evaluated += len(completed_env_ids)
        env._yaw_task_curriculum_successes += int(successful.sum().item())
        env._yaw_task_curriculum_yaw_score_sum += float(yaw_score.sum().item())
        env._yaw_task_curriculum_yaw_score_samples += len(completed_env_ids)
        env._yaw_task_curriculum_command_abs_sum += float(command_abs_sum.item())
        env._yaw_task_curriculum_yaw_abs_error_sum += float(yaw_abs_error_sum.item())
        env._yaw_task_curriculum_tracking_samples += int(tracking_samples.item())
        env._yaw_task_curriculum_edge_command_abs_sum += float(edge_command_abs_sum.item())
        env._yaw_task_curriculum_edge_yaw_abs_error_sum += float(edge_yaw_abs_error_sum.item())
        env._yaw_task_curriculum_edge_tracking_samples += int(edge_tracking_samples.item())

    # Curriculum is evaluated before manager buffers are reset. Clear the
    # custom per-episode metric here so the next episode starts from zero.
    if hasattr(env, "_yaw_lift_min_progress_sum"):
        env._yaw_lift_min_progress_sum[selected_env_ids] = 0.0
        env._yaw_lift_min_progress_samples[selected_env_ids] = 0
    if hasattr(env, "_yaw_support_score_sum"):
        env._yaw_support_score_sum[selected_env_ids] = 0.0
        env._yaw_support_score_samples[selected_env_ids] = 0
    if hasattr(env, "_yaw_gate_open_sum"):
        env._yaw_gate_open_sum[selected_env_ids] = 0.0
        env._yaw_gate_open_samples[selected_env_ids] = 0
    if hasattr(env, "_yaw_command_abs_sum"):
        env._yaw_command_abs_sum[selected_env_ids] = 0.0
        env._yaw_rate_abs_error_sum[selected_env_ids] = 0.0
        env._yaw_tracking_metric_samples[selected_env_ids] = 0
    if hasattr(env, "_yaw_edge_command_abs_sum"):
        env._yaw_edge_command_abs_sum[selected_env_ids] = 0.0
        env._yaw_edge_rate_abs_error_sum[selected_env_ids] = 0.0
        env._yaw_edge_tracking_samples[selected_env_ids] = 0
    if hasattr(env, "_yaw_base_height_min"):
        env._yaw_base_height_min[selected_env_ids] = torch.inf

    env._yaw_task_curriculum_stage_advanced = 0.0
    env._yaw_task_curriculum_yaw_limit_advanced = 0.0
    if env._yaw_task_curriculum_evaluated >= min_evaluated_episodes:
        env._yaw_task_curriculum_last_window_evaluated = env._yaw_task_curriculum_evaluated
        env._yaw_task_curriculum_last_window_successes = env._yaw_task_curriculum_successes
        success_rate = env._yaw_task_curriculum_successes / env._yaw_task_curriculum_evaluated
        window_yaw_score = (
            env._yaw_task_curriculum_yaw_score_sum
            / max(env._yaw_task_curriculum_yaw_score_samples, 1)
        )
        env._yaw_task_curriculum_last_success_rate = success_rate
        env._yaw_task_curriculum_last_window_yaw_score = window_yaw_score
        window_mean_abs_yaw_cmd = (
            env._yaw_task_curriculum_command_abs_sum
            / max(env._yaw_task_curriculum_tracking_samples, 1)
        )
        window_mean_yaw_rate_error = (
            env._yaw_task_curriculum_yaw_abs_error_sum
            / max(env._yaw_task_curriculum_tracking_samples, 1)
        )
        window_tracking_ratio = float(
            yaw_tracking_ratio(
                torch.tensor(window_mean_yaw_rate_error, device=env.device),
                torch.tensor(window_mean_abs_yaw_cmd, device=env.device),
            ).item()
        )
        env._yaw_task_curriculum_last_window_mean_abs_yaw_cmd = window_mean_abs_yaw_cmd
        env._yaw_task_curriculum_last_window_mean_yaw_rate_error = window_mean_yaw_rate_error
        env._yaw_task_curriculum_last_window_tracking_ratio = window_tracking_ratio
        window_mean_abs_edge_yaw_cmd = (
            env._yaw_task_curriculum_edge_command_abs_sum
            / max(env._yaw_task_curriculum_edge_tracking_samples, 1)
        )
        window_mean_edge_yaw_rate_error = (
            env._yaw_task_curriculum_edge_yaw_abs_error_sum
            / max(env._yaw_task_curriculum_edge_tracking_samples, 1)
        )
        window_edge_tracking_ratio = float(
            yaw_tracking_ratio(
                torch.tensor(window_mean_edge_yaw_rate_error, device=env.device),
                torch.tensor(window_mean_abs_edge_yaw_cmd, device=env.device),
            ).item()
        )
        env._yaw_task_curriculum_last_window_mean_abs_edge_yaw_cmd = window_mean_abs_edge_yaw_cmd
        env._yaw_task_curriculum_last_window_mean_edge_yaw_rate_error = window_mean_edge_yaw_rate_error
        env._yaw_task_curriculum_last_window_edge_tracking_ratio = window_edge_tracking_ratio

        clearance_stage = env._yaw_task_curriculum_stage
        yaw_stage = env._yaw_task_curriculum_yaw_stage
        clearance_complete = clearance_stage == clearance_stage_count - 1
        if clearance_complete:
            window_passed = yaw_curriculum_window_passes(
                success_rate,
                window_tracking_ratio,
                window_edge_tracking_ratio,
                required_success_rate,
                tracking_ratio_thresholds[yaw_stage],
                edge_tracking_ratio_thresholds[yaw_stage],
            )
            required_stage_steps = min_yaw_stage_steps
        else:
            window_passed = success_rate >= required_success_rate
            required_stage_steps = min_clearance_stage_steps

        env._yaw_task_curriculum_last_window_passed = float(window_passed)
        if window_passed:
            env._yaw_task_curriculum_consecutive_passes = min(
                env._yaw_task_curriculum_consecutive_passes + 1,
                required_consecutive_windows,
            )
        else:
            env._yaw_task_curriculum_consecutive_passes = 0

        stage_steps = current_step - int(env._yaw_task_curriculum_stage_start_step)
        can_promote = (
            env._yaw_task_curriculum_consecutive_passes >= required_consecutive_windows
            and stage_steps >= required_stage_steps
        )
        if can_promote and not clearance_complete:
            env._yaw_task_curriculum_stage += 1
            env._yaw_task_curriculum_stage_advanced = 1.0
            env._yaw_task_curriculum_consecutive_passes = 0
            env._yaw_task_curriculum_stage_start_step = current_step
        elif can_promote and yaw_stage < yaw_stage_count - 1:
            env._yaw_task_curriculum_yaw_stage += 1
            env._yaw_task_curriculum_yaw_limit_advanced = 1.0
            env._yaw_task_curriculum_consecutive_passes = 0
            env._yaw_task_curriculum_stage_start_step = current_step
        env._yaw_task_curriculum_evaluated = 0
        env._yaw_task_curriculum_successes = 0
        env._yaw_task_curriculum_yaw_score_sum = 0.0
        env._yaw_task_curriculum_yaw_score_samples = 0
        env._yaw_task_curriculum_command_abs_sum = 0.0
        env._yaw_task_curriculum_yaw_abs_error_sum = 0.0
        env._yaw_task_curriculum_tracking_samples = 0
        env._yaw_task_curriculum_edge_command_abs_sum = 0.0
        env._yaw_task_curriculum_edge_yaw_abs_error_sum = 0.0
        env._yaw_task_curriculum_edge_tracking_samples = 0

    clearance_stage = env._yaw_task_curriculum_stage
    yaw_stage = env._yaw_task_curriculum_yaw_stage
    yaw_limit = float(yaw_rate_levels[yaw_stage])
    target_clearance = float(clearance_levels[clearance_stage])
    dr_scale = float(dr_scale_levels[yaw_stage])
    tracking_ratio_threshold = float(tracking_ratio_thresholds[yaw_stage])
    edge_tracking_ratio_threshold = float(edge_tracking_ratio_thresholds[yaw_stage])
    stage_steps = current_step - int(env._yaw_task_curriculum_stage_start_step)

    # Command curriculum.
    command_term = env.command_manager.get_term(command_name)
    command_term.cfg.yaw_rate_range = (-yaw_limit, yaw_limit)

    # Clearance curriculum: both the signed shaping reward and the yaw gate must
    # use the same target at every stage.
    for reward_name in (lift_reward_name, yaw_reward_name):
        reward_cfg = env.reward_manager.get_term_cfg(reward_name)
        reward_cfg.params["target_clearance"] = target_clearance
        env.reward_manager.set_term_cfg(reward_name, reward_cfg)

    # Reset/interval DR progresses with yaw. Startup-only material/mass/inertia
    # and CoM randomization remains unchanged.
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

    def log_scalar(value: float | int) -> torch.Tensor:
        return torch.tensor(float(value), device=env.device)

    # CurriculumManager prefixes these keys with ``Curriculum/task_levels/``.
    # RSL-RL forwards them unchanged to TensorBoard and its iteration terminal log.
    return {
        # Keep ``stage`` as the legacy clearance-stage curve.
        "stage": log_scalar(clearance_stage),
        "clearance_stage": log_scalar(clearance_stage),
        "yaw_stage": log_scalar(yaw_stage),
        "yaw_limit": log_scalar(yaw_limit),
        "target_clearance": log_scalar(target_clearance),
        "dr_scale": log_scalar(dr_scale),
        "online_dr_scale": log_scalar(dr_scale),
        "stage_steps": log_scalar(stage_steps),
        "consecutive_pass_windows": log_scalar(env._yaw_task_curriculum_consecutive_passes),
        "support_score": log_scalar(env._yaw_task_curriculum_last_support_score),
        "lift_min_progress": log_scalar(env._yaw_task_curriculum_last_lift_progress),
        "balance_score": log_scalar(env._yaw_task_curriculum_last_balance_score),
        "yaw_score": log_scalar(env._yaw_task_curriculum_last_yaw_score),
        "mean_abs_yaw_cmd": log_scalar(env._yaw_task_curriculum_last_mean_abs_yaw_cmd),
        "mean_error_yaw_rate": log_scalar(env._yaw_task_curriculum_last_mean_yaw_rate_error),
        "tracking_ratio": log_scalar(env._yaw_task_curriculum_last_tracking_ratio),
        "edge_tracking_ratio": log_scalar(env._yaw_task_curriculum_last_edge_tracking_ratio),
        "mean_abs_edge_yaw_cmd": log_scalar(env._yaw_task_curriculum_last_mean_abs_edge_yaw_cmd),
        "mean_edge_error_yaw_rate": log_scalar(env._yaw_task_curriculum_last_mean_edge_yaw_rate_error),
        "minimum_episode_base_height": log_scalar(env._yaw_task_curriculum_last_min_base_height),
        "torso_contact_rate": log_scalar(env._yaw_task_curriculum_last_torso_contact_rate),
        "gate_open_rate": log_scalar(env._yaw_task_curriculum_last_gate_open_rate),
        "fail_reason/support": log_scalar(env._yaw_task_curriculum_last_fail_support),
        "fail_reason/lift": log_scalar(env._yaw_task_curriculum_last_fail_lift),
        "fail_reason/balance": log_scalar(env._yaw_task_curriculum_last_fail_balance),
        "fail_reason/yaw": log_scalar(env._yaw_task_curriculum_last_fail_yaw),
        "fail_reason/base_height": log_scalar(env._yaw_task_curriculum_last_fail_base_height),
        "fail_reason/torso_contact": log_scalar(env._yaw_task_curriculum_last_fail_torso_contact),
        "batch_success_rate": log_scalar(env._yaw_task_curriculum_last_batch_success_rate),
        "window_success_rate": log_scalar(env._yaw_task_curriculum_last_success_rate),
        "window_evaluated_episodes": log_scalar(env._yaw_task_curriculum_last_window_evaluated),
        "window_successful_episodes": log_scalar(env._yaw_task_curriculum_last_window_successes),
        "pending_evaluated_episodes": log_scalar(env._yaw_task_curriculum_evaluated),
        "pending_successful_episodes": log_scalar(env._yaw_task_curriculum_successes),
        "stage_advanced": log_scalar(env._yaw_task_curriculum_stage_advanced),
        "yaw_limit_advanced": log_scalar(env._yaw_task_curriculum_yaw_limit_advanced),
        "window_passed": log_scalar(env._yaw_task_curriculum_last_window_passed),
        "window_yaw_score": log_scalar(env._yaw_task_curriculum_last_window_yaw_score),
        "window_mean_abs_yaw_cmd": log_scalar(env._yaw_task_curriculum_last_window_mean_abs_yaw_cmd),
        "window_mean_error_yaw_rate": log_scalar(env._yaw_task_curriculum_last_window_mean_yaw_rate_error),
        "window_tracking_ratio": log_scalar(env._yaw_task_curriculum_last_window_tracking_ratio),
        "window_mean_abs_edge_yaw_cmd": log_scalar(
            env._yaw_task_curriculum_last_window_mean_abs_edge_yaw_cmd
        ),
        "window_mean_edge_error_yaw_rate": log_scalar(
            env._yaw_task_curriculum_last_window_mean_edge_yaw_rate_error
        ),
        "window_edge_tracking_ratio": log_scalar(
            env._yaw_task_curriculum_last_window_edge_tracking_ratio
        ),
        "support_threshold": log_scalar(support_threshold),
        "lift_progress_threshold": log_scalar(lift_progress_threshold),
        "balance_threshold": log_scalar(balance_threshold),
        "yaw_threshold": log_scalar(yaw_threshold),
        "tracking_ratio_threshold": log_scalar(tracking_ratio_threshold),
        "edge_tracking_ratio_threshold": log_scalar(edge_tracking_ratio_threshold),
        "minimum_base_height": log_scalar(minimum_base_height),
        "required_success_rate": log_scalar(required_success_rate),
        "required_consecutive_windows": log_scalar(required_consecutive_windows),
    }
