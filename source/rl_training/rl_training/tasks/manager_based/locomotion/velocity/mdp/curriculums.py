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
    transition_reward_name: str | None = None,
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
    # The FSM predicate must use the same clearance stage as the lift/yaw
    # rewards.  Scalar YawRateCommandCfg has no such field, so the baseline
    # task remains bit-identical through this conditional extension.
    if hasattr(command_term.cfg, "target_clearance"):
        command_term.cfg.target_clearance = target_clearance

    # Clearance curriculum: both the signed shaping reward and the yaw gate must
    # use the same target at every stage.
    reward_names = [lift_reward_name, yaw_reward_name]
    if transition_reward_name is not None:
        reward_names.append(transition_reward_name)
    for reward_name in reward_names:
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


def yaw_fsm_task_levels(
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
    transition_reward_name: str,
    spin_center_drift_reward_name: str = "spin_center_drift",
    safe_recovery_reward_name: str = "safe_recovery_entry",
    min_directional_episodes: int | None = None,
    transition_success_threshold: float | None = None,
    drift_thresholds: Sequence[float] | None = None,
    reward_ramp_steps: int = 3600,
) -> dict[str, torch.Tensor]:
    """Three-phase FSM curriculum with independent POS/NEG gates.

    Phase A certifies lift at every clearance target, phase B certifies that
    each diagonal reaches ``YAW_*`` within the watchdog's three-second
    transition, and phase C climbs the yaw/DR ladder.  A phase can advance
    only when *both* diagonals pass the same evaluation window.  In
    particular, this intentionally never averages POS and NEG scores.

    The FSM rewards own the directional per-episode buffers.  They coexist
    with the ``_yaw_*`` buffers used by :func:`yaw_task_levels`, preserving the
    original task and its checkpoint-export contract.
    """
    del balance_reward_name, torso_contact_termination_name, minimum_base_height, balance_threshold
    del yaw_threshold, edge_tracking_ratio_thresholds
    validate_yaw_curriculum_levels(
        clearance_levels,
        yaw_rate_levels,
        dr_scale_levels,
        tracking_ratio_thresholds,
        # FSM phase C uses signed YAW-state samples instead of legacy edge samples.
        tracking_ratio_thresholds,
    )
    if min_evaluated_episodes <= 0:
        raise ValueError("min_evaluated_episodes must be positive.")
    if required_consecutive_windows <= 0:
        raise ValueError("required_consecutive_windows must be positive.")
    if min_clearance_stage_steps < 0 or min_yaw_stage_steps < 0:
        raise ValueError("Minimum stage durations must be non-negative.")
    if reward_ramp_steps < 0:
        raise ValueError("reward_ramp_steps must be non-negative.")
    if not 0.0 <= required_success_rate <= 1.0:
        raise ValueError("required_success_rate must be in [0, 1].")

    min_directional_episodes = (
        max(1, min_evaluated_episodes // 2)
        if min_directional_episodes is None
        else int(min_directional_episodes)
    )
    if min_directional_episodes <= 0:
        raise ValueError("min_directional_episodes must be positive.")
    transition_success_threshold = (
        required_success_rate
        if transition_success_threshold is None
        else float(transition_success_threshold)
    )
    if not 0.0 <= transition_success_threshold <= 1.0:
        raise ValueError("transition_success_threshold must be in [0, 1].")
    if drift_thresholds is None:
        drift_thresholds = (0.08,) * len(yaw_rate_levels)
    if len(drift_thresholds) != len(yaw_rate_levels) or any(value <= 0.0 for value in drift_thresholds):
        raise ValueError("drift_thresholds must be positive and match yaw_rate_levels.")

    current_step = int(env.common_step_counter)
    defaults = {
        "_yaw_fsm_task_curriculum_phase": 0,  # 0=lift, 1=transition, 2=yaw+DR
        "_yaw_task_curriculum_stage": 0,  # Retained for existing curriculum-state export.
        "_yaw_task_curriculum_yaw_stage": 0,
        "_yaw_task_curriculum_consecutive_passes": 0,
        "_yaw_fsm_task_curriculum_stage_start_step": current_step,
        "_yaw_fsm_task_curriculum_reward_ramp_start_step": current_step,
        "_yaw_fsm_task_curriculum_last_window_passed": 0.0,
        "_yaw_fsm_task_curriculum_last_pos_score": 0.0,
        "_yaw_fsm_task_curriculum_last_neg_score": 0.0,
        "_yaw_fsm_task_curriculum_last_pos_tracking_ratio": 0.0,
        "_yaw_fsm_task_curriculum_last_neg_tracking_ratio": 0.0,
        "_yaw_fsm_task_curriculum_last_pos_drift": 0.0,
        "_yaw_fsm_task_curriculum_last_neg_drift": 0.0,
        "_yaw_fsm_task_curriculum_phase_advanced": 0.0,
        "_yaw_fsm_task_curriculum_yaw_limit_advanced": 0.0,
    }
    for suffix in ("pos", "neg"):
        defaults.update(
            {
                f"_yaw_fsm_task_curriculum_{suffix}_episodes": 0,
                f"_yaw_fsm_task_curriculum_{suffix}_successes": 0,
                f"_yaw_fsm_task_curriculum_{suffix}_lift_sum": 0.0,
                f"_yaw_fsm_task_curriculum_{suffix}_lift_samples": 0,
                f"_yaw_fsm_task_curriculum_{suffix}_command_sum": 0.0,
                f"_yaw_fsm_task_curriculum_{suffix}_error_sum": 0.0,
                f"_yaw_fsm_task_curriculum_{suffix}_tracking_samples": 0,
                f"_yaw_fsm_task_curriculum_{suffix}_drift_sum": 0.0,
                f"_yaw_fsm_task_curriculum_{suffix}_drift_samples": 0,
            }
        )
    for name, default in defaults.items():
        if not hasattr(env, name):
            setattr(env, name, default)
    # Lifetime diagnostic aggregates are deliberately separate from the
    # promotion windows above.  Clearing an evaluation window must not erase
    # what TensorBoard reports about FSM occupancy and recovery behavior.
    telemetry_defaults = {
        "_yaw_fsm_telemetry_state_totals": torch.zeros(7, dtype=torch.long, device=env.device),
        "_yaw_fsm_telemetry_steps": 0,
        "_yaw_fsm_telemetry_switches": 0,
        "_yaw_fsm_telemetry_positive_budget": torch.zeros(7, dtype=torch.float32, device=env.device),
        "_yaw_fsm_telemetry_positive_budget_pos": 0.0,
        "_yaw_fsm_telemetry_positive_budget_neg": 0.0,
    }
    for suffix in ("pos", "neg"):
        telemetry_defaults.update({
            f"_yaw_fsm_telemetry_{suffix}_transition_attempts": 0,
            f"_yaw_fsm_telemetry_{suffix}_transition_successes": 0,
            f"_yaw_fsm_telemetry_{suffix}_yaw_error_sum": 0.0,
            f"_yaw_fsm_telemetry_{suffix}_yaw_samples": 0,
            f"_yaw_fsm_telemetry_{suffix}_drift_10s_sum": 0.0,
            f"_yaw_fsm_telemetry_{suffix}_drift_10s_samples": 0,
            f"_yaw_fsm_telemetry_{suffix}_swing_contact_sum": 0.0,
            f"_yaw_fsm_telemetry_{suffix}_swing_contact_samples": 0,
        })
    for name, default in telemetry_defaults.items():
        if not hasattr(env, name):
            setattr(env, name, default)

    phase = max(0, min(int(env._yaw_fsm_task_curriculum_phase), 2))
    env._yaw_fsm_task_curriculum_phase = phase
    env._yaw_task_curriculum_stage = max(
        0, min(int(env._yaw_task_curriculum_stage), len(clearance_levels) - 1)
    )
    env._yaw_task_curriculum_yaw_stage = max(
        0, min(int(env._yaw_task_curriculum_yaw_stage), len(yaw_rate_levels) - 1)
    )

    if isinstance(env_ids, slice):
        selected_env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        selected_env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    completed_env_ids = selected_env_ids[env.episode_length_buf[selected_env_ids] > 0]

    def episode_buffer(name: str, dtype: torch.dtype) -> torch.Tensor:
        value = getattr(env, name, None)
        if value is None:
            return torch.zeros(len(completed_env_ids), dtype=dtype, device=env.device)
        return torch.as_tensor(value, device=env.device, dtype=dtype)[completed_env_ids]

    if len(completed_env_ids):
        for suffix in ("pos", "neg"):
            yaw_samples = episode_buffer(f"_yaw_fsm_{suffix}_yaw_samples", torch.long)
            has_yaw = yaw_samples > 0
            attempted = episode_buffer(f"_yaw_fsm_{suffix}_transition_attempted", torch.bool)
            transitioned = episode_buffer(f"_yaw_fsm_{suffix}_transition_succeeded", torch.bool)
            lift_sum = episode_buffer(f"_yaw_fsm_{suffix}_lift_sum", torch.float32)
            support_sum = episode_buffer(f"_yaw_fsm_{suffix}_support_sum", torch.float32)
            lift_score = lift_sum / yaw_samples.clamp_min(1)
            support_score = support_sum / yaw_samples.clamp_min(1)

            # A and C evaluate completed episodes that genuinely spent time
            # in their direction's YAW state.  B instead evaluates transition
            # attempts, so a branch cannot inflate its score by never trying.
            evaluated = attempted if phase == 1 else has_yaw
            success = (
                transitioned
                if phase == 1
                else (lift_score >= lift_progress_threshold) & (support_score >= support_threshold)
            )
            env_name = f"_yaw_fsm_task_curriculum_{suffix}"
            setattr(env, f"{env_name}_episodes", getattr(env, f"{env_name}_episodes") + int(evaluated.sum().item()))
            setattr(
                env,
                f"{env_name}_successes",
                getattr(env, f"{env_name}_successes") + int((success & evaluated).sum().item()),
            )
            setattr(env, f"{env_name}_lift_sum", getattr(env, f"{env_name}_lift_sum") + float(lift_sum.sum().item()))
            setattr(env, f"{env_name}_lift_samples", getattr(env, f"{env_name}_lift_samples") + int(yaw_samples.sum().item()))
            setattr(
                env,
                f"{env_name}_command_sum",
                getattr(env, f"{env_name}_command_sum")
                + float(episode_buffer(f"_yaw_fsm_{suffix}_command_abs_sum", torch.float32).sum().item()),
            )
            setattr(
                env,
                f"{env_name}_error_sum",
                getattr(env, f"{env_name}_error_sum")
                + float(episode_buffer(f"_yaw_fsm_{suffix}_yaw_abs_error_sum", torch.float32).sum().item()),
            )
            setattr(
                env,
                f"{env_name}_tracking_samples",
                getattr(env, f"{env_name}_tracking_samples") + int(yaw_samples.sum().item()),
            )
            setattr(
                env,
                f"{env_name}_drift_sum",
                getattr(env, f"{env_name}_drift_sum")
                + float(episode_buffer(f"_yaw_fsm_{suffix}_drift_sum", torch.float32).sum().item()),
            )
            setattr(
                env,
                f"{env_name}_drift_samples",
                getattr(env, f"{env_name}_drift_samples")
                + int(episode_buffer(f"_yaw_fsm_{suffix}_drift_samples", torch.long).sum().item()),
            )

        if hasattr(env, "_yaw_fsm_state_steps"):
            state_steps = env._yaw_fsm_state_steps[completed_env_ids]
            env._yaw_fsm_telemetry_state_totals += state_steps.sum(dim=0)
            env._yaw_fsm_telemetry_steps += int(state_steps.sum().item())
            env._yaw_fsm_telemetry_switches += int(
                env._yaw_fsm_switches[completed_env_ids].sum().item()
            )
        if hasattr(env, "_yaw_fsm_positive_budget"):
            env._yaw_fsm_telemetry_positive_budget += env._yaw_fsm_positive_budget[
                completed_env_ids
            ].sum(dim=0)
            env._yaw_fsm_telemetry_positive_budget_pos += float(
                env._yaw_fsm_positive_budget_pos[completed_env_ids].sum().item()
            )
            env._yaw_fsm_telemetry_positive_budget_neg += float(
                env._yaw_fsm_positive_budget_neg[completed_env_ids].sum().item()
            )
        for suffix in ("pos", "neg"):
            prefix = f"_yaw_fsm_telemetry_{suffix}"
            setattr(env, f"{prefix}_transition_attempts", getattr(env, f"{prefix}_transition_attempts") + int(
                episode_buffer(f"_yaw_fsm_{suffix}_transition_attempted", torch.bool).sum().item()
            ))
            setattr(env, f"{prefix}_transition_successes", getattr(env, f"{prefix}_transition_successes") + int(
                episode_buffer(f"_yaw_fsm_{suffix}_transition_succeeded", torch.bool).sum().item()
            ))
            setattr(env, f"{prefix}_yaw_error_sum", getattr(env, f"{prefix}_yaw_error_sum") + float(
                episode_buffer(f"_yaw_fsm_{suffix}_yaw_abs_error_sum", torch.float32).sum().item()
            ))
            setattr(env, f"{prefix}_yaw_samples", getattr(env, f"{prefix}_yaw_samples") + int(
                episode_buffer(f"_yaw_fsm_{suffix}_tracking_samples", torch.long).sum().item()
            ))
            setattr(env, f"{prefix}_drift_10s_sum", getattr(env, f"{prefix}_drift_10s_sum") + float(
                episode_buffer(f"_yaw_fsm_{suffix}_drift_10s_sum", torch.float32).sum().item()
            ))
            setattr(env, f"{prefix}_drift_10s_samples", getattr(env, f"{prefix}_drift_10s_samples") + int(
                episode_buffer(f"_yaw_fsm_{suffix}_drift_10s_samples", torch.long).sum().item()
            ))
            setattr(env, f"{prefix}_swing_contact_sum", getattr(env, f"{prefix}_swing_contact_sum") + float(
                episode_buffer(f"_yaw_fsm_{suffix}_swing_contact_sum", torch.float32).sum().item()
            ))
            setattr(env, f"{prefix}_swing_contact_samples", getattr(env, f"{prefix}_swing_contact_samples") + int(
                episode_buffer(f"_yaw_fsm_{suffix}_swing_contact_samples", torch.long).sum().item()
            ))

    # These are reward-owned episode buffers.  Curriculum executes before
    # manager reset, hence it must clear them after consuming completed data.
    if hasattr(env, "_yaw_fsm_state_steps"):
        env._yaw_fsm_state_steps[selected_env_ids] = 0
        env._yaw_fsm_switches[selected_env_ids] = 0
        env._yaw_fsm_episode_steps[selected_env_ids] = 0
    if hasattr(env, "_yaw_fsm_positive_budget"):
        env._yaw_fsm_positive_budget[selected_env_ids] = 0.0
        env._yaw_fsm_positive_budget_pos[selected_env_ids] = 0.0
        env._yaw_fsm_positive_budget_neg[selected_env_ids] = 0.0
    for suffix in ("pos", "neg"):
        for name in (
            f"_yaw_fsm_{suffix}_lift_sum",
            f"_yaw_fsm_{suffix}_support_sum",
            f"_yaw_fsm_{suffix}_command_abs_sum",
            f"_yaw_fsm_{suffix}_yaw_abs_error_sum",
            f"_yaw_fsm_{suffix}_drift_sum",
            f"_yaw_fsm_{suffix}_drift_10s_sum",
            f"_yaw_fsm_{suffix}_swing_contact_sum",
        ):
            if hasattr(env, name):
                getattr(env, name)[selected_env_ids] = 0.0
        for name in (
            f"_yaw_fsm_{suffix}_yaw_samples",
            f"_yaw_fsm_{suffix}_support_samples",
            f"_yaw_fsm_{suffix}_tracking_samples",
            f"_yaw_fsm_{suffix}_drift_samples",
            f"_yaw_fsm_{suffix}_drift_10s_samples",
            f"_yaw_fsm_{suffix}_swing_contact_samples",
        ):
            if hasattr(env, name):
                getattr(env, name)[selected_env_ids] = 0
        for name in (
            f"_yaw_fsm_{suffix}_transition_attempted",
            f"_yaw_fsm_{suffix}_transition_succeeded",
        ):
            if hasattr(env, name):
                getattr(env, name)[selected_env_ids] = False

    env._yaw_fsm_task_curriculum_phase_advanced = 0.0
    env._yaw_fsm_task_curriculum_yaw_limit_advanced = 0.0
    pos_episodes = env._yaw_fsm_task_curriculum_pos_episodes
    neg_episodes = env._yaw_fsm_task_curriculum_neg_episodes
    window_ready = pos_episodes >= min_directional_episodes and neg_episodes >= min_directional_episodes
    if window_ready:
        scores: dict[str, float] = {}
        tracking: dict[str, float] = {}
        drift: dict[str, float] = {}
        for suffix in ("pos", "neg"):
            prefix = f"_yaw_fsm_task_curriculum_{suffix}"
            episodes = getattr(env, f"{prefix}_episodes")
            scores[suffix] = getattr(env, f"{prefix}_successes") / max(episodes, 1)
            samples = getattr(env, f"{prefix}_tracking_samples")
            command = getattr(env, f"{prefix}_command_sum") / max(samples, 1)
            error = getattr(env, f"{prefix}_error_sum") / max(samples, 1)
            tracking[suffix] = float(yaw_tracking_ratio(torch.tensor(error), torch.tensor(command)).item())
            drift[suffix] = getattr(env, f"{prefix}_drift_sum") / max(
                getattr(env, f"{prefix}_drift_samples"), 1
            )

        env._yaw_fsm_task_curriculum_last_pos_score = scores["pos"]
        env._yaw_fsm_task_curriculum_last_neg_score = scores["neg"]
        env._yaw_fsm_task_curriculum_last_pos_tracking_ratio = tracking["pos"]
        env._yaw_fsm_task_curriculum_last_neg_tracking_ratio = tracking["neg"]
        env._yaw_fsm_task_curriculum_last_pos_drift = drift["pos"]
        env._yaw_fsm_task_curriculum_last_neg_drift = drift["neg"]

        if phase == 0:
            window_passed = min(scores.values()) >= required_success_rate
            required_steps = min_clearance_stage_steps
        elif phase == 1:
            window_passed = min(scores.values()) >= transition_success_threshold
            required_steps = min_yaw_stage_steps
        else:
            yaw_stage = env._yaw_task_curriculum_yaw_stage
            window_passed = (
                min(tracking.values()) >= tracking_ratio_thresholds[yaw_stage]
                and max(drift.values()) <= drift_thresholds[yaw_stage]
            )
            required_steps = min_yaw_stage_steps
        env._yaw_fsm_task_curriculum_last_window_passed = float(window_passed)
        env._yaw_task_curriculum_consecutive_passes = (
            min(env._yaw_task_curriculum_consecutive_passes + 1, required_consecutive_windows)
            if window_passed
            else 0
        )
        stage_steps = current_step - int(env._yaw_fsm_task_curriculum_stage_start_step)
        if (
            env._yaw_task_curriculum_consecutive_passes >= required_consecutive_windows
            and stage_steps >= required_steps
        ):
            if phase == 0 and env._yaw_task_curriculum_stage < len(clearance_levels) - 1:
                env._yaw_task_curriculum_stage += 1
                env._yaw_fsm_task_curriculum_phase_advanced = 1.0
            elif phase == 0:
                env._yaw_fsm_task_curriculum_phase = 1
                env._yaw_fsm_task_curriculum_phase_advanced = 1.0
            elif phase == 1:
                env._yaw_fsm_task_curriculum_phase = 2
                env._yaw_fsm_task_curriculum_phase_advanced = 1.0
            elif env._yaw_task_curriculum_yaw_stage < len(yaw_rate_levels) - 1:
                env._yaw_task_curriculum_yaw_stage += 1
                env._yaw_fsm_task_curriculum_yaw_limit_advanced = 1.0
            if env._yaw_fsm_task_curriculum_phase_advanced or env._yaw_fsm_task_curriculum_yaw_limit_advanced:
                env._yaw_task_curriculum_consecutive_passes = 0
                env._yaw_fsm_task_curriculum_stage_start_step = current_step
                env._yaw_fsm_task_curriculum_reward_ramp_start_step = current_step

        for suffix in ("pos", "neg"):
            prefix = f"_yaw_fsm_task_curriculum_{suffix}"
            for field in (
                "episodes", "successes", "lift_sum", "lift_samples", "command_sum",
                "error_sum", "tracking_samples", "drift_sum", "drift_samples",
            ):
                setattr(env, f"{prefix}_{field}", 0 if field.endswith("samples") or field in {"episodes", "successes", "tracking_samples"} else 0.0)

    phase = env._yaw_fsm_task_curriculum_phase
    lift_stage = env._yaw_task_curriculum_stage
    yaw_stage = env._yaw_task_curriculum_yaw_stage
    target_clearance = float(clearance_levels[lift_stage if phase == 0 else -1])
    yaw_limit = float(yaw_rate_levels[yaw_stage if phase == 2 else 0])
    dr_scale = float(dr_scale_levels[yaw_stage] if phase == 2 else dr_scale_levels[0])

    command_term = env.command_manager.get_term(command_name)
    command_term.cfg.yaw_rate_range = (-yaw_limit, yaw_limit)
    if hasattr(command_term.cfg, "target_clearance"):
        command_term.cfg.target_clearance = target_clearance
    for reward_name in (lift_reward_name, yaw_reward_name, transition_reward_name):
        reward_cfg = env.reward_manager.get_term_cfg(reward_name)
        reward_cfg.params["target_clearance"] = target_clearance
        env.reward_manager.set_term_cfg(reward_name, reward_cfg)

    # Changes to phase rewards are linearly ramped, avoiding a discontinuous
    # value-target change when B/C opens.  `common_step_counter` advances once
    # per vectorized environment step, independent of how many envs reset.
    phase_weights = ((8.0, 2.0, 0.0, 0.0), (8.0, 2.0, -0.5, 0.0), (8.0, 2.0, -2.0, -2.0))[phase]
    phase_terms = (
        yaw_reward_name,
        transition_reward_name,
        spin_center_drift_reward_name,
        safe_recovery_reward_name,
    )
    ramp = 1.0 if reward_ramp_steps == 0 else min(
        1.0,
        max(0.0, (current_step - env._yaw_fsm_task_curriculum_reward_ramp_start_step) / reward_ramp_steps),
    )
    for reward_name, target_weight in zip(phase_terms, phase_weights):
        reward_cfg = env.reward_manager.get_term_cfg(reward_name)
        start_name = f"_yaw_fsm_task_curriculum_ramp_start_{reward_name}"
        if not hasattr(env, start_name):
            setattr(env, start_name, float(reward_cfg.weight))
        if current_step == env._yaw_fsm_task_curriculum_reward_ramp_start_step:
            setattr(env, start_name, float(reward_cfg.weight))
        reward_cfg.weight = float(getattr(env, start_name) + ramp * (target_weight - getattr(env, start_name)))
        if reward_name == yaw_reward_name:
            reward_cfg.params["clearance_gate_floor"] = 0.25 if phase == 1 else 0.0
            reward_cfg.params["clearance_gate_floor_decay_s"] = 3.0 if phase == 1 else 0.0
        env.reward_manager.set_term_cfg(reward_name, reward_cfg)

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
    push_cfg.params["velocity_range"] = {"x": (-push_limit, push_limit), "y": (-push_limit, push_limit)}
    env.event_manager.set_term_cfg("randomize_push_robot", push_cfg)
    reset_cfg = env.event_manager.get_term_cfg("randomize_reset_base")
    reset_cfg.params["pose_range"].update({"roll": (-0.3 * dr_scale, 0.3 * dr_scale), "pitch": (-0.3 * dr_scale, 0.3 * dr_scale)})
    reset_cfg.params["velocity_range"].update({
        "x": (-0.2 * dr_scale, 0.2 * dr_scale), "y": (-0.2 * dr_scale, 0.2 * dr_scale),
        "z": (-0.2 * dr_scale, 0.2 * dr_scale), "roll": (-0.05 * dr_scale, 0.05 * dr_scale),
        "pitch": (-0.05 * dr_scale, 0.05 * dr_scale),
    })
    env.event_manager.set_term_cfg("randomize_reset_base", reset_cfg)

    def scalar(value: float | int) -> torch.Tensor:
        return torch.tensor(float(value), device=env.device)

    telemetry_steps = max(env._yaw_fsm_telemetry_steps, 1)
    state_fractions = env._yaw_fsm_telemetry_state_totals.to(dtype=torch.float32) / telemetry_steps
    telemetry = {
        f"state_fraction/{state}": state_fractions[state]
        for state in range(7)
    }
    telemetry["switch_rate"] = scalar(env._yaw_fsm_telemetry_switches / telemetry_steps)
    budgets = env._yaw_fsm_telemetry_positive_budget / telemetry_steps
    telemetry.update({f"budget/state/{state}": budgets[state] for state in range(7)})
    telemetry["budget/pos"] = scalar(
        env._yaw_fsm_telemetry_positive_budget_pos / telemetry_steps
    )
    telemetry["budget/neg"] = scalar(
        env._yaw_fsm_telemetry_positive_budget_neg / telemetry_steps
    )
    for suffix in ("pos", "neg"):
        prefix = f"_yaw_fsm_telemetry_{suffix}"
        telemetry[f"{suffix}/transition_success"] = scalar(
            getattr(env, f"{prefix}_transition_successes")
            / max(getattr(env, f"{prefix}_transition_attempts"), 1)
        )
        telemetry[f"{suffix}/yaw_mae"] = scalar(
            getattr(env, f"{prefix}_yaw_error_sum")
            / max(getattr(env, f"{prefix}_yaw_samples"), 1)
        )
        telemetry[f"{suffix}/drift_10s"] = scalar(
            getattr(env, f"{prefix}_drift_10s_sum")
            / max(getattr(env, f"{prefix}_drift_10s_samples"), 1)
        )
        telemetry[f"{suffix}/drift_10s_samples"] = scalar(
            getattr(env, f"{prefix}_drift_10s_samples")
        )
        telemetry[f"{suffix}/swing_contact_rate"] = scalar(
            getattr(env, f"{prefix}_swing_contact_sum")
            / max(getattr(env, f"{prefix}_swing_contact_samples"), 1)
        )

    return {
        "phase": scalar(phase), "lift_stage": scalar(lift_stage), "yaw_stage": scalar(yaw_stage),
        "yaw_limit": scalar(yaw_limit), "target_clearance": scalar(target_clearance),
        "online_dr_scale": scalar(dr_scale), "stage_steps": scalar(current_step - env._yaw_fsm_task_curriculum_stage_start_step),
        "consecutive_pass_windows": scalar(env._yaw_task_curriculum_consecutive_passes),
        "pos/episodes": scalar(env._yaw_fsm_task_curriculum_pos_episodes),
        "neg/episodes": scalar(env._yaw_fsm_task_curriculum_neg_episodes),
        "pos/score": scalar(env._yaw_fsm_task_curriculum_last_pos_score),
        "neg/score": scalar(env._yaw_fsm_task_curriculum_last_neg_score),
        "pos/tracking_ratio": scalar(env._yaw_fsm_task_curriculum_last_pos_tracking_ratio),
        "neg/tracking_ratio": scalar(env._yaw_fsm_task_curriculum_last_neg_tracking_ratio),
        "pos/drift": scalar(env._yaw_fsm_task_curriculum_last_pos_drift),
        "neg/drift": scalar(env._yaw_fsm_task_curriculum_last_neg_drift),
        "window_passed": scalar(env._yaw_fsm_task_curriculum_last_window_passed),
        "phase_advanced": scalar(env._yaw_fsm_task_curriculum_phase_advanced),
        "yaw_limit_advanced": scalar(env._yaw_fsm_task_curriculum_yaw_limit_advanced),
        "required_directional_episodes": scalar(min_directional_episodes),
        **telemetry,
    }
