"""Active-only yaw curriculum with separate neutral diagnostics."""

from __future__ import annotations

from typing import Sequence

import torch

from .curriculums import yaw_task_levels


def yaw_pos_task_levels(
    env, env_ids: Sequence[int], command_name: str, clearance_levels: Sequence[float],
    yaw_rate_levels: Sequence[float], dr_scale_levels: Sequence[float],
    tracking_ratio_thresholds: Sequence[float], edge_tracking_ratio_thresholds: Sequence[float],
    lift_reward_name: str, balance_reward_name: str, yaw_reward_name: str,
    torso_contact_termination_name: str, minimum_base_height: float,
    support_threshold: float, lift_progress_threshold: float, balance_threshold: float,
    yaw_threshold: float, min_evaluated_episodes: int, required_success_rate: float,
    required_consecutive_windows: int, min_clearance_stage_steps: int, min_yaw_stage_steps: int,
    transition_reward_name: str | None = None,
) -> dict[str, torch.Tensor]:
    if isinstance(env_ids, slice):
        selected = torch.arange(env.num_envs, device=env.device)
    else:
        selected = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    def total(name: str) -> torch.Tensor:
        values = getattr(env, name, None)
        return values[selected].sum() if values is not None else torch.zeros((), device=env.device)

    active_count = total("_yaw_tracking_metric_samples")
    neutral_count = total("_yaw_pos_neutral_samples")
    denominator = (active_count + neutral_count).clamp_min(1)
    neutral_denominator = neutral_count.clamp_min(1)
    diagnostics = {
        "mode_positive_fraction": active_count / denominator,
        "mode_neutral_fraction": neutral_count / denominator,
        "neutral_four_contact_rate": total("_yaw_pos_neutral_four_contact_sum") / neutral_denominator,
        "neutral_pose_error": total("_yaw_pos_neutral_pose_error_sum") / neutral_denominator,
        "neutral_abs_yaw_rate": total("_yaw_pos_neutral_abs_yaw_rate_sum") / neutral_denominator,
        "neutral_planar_speed": total("_yaw_pos_neutral_planar_speed_sum") / neutral_denominator,
    }
    result = yaw_task_levels(
        env, env_ids, command_name, clearance_levels, yaw_rate_levels, dr_scale_levels,
        tracking_ratio_thresholds, edge_tracking_ratio_thresholds, lift_reward_name,
        balance_reward_name, yaw_reward_name, torso_contact_termination_name,
        minimum_base_height, support_threshold, lift_progress_threshold, balance_threshold,
        yaw_threshold, min_evaluated_episodes, required_success_rate,
        required_consecutive_windows, min_clearance_stage_steps, min_yaw_stage_steps,
        transition_reward_name=transition_reward_name, active_only=True,
    )
    for name in (
        "_yaw_pos_neutral_four_contact_sum", "_yaw_pos_neutral_four_contact_samples",
        "_yaw_pos_neutral_pose_error_sum", "_yaw_pos_neutral_pose_error_samples",
        "_yaw_pos_neutral_abs_yaw_rate_sum", "_yaw_pos_neutral_abs_yaw_rate_samples",
        "_yaw_pos_neutral_planar_speed_sum", "_yaw_pos_neutral_planar_speed_samples",
    ):
        if hasattr(env, name):
            getattr(env, name)[selected] = 0
    result.update(diagnostics)
    return result
