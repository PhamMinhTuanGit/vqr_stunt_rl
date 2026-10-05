"""Active-only ground-heading yaw certification with separate neutral diagnostics."""

from __future__ import annotations

from typing import Sequence

import torch

from .curriculums import yaw_task_levels
from .yaw_pos_kinematics import POS_DIFFERENTIAL_METRICS, POS_MOTION_METRICS


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
    certify_behavior: bool = False, behavior_pass_threshold: float = 0.90,
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
    heading_count = total("_yaw_pos_heading_support_x_rms_samples").clamp_min(1)
    diagnostics = {
        "mode_positive_fraction": active_count / denominator,
        "mode_neutral_fraction": neutral_count / denominator,
        "neutral_four_contact_rate": total("_yaw_pos_neutral_four_contact_sum") / neutral_denominator,
        "neutral_pose_error": total("_yaw_pos_neutral_pose_error_sum") / neutral_denominator,
        "neutral_abs_yaw_rate": total("_yaw_pos_neutral_abs_yaw_rate_sum") / neutral_denominator,
        "neutral_planar_speed": total("_yaw_pos_neutral_planar_speed_sum") / neutral_denominator,
        "heading_fl_x_from_com_m": total("_yaw_pos_heading_fl_x_from_com_sum") / heading_count,
        "heading_hr_x_from_com_m": total("_yaw_pos_heading_hr_x_from_com_sum") / heading_count,
        "heading_support_x_rms_m": total("_yaw_pos_heading_support_x_rms_sum") / heading_count,
        "heading_support_line_body_y_alignment": (
            total("_yaw_pos_heading_support_line_body_y_alignment_sum") / heading_count
        ),
    }
    geometry_metrics = (
        "support_line_error", "wheel1_line_error", "wheel2_line_error",
        "support_y_separation", "rolling_error", "body_omega_xy_squared",
        *POS_MOTION_METRICS, *POS_DIFFERENTIAL_METRICS,
    )
    for metric in geometry_metrics:
        diagnostics[metric] = total(f"_yaw_pos_{metric}_sum") / total(
            f"_yaw_pos_{metric}_samples"
        ).clamp_min(1)
    certification = {}
    episode_success_mask = None
    if certify_behavior:
        for metric in ("differential_pass", "neutral_hold_pass"):
            passed = getattr(env, f"_yaw_pos_{metric}_sum", torch.zeros(env.num_envs, device=env.device))
            samples = getattr(env, f"_yaw_pos_{metric}_samples", torch.zeros(env.num_envs, device=env.device))
            certification[metric] = (passed, samples, behavior_pass_threshold)
        passed, samples, threshold = certification["differential_pass"]
        episode_success_mask = (samples > 0) & (passed / samples.clamp_min(1) >= threshold)
    # Certification consumes the heading-based score/error accumulators from
    # yaw_pos_gated_tracking; it must not reconstruct yaw from body-Z velocity.
    result = yaw_task_levels(
        env, env_ids, command_name, clearance_levels, yaw_rate_levels, dr_scale_levels,
        tracking_ratio_thresholds, edge_tracking_ratio_thresholds, lift_reward_name,
        balance_reward_name, yaw_reward_name, torso_contact_termination_name,
        minimum_base_height, support_threshold, lift_progress_threshold, balance_threshold,
        yaw_threshold, min_evaluated_episodes, required_success_rate,
        required_consecutive_windows, min_clearance_stage_steps, min_yaw_stage_steps,
        transition_reward_name=transition_reward_name, active_only=True,
        certification_metrics=certification, episode_success_mask=episode_success_mask,
    )
    # This POS experiment retains the same DR stage, forces, gains and reset
    # distribution, with a smaller interval push at DR=1 (and scaled below it).
    push_limit = 0.15 * dr_scale_levels[env._yaw_task_curriculum_yaw_stage]
    push_cfg = env.event_manager.get_term_cfg("randomize_push_robot")
    push_cfg.params["velocity_range"] = {"x": (-push_limit, push_limit), "y": (-push_limit, push_limit)}
    env.event_manager.set_term_cfg("randomize_push_robot", push_cfg)
    # The POS-only landing signal uses the same clearance target as active lift.
    landing_cfg = env.reward_manager.get_term_cfg("neutral_landing_progress")
    landing_cfg.params["target_clearance"] = env.reward_manager.get_term_cfg(lift_reward_name).params["target_clearance"]
    env.reward_manager.set_term_cfg("neutral_landing_progress", landing_cfg)
    for name in (
        "_yaw_pos_neutral_four_contact_sum", "_yaw_pos_neutral_four_contact_samples",
        "_yaw_pos_neutral_pose_error_sum", "_yaw_pos_neutral_pose_error_samples",
        "_yaw_pos_neutral_abs_yaw_rate_sum", "_yaw_pos_neutral_abs_yaw_rate_samples",
        "_yaw_pos_neutral_planar_speed_sum", "_yaw_pos_neutral_planar_speed_samples",
        "_yaw_pos_heading_fl_x_from_com_sum", "_yaw_pos_heading_fl_x_from_com_samples",
        "_yaw_pos_heading_hr_x_from_com_sum", "_yaw_pos_heading_hr_x_from_com_samples",
        "_yaw_pos_heading_support_x_rms_sum", "_yaw_pos_heading_support_x_rms_samples",
        "_yaw_pos_heading_support_line_body_y_alignment_sum",
        "_yaw_pos_heading_support_line_body_y_alignment_samples",
    ):
        if hasattr(env, name):
            getattr(env, name)[selected] = 0
    for metric in geometry_metrics:
        for suffix in ("sum", "samples"):
            name = f"_yaw_pos_{metric}_{suffix}"
            if hasattr(env, name):
                getattr(env, name)[selected] = 0
    # Episode-local dwell and position anchors never survive an environment reset.
    for name in ("_yaw_pos_active_age", "_yaw_pos_neutral_contact_age",
                 "_yaw_pos_neutral_anchored", "_yaw_pos_neutral_anchor"):
        if hasattr(env, name):
            getattr(env, name)[selected] = 0
    result.update(diagnostics)
    # Logging aliases for the existing ground-heading diagnostics.
    for alias, original in (
        ("heading_error", "mean_error_yaw_rate"),
        ("support_contact", "support_score"),
        ("lift_progress", "lift_min_progress"),
    ):
        if original in result:
            result[alias] = result[original]
    result["body_omega_xy"] = diagnostics["body_omega_xy_squared"].clamp_min(0).sqrt()
    return result
