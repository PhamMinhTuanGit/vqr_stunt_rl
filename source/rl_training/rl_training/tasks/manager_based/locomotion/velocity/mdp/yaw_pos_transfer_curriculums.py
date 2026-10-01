"""Base POS progression, certified on POS steps and gated by neutral evidence."""

from __future__ import annotations

import torch

from .curriculums import yaw_task_levels
from .yaw_pos_transfer_rewards import DEADBAND


def yaw_pos_transfer_task_levels(
    env, env_ids, command_name, clearance_levels, yaw_rate_levels, dr_scale_levels,
    tracking_ratio_thresholds, edge_tracking_ratio_thresholds, lift_reward_name,
    balance_reward_name, yaw_reward_name, torso_contact_termination_name,
    minimum_base_height, support_threshold, lift_progress_threshold, balance_threshold,
    yaw_threshold, min_evaluated_episodes, required_success_rate,
    required_consecutive_windows, min_clearance_stage_steps, min_yaw_stage_steps,
    transition_reward_name=None, active_only=True, neutral_planar_speed_threshold=0.10,
):
    if not active_only:
        raise ValueError("POS-Transfer curriculum requires active_only=True.")
    if neutral_planar_speed_threshold <= 0:
        raise ValueError("neutral_planar_speed_threshold must be positive.")
    selected = torch.arange(env.num_envs, device=env.device)[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    def buffer(name):
        return getattr(env, name, torch.zeros(env.num_envs, device=env.device))

    count = buffer("_yaw_pos_neutral_samples")[selected]
    neutral_ids = selected[(env.episode_length_buf[selected] > 0) & (count > 0)]
    neutral_count = buffer("_yaw_pos_neutral_samples")[neutral_ids].clamp_min(1)

    def neutral_mean(name):
        return buffer(name)[neutral_ids] / neutral_count

    neutral_ok = (
        (neutral_mean("_yaw_pos_neutral_four_contact_sum") >= support_threshold)
        & (neutral_mean("_yaw_pos_transfer_neutral_pose_score_sum") >= balance_threshold)
        & (neutral_mean("_yaw_pos_neutral_abs_yaw_rate_sum") <= DEADBAND)
        & (neutral_mean("_yaw_pos_neutral_planar_speed_sum") <= neutral_planar_speed_threshold)
        & (buffer("_yaw_base_height_min")[neutral_ids] >= minimum_base_height)
        & ~env.termination_manager.get_term(torso_contact_termination_name)[neutral_ids]
    )
    prefix = "_yaw_pos_transfer_curriculum_"
    for name in ("neutral_evaluated", "neutral_successes", "last_neutral_success_rate"):
        if not hasattr(env, prefix + name):
            setattr(env, prefix + name, 0)
    env._yaw_pos_transfer_curriculum_neutral_evaluated += len(neutral_ids)
    env._yaw_pos_transfer_curriculum_neutral_successes += int(neutral_ok.sum().item())
    neutral_evaluated = env._yaw_pos_transfer_curriculum_neutral_evaluated
    neutral_rate = env._yaw_pos_transfer_curriculum_neutral_successes / max(neutral_evaluated, 1)
    neutral_passes = neutral_evaluated >= min_evaluated_episodes and neutral_rate >= required_success_rate

    active_count = buffer("_yaw_tracking_metric_samples")[selected]
    positive_episodes = int(((env.episode_length_buf[selected] > 0) & (active_count > 0)).sum().item())
    window_closes = getattr(env, "_yaw_task_curriculum_evaluated", 0) + positive_episodes >= min_evaluated_episodes
    previous_stage = getattr(env, "_yaw_task_curriculum_stage", 0)
    previous_yaw_stage = getattr(env, "_yaw_task_curriculum_yaw_stage", 0)
    previous_start = getattr(env, "_yaw_task_curriculum_stage_start_step", int(env.common_step_counter))

    base_params = dict(
        command_name=command_name, clearance_levels=clearance_levels, yaw_rate_levels=yaw_rate_levels,
        dr_scale_levels=dr_scale_levels, tracking_ratio_thresholds=tracking_ratio_thresholds,
        edge_tracking_ratio_thresholds=edge_tracking_ratio_thresholds, lift_reward_name=lift_reward_name,
        balance_reward_name=balance_reward_name, yaw_reward_name=yaw_reward_name,
        torso_contact_termination_name=torso_contact_termination_name, minimum_base_height=minimum_base_height,
        support_threshold=support_threshold, lift_progress_threshold=lift_progress_threshold,
        balance_threshold=balance_threshold, yaw_threshold=yaw_threshold,
        min_evaluated_episodes=min_evaluated_episodes, required_success_rate=required_success_rate,
        required_consecutive_windows=required_consecutive_windows,
        min_clearance_stage_steps=min_clearance_stage_steps, min_yaw_stage_steps=min_yaw_stage_steps,
        transition_reward_name=transition_reward_name, active_only=True,
    )
    # The base divides balance reward by full episode duration. Substitute a
    # POS-only mean for certification, then restore the actual reward log sum.
    balance_sums = env.reward_manager._episode_sums[balance_reward_name]
    original_balance = balance_sums[selected].clone()
    duration = env.episode_length_buf[selected].float() * env.step_dt
    balance_mean = buffer("_yaw_pos_transfer_balance_sum")[selected] / buffer("_yaw_pos_transfer_balance_samples")[selected].clamp_min(1)
    balance_sums[selected] = balance_mean * duration * env.reward_manager.get_term_cfg(balance_reward_name).weight
    try:
        result = yaw_task_levels(env, env_ids, **base_params)
    finally:
        balance_sums[selected] = original_balance

    if window_closes:
        env._yaw_pos_transfer_curriculum_last_neutral_success_rate = neutral_rate
        if not neutral_passes:
            promoted = env._yaw_task_curriculum_stage != previous_stage or env._yaw_task_curriculum_yaw_stage != previous_yaw_stage
            env._yaw_task_curriculum_stage = previous_stage
            env._yaw_task_curriculum_yaw_stage = previous_yaw_stage
            env._yaw_task_curriculum_stage_start_step = previous_start
            env._yaw_task_curriculum_consecutive_passes = 0
            env._yaw_task_curriculum_last_window_passed = 0.0
            if promoted:
                # Undo all difficulty changes, including reward targets and DR.
                result = yaw_task_levels(env, [], **base_params)
            result["consecutive_pass_windows"] = torch.tensor(0.0, device=env.device)
            result["window_passed"] = torch.tensor(0.0, device=env.device)
        # Keep insufficient neutral evidence until the next positive window.
        if neutral_evaluated >= min_evaluated_episodes:
            env._yaw_pos_transfer_curriculum_neutral_evaluated = 0
            env._yaw_pos_transfer_curriculum_neutral_successes = 0

    landing_cfg = env.reward_manager.get_term_cfg("neutral_landing_progress")
    landing_cfg.params["target_clearance"] = env.reward_manager.get_term_cfg(lift_reward_name).params["target_clearance"]
    env.reward_manager.set_term_cfg("neutral_landing_progress", landing_cfg)
    result.update({
        "positive_success": result["batch_success_rate"],
        "neutral_success": torch.tensor(neutral_rate, device=env.device),
        "neutral_window_success_rate": torch.tensor(float(env._yaw_pos_transfer_curriculum_last_neutral_success_rate), device=env.device),
        "neutral_evaluated_episodes": torch.tensor(float(neutral_evaluated), device=env.device),
        "neutral_window_passed": torch.tensor(float(neutral_passes), device=env.device),
    })
    # Clear only reset rows, retaining incomplete episodes in all other envs.
    for name in tuple(vars(env)):
        if name.startswith(("_yaw_pos_neutral_", "_yaw_pos_transfer_balance_", "_yaw_pos_transfer_neutral_")) and name.endswith(("_sum", "_samples")):
            getattr(env, name)[selected] = 0
    return result
