"""Mode masks around the base yaw rewards, with their arithmetic left intact."""

from __future__ import annotations

from functools import wraps

import torch

from . import rewards as base
from .yaw_pos_rewards import _accumulate

DEADBAND = 0.10
COMMAND_NAME = "yaw_rate_cmd"


def mode_masks(env):
    command = env.command_manager.get_command(COMMAND_NAME)[:, 0]
    positive = command > DEADBAND
    return command, positive, ~positive


def _positive_only(function, metric_fields=(), score_metric=None):
    """Preserve the callable's signature for Isaac Lab's manager validation.

    The base lift/tracking functions also update episode statistics. Restore
    neutral rows of those buffers so neutral steps cannot certify POS skill.
    """
    @wraps(function)
    def gated(env, *args, **kwargs):
        _, positive, neutral = mode_masks(env)
        previous = {name: getattr(env, name).clone() for name in metric_fields if hasattr(env, name)}
        value = function(env, *args, **kwargs)
        for name in metric_fields:
            current = getattr(env, name)
            current[neutral] = previous[name][neutral] if name in previous else 0
        if score_metric is not None:
            _accumulate(env, score_metric, value, positive)
            setattr(env, f"{score_metric}_current", value.detach())
        return torch.where(positive, value, 0.0)

    # A distinct, importable name keeps YAML configs faithful to the gating.
    gated.__name__ = f"positive_{function.__name__}"
    gated.__qualname__ = gated.__name__
    gated.__module__ = __name__
    return gated


positive_yaw_com_support = _positive_only(base.yaw_com_support)
positive_yaw_base_height_tracking = _positive_only(base.yaw_base_height_tracking)
positive_yaw_support_span_band_l2 = _positive_only(base.yaw_support_span_band_l2)
positive_yaw_lift_clearance = _positive_only(
    base.yaw_lift_clearance,
    ("_yaw_lift_min_progress_sum", "_yaw_lift_min_progress_samples"),
)
positive_yaw_com_inside_support_segment = _positive_only(base.yaw_com_inside_support_segment)
positive_yaw_balance = _positive_only(base.yaw_balance, score_metric="_yaw_pos_transfer_balance")
positive_yaw_gated_tracking = _positive_only(
    base.yaw_gated_tracking,
    (
        "_yaw_support_score_sum", "_yaw_support_score_samples",
        "_yaw_gate_open_sum", "_yaw_gate_open_samples",
        "_yaw_command_abs_sum", "_yaw_rate_abs_error_sum", "_yaw_tracking_metric_samples",
        "_yaw_edge_command_abs_sum", "_yaw_edge_rate_abs_error_sum", "_yaw_edge_tracking_samples",
    ),
    score_metric="_yaw_active_yaw_score",
)
positive_yaw_lifted_wheel_spin_l2 = _positive_only(base.yaw_lifted_wheel_spin_l2)


def neutral_yaw_tracking(env, std: float, command_name: str, deadband: float):
    """Reward low absolute yaw, independently of the requested deadband value."""
    _, _, neutral = mode_masks(env)
    robot = env.scene["robot"]
    yaw_rate = robot.data.root_ang_vel_b[:, 2].abs()
    planar_speed = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=1)
    score = torch.exp(-yaw_rate.square() / std**2)
    _accumulate(env, "_yaw_pos_neutral_abs_yaw_rate", yaw_rate, neutral)
    _accumulate(env, "_yaw_pos_neutral_planar_speed", planar_speed, neutral)
    _accumulate(env, "_yaw_pos_transfer_neutral_yaw_score", score, neutral)
    env._yaw_pos_neutral_samples = env._yaw_pos_neutral_abs_yaw_rate_samples
    _record_diagnostics(env, neutral)
    return torch.where(neutral, score, 0.0)


def _record_diagnostics(env, neutral):
    """Emit diagnostics each step, including rollouts without episode endings."""
    command, positive, _ = mode_masks(env)
    robot = env.scene["robot"]
    params = env.cfg.curriculum.task_levels.params
    pose_cfg = env.reward_manager.get_term_cfg("four_stand_pose")
    joint_ids = pose_cfg.params["asset_cfg"].joint_ids
    pose_error = (robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]).square().mean(1)
    pose_score = torch.exp(-pose_error / pose_cfg.params["std"]**2)
    _accumulate(env, "_yaw_pos_transfer_neutral_pose_score", pose_score, neutral)
    contact_cfg = env.reward_manager.get_term_cfg("four_wheel_contact").params
    four_contact = base._yaw_wheel_contacts(env, contact_cfg["sensor_cfg"], contact_cfg["threshold"]).all(1).float()
    lift_cfg = env.reward_manager.get_term_cfg("lift_clearance").params
    lift = base._yaw_lift_progress(env, lift_cfg["asset_cfg"], lift_cfg["wheel_radius"], lift_cfg["target_clearance"]).amin(1)
    support = env._yaw_gate_open_current
    balance = env._yaw_pos_transfer_balance_current
    yaw_score = env._yaw_active_yaw_score_current
    yaw_error = env._yaw_rate_abs_error_current
    planar_speed = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=1)
    yaw_rate = robot.data.root_ang_vel_b[:, 2].abs()
    height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    safe = (height >= params["minimum_base_height"]) & ~env.termination_manager.get_term(params["torso_contact_termination_name"])
    positive_success = safe & (support >= params["support_threshold"]) & (lift >= params["lift_progress_threshold"]) & (balance >= params["balance_threshold"]) & (yaw_score >= params["yaw_threshold"])
    neutral_success = safe & four_contact.bool() & (pose_score >= params["balance_threshold"]) & (yaw_rate <= DEADBAND) & (planar_speed <= params["neutral_planar_speed_threshold"])

    def mean(value, mask):
        return torch.where(mask, value, 0.0).sum() / mask.sum().clamp_min(1)

    metrics = {
        "mode_positive_fraction": positive.float().mean(),
        "mode_neutral_fraction": neutral.float().mean(),
        "mean_abs_yaw_cmd": command.abs().mean(),
        "positive_yaw_score": mean(yaw_score, positive),
        "positive_yaw_rate_error": mean(yaw_error, positive),
        "positive_tracking_ratio": torch.where(positive, command.abs() - yaw_error, 0.0).sum() / torch.where(positive, command.abs(), 0.0).sum().clamp_min(1.0e-6),
        "support_score": mean(support, positive),
        "lift_progress": mean(lift, positive),
        "balance_score": mean(balance, positive),
        "positive_success": mean(positive_success.float(), positive),
        "neutral_success": mean(neutral_success.float(), neutral),
        "neutral_four_contact_rate": mean(four_contact, neutral),
        "neutral_pose_error": mean(pose_error, neutral),
        "neutral_abs_yaw_rate": mean(yaw_rate, neutral),
        "neutral_planar_speed": mean(planar_speed, neutral),
    }
    for name in ("torso_contact", "terrain_out_of_bounds", "time_out"):
        metrics[name] = env.termination_manager.get_term(name).float().mean()
    env.extras.setdefault("log", {}).update({f"Transfer/{name}": value for name, value in metrics.items()})
