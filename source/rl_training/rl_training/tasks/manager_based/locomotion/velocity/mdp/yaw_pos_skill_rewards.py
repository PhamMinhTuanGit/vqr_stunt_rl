"""Skill-only active-yaw shaping and unchanged POS neutral reward telemetry."""

import torch
from isaaclab.managers import SceneEntityCfg

from .yaw_pos_rewards import yaw_pos_neutral_position, yaw_pos_masks, _accumulate
from .rewards import _yaw_whole_body_com_xy, _yaw_wheel_contacts
from .yaw_pos_kinematics import differential_rolling_kinematics, ground_heading_axes


SUPPORT_COVERAGE_METRICS = tuple(
    f"support_{wheel}_{quantity}"
    for wheel in ("fl", "hr") for quantity in ("motor_fraction", "ground_speed_coverage")
)


def motor_participation_scores(joint_qdot, target, wheel_radius=0.091):
    """Per-wheel dense scores using the certificate's actual motor fraction."""
    fraction = wheel_radius * joint_qdot.abs() / target.abs().clamp_min(1.e-4)
    return (fraction / 0.5).clamp(0., 1.)


def ground_speed_coverage_scores(measured, target):
    """Per-wheel signed coverage of half the target ground speed."""
    coverage = (measured.abs() / (0.5 * target.abs()).clamp_min(1.e-4)).clamp(0., 1.)
    return torch.where(measured * target > 0., coverage, 0.)


def _support_inputs(env, support_asset_cfg, support_joint_cfg, support_sensor_cfg,
                    command_name, deadband, contact_threshold):
    expected = ("FL_WHEEL", "HR_WHEEL")
    if (tuple(support_asset_cfg.body_names) != expected
            or tuple(support_joint_cfg.joint_names) != expected
            or tuple(support_sensor_cfg.body_names) != expected):
        raise ValueError("Skill coverage requires support bodies, joints and sensors ordered FL, HR.")
    command, active, _ = yaw_pos_masks(env, command_name, deadband)
    robot = env.scene[support_asset_cfg.name]
    data = robot.data
    target, measured, valid = differential_rolling_kinematics(
        data.body_quat_w[:, support_asset_cfg.body_ids],
        data.body_pos_w[:, support_asset_cfg.body_ids],
        data.body_link_lin_vel_w[:, support_asset_cfg.body_ids],
        _yaw_whole_body_com_xy(robot), command,
    )
    qdot = data.joint_vel[:, support_joint_cfg.joint_ids]
    contacts = _yaw_wheel_contacts(env, support_sensor_cfg, contact_threshold)
    if target.shape[1] != 2 or qdot.shape != target.shape or contacts.shape != target.shape:
        raise ValueError("Skill coverage requires exactly two resolved support wheels.")
    _, _, heading_valid = ground_heading_axes(data.root_quat_w)
    valid = valid & heading_valid[:, None] & (target.abs() > 1.e-4)
    eligible = active & (valid & contacts).all(dim=1)
    return target, measured, qdot, eligible


def _coverage_telemetry(env, quantity, values, eligible):
    for index, wheel in enumerate(("fl", "hr")):
        name = f"_yaw_pos_skill_support_{wheel}_{quantity}"
        value = values[:, index].detach()
        _accumulate(env, name, value, eligible)
        minimum_name = name + "_min"
        if not hasattr(env, minimum_name):
            setattr(env, minimum_name, torch.full_like(value, torch.inf))
        minimum = getattr(env, minimum_name)
        minimum.copy_(torch.minimum(minimum, torch.where(eligible, value, torch.inf)))


def skill_motor_participation(
    env, support_asset_cfg: SceneEntityCfg, support_joint_cfg: SceneEntityCfg,
    support_sensor_cfg: SceneEntityCfg, command_name: str, deadband: float,
    wheel_radius: float = 0.091, contact_threshold: float = 1.0,
) -> torch.Tensor:
    target, _, qdot, eligible = _support_inputs(
        env, support_asset_cfg, support_joint_cfg, support_sensor_cfg,
        command_name, deadband, contact_threshold,
    )
    eligible = eligible & torch.isfinite(qdot).all(dim=1)
    fraction = wheel_radius * qdot.abs() / target.abs().clamp_min(1.e-4)
    _coverage_telemetry(env, "motor_fraction", fraction, eligible)
    score = motor_participation_scores(qdot, target, wheel_radius).amin(dim=1)
    return torch.where(eligible, score, 0.)


def skill_ground_speed_coverage(
    env, support_asset_cfg: SceneEntityCfg, support_joint_cfg: SceneEntityCfg,
    support_sensor_cfg: SceneEntityCfg, command_name: str, deadband: float,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    target, measured, _, eligible = _support_inputs(
        env, support_asset_cfg, support_joint_cfg, support_sensor_cfg,
        command_name, deadband, contact_threshold,
    )
    coverage = ground_speed_coverage_scores(measured, target)
    _coverage_telemetry(env, "ground_speed_coverage", coverage, eligible)
    return torch.where(eligible, coverage.amin(dim=1), 0.)


def skill_neutral_position(
    env, sensor_cfg: SceneEntityCfg, command_name: str, deadband: float,
    std: float = 0.05, contact_dwell: float = 0.2, contact_threshold: float = 1.0,
    speed_threshold: float = 0.03, drift_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    value = yaw_pos_neutral_position(
        env, sensor_cfg, command_name, deadband, std, contact_dwell,
        contact_threshold, speed_threshold, drift_threshold, asset_cfg,
    )
    _, _, neutral = yaw_pos_masks(env, command_name, deadband)
    if not hasattr(env, "_yaw_pos_skill_anchor_sum"):
        env._yaw_pos_skill_anchor_sum = torch.zeros(env.num_envs, device=env.device)
        env._yaw_pos_skill_neutral_samples = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._yaw_pos_skill_anchor_sum.add_((neutral & env._yaw_pos_neutral_anchored).float())
    env._yaw_pos_skill_neutral_samples.add_(neutral.long())
    return value
