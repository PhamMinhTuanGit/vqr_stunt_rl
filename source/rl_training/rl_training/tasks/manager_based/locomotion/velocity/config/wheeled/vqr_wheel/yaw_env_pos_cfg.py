"""Neutral/positive command-conditioned yaw task; no FSM or scripted transition."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.mdp import yaw_pos_commands as pos_commands
from rl_training.tasks.manager_based.locomotion.velocity.mdp import yaw_pos_curriculums as pos_curriculums
from rl_training.tasks.manager_based.locomotion.velocity.mdp import yaw_pos_rewards as pos_rewards
from rl_training.tasks.manager_based.locomotion.velocity.velocity_yaw_env_cfg import CommandsCfg

from .yaw_env_cfg import (
    LEG_JOINT_NAMES,
    LIFT_CLEARANCE_LEVELS,
    MIN_BASE_HEIGHT,
    ONLINE_DR_SCALE_LEVELS,
    SUPPORT_SPAN_MAX,
    SUPPORT_SPAN_MIN,
    SUPPORT_WHEEL_NAMES,
    LIFTED_WHEEL_NAMES,
    WHEEL_NAMES,
    WHEEL_RADIUS,
    YAW_EDGE_TRACKING_RATIO_THRESHOLDS,
    YAW_RATE_LEVELS,
    YAW_REF,
    YAW_TRACKING_RATIO_THRESHOLDS,
    VQRWheelFlatEnvCfg,
    VQRWheelRewardsCfg,
)

YAW_DEADBAND = 0.1
POS_SUPPORT_WHEELS = SUPPORT_WHEEL_NAMES
POS_LIFTED_WHEELS = LIFTED_WHEEL_NAMES
ALL_WHEELS = WHEEL_NAMES


@configclass
class VQRWheelYawPosCommandsCfg(CommandsCfg):
    yaw_rate_cmd = pos_commands.YawPosCommandCfg(
        asset_name="robot", resampling_time_range=(4.0, 6.0),
        yaw_rate_range=(0.0, YAW_RATE_LEVELS[0]), deadband=YAW_DEADBAND,
        neutral_probability=0.30, debug_vis=False,
    )


@configclass
class VQRWheelYawPosRewardsCfg(VQRWheelRewardsCfg):
    """Inherit all baseline safety terms; replace only pose-dependent terms."""

    com_support = RewTerm(
        func=pos_rewards.yaw_pos_com_support, weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=POS_SUPPORT_WHEELS, preserve_order=True),
                "std": 0.08, "command_name": "yaw_rate_cmd", "deadband": YAW_DEADBAND},
    )
    support_span_band = RewTerm(
        func=pos_rewards.yaw_pos_support_span_band_l2, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=POS_SUPPORT_WHEELS, preserve_order=True),
                "minimum_span": SUPPORT_SPAN_MIN, "maximum_span": SUPPORT_SPAN_MAX,
                "std": 0.05, "command_name": "yaw_rate_cmd", "deadband": YAW_DEADBAND},
    )
    lift_clearance = RewTerm(
        func=pos_rewards.yaw_pos_lift_clearance, weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=POS_LIFTED_WHEELS, preserve_order=True),
                "support_sensor_cfg": SceneEntityCfg("contact_forces", body_names=POS_SUPPORT_WHEELS, preserve_order=True),
                "wheel_radius": WHEEL_RADIUS, "target_clearance": LIFT_CLEARANCE_LEVELS[0],
                "command_name": "yaw_rate_cmd", "deadband": YAW_DEADBAND},
    )
    com_inside_segment = RewTerm(
        func=pos_rewards.yaw_pos_com_inside_segment, weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=POS_SUPPORT_WHEELS, preserve_order=True),
                "std": 0.05, "command_name": "yaw_rate_cmd", "deadband": YAW_DEADBAND},
    )
    gated_yaw_tracking = RewTerm(
        func=pos_rewards.yaw_pos_gated_tracking, weight=8.0,
        params={"command_name": "yaw_rate_cmd",
                "support_sensor_cfg": SceneEntityCfg("contact_forces", body_names=POS_SUPPORT_WHEELS, preserve_order=True),
                "lifted_asset_cfg": SceneEntityCfg("robot", body_names=POS_LIFTED_WHEELS, preserve_order=True),
                "wheel_radius": WHEEL_RADIUS, "target_clearance": LIFT_CLEARANCE_LEVELS[0],
                "std": 0.30, "deadband": YAW_DEADBAND, "contact_threshold": 1.0,
                "clearance_gate_floor": 0.25, "edge_command_fraction": 0.80},
    )
    lifted_wheel_spin = RewTerm(
        func=pos_rewards.yaw_pos_lifted_wheel_spin_l2, weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=POS_LIFTED_WHEELS, preserve_order=True),
                "command_name": "yaw_rate_cmd", "deadband": YAW_DEADBAND, "yaw_reference": YAW_REF},
    )
    neutral_landing_progress = RewTerm(
        func=pos_rewards.yaw_pos_neutral_landing_progress, weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=POS_LIFTED_WHEELS, preserve_order=True),
                "wheel_radius": WHEEL_RADIUS, "target_clearance": LIFT_CLEARANCE_LEVELS[0],
                "command_name": "yaw_rate_cmd", "deadband": YAW_DEADBAND},
    )
    four_stand_pose = RewTerm(
        func=pos_rewards.yaw_pos_four_stand_pose, weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
                "command_name": "yaw_rate_cmd", "deadband": YAW_DEADBAND, "std": 0.25},
    )
    four_wheel_contact = RewTerm(
        func=pos_rewards.yaw_pos_four_wheel_contact, weight=3.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=ALL_WHEELS, preserve_order=True),
                "command_name": "yaw_rate_cmd", "deadband": YAW_DEADBAND, "threshold": 1.0},
    )


@configclass
class VQRWheelYawPosCurriculumCfg:
    task_levels = CurrTerm(
        func=pos_curriculums.yaw_pos_task_levels,
        params={"command_name": "yaw_rate_cmd", "clearance_levels": LIFT_CLEARANCE_LEVELS,
                "yaw_rate_levels": YAW_RATE_LEVELS, "dr_scale_levels": ONLINE_DR_SCALE_LEVELS,
                "tracking_ratio_thresholds": YAW_TRACKING_RATIO_THRESHOLDS,
                "edge_tracking_ratio_thresholds": YAW_EDGE_TRACKING_RATIO_THRESHOLDS,
                "lift_reward_name": "lift_clearance", "balance_reward_name": "balance",
                "yaw_reward_name": "gated_yaw_tracking", "transition_reward_name": None,
                "torso_contact_termination_name": "torso_contact", "minimum_base_height": MIN_BASE_HEIGHT,
                "support_threshold": 0.85, "lift_progress_threshold": 0.80,
                "balance_threshold": 0.75, "yaw_threshold": 0.65,
                "min_evaluated_episodes": 2048, "required_success_rate": 0.85,
                "required_consecutive_windows": 3, "min_clearance_stage_steps": 1000,
                "min_yaw_stage_steps": 6000},
    )


@configclass
class VQRWheelFlatEnvPOSCfg(VQRWheelFlatEnvCfg):
    commands: VQRWheelYawPosCommandsCfg = VQRWheelYawPosCommandsCfg()
    rewards: VQRWheelYawPosRewardsCfg = VQRWheelYawPosRewardsCfg()
    curriculum: VQRWheelYawPosCurriculumCfg = VQRWheelYawPosCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # The baseline post-init initializes a symmetric range; this task is positive-only.
        self.commands.yaw_rate_cmd.yaw_rate_range = (0.0, YAW_RATE_LEVELS[0])
