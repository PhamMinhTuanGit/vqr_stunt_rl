# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.velocity_yaw_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    ObservationsCfg,
    LocomotionVelocityRoughEnvCfg,
    TerminationsCfg,
)

##
# Pre-defined configs
##
from rl_training.assets.deeprobotics import VQRWHEEL_CFG  # isort: skip


# Keep the configuration and the reward implementation on the same canonical
# front-first mirror mapping.  ``fsm_mirror.py`` is the single source of truth.
SUPPORT_WHEEL_NAMES = list(mdp.SUPPORT_POS)
LIFTED_WHEEL_NAMES = list(mdp.SWING_POS)
SUPPORT_WHEEL_NAMES_MIRROR = list(mdp.SUPPORT_NEG)
LIFTED_WHEEL_NAMES_MIRROR = list(mdp.SWING_NEG)
WHEEL_NAMES = ["FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL"]
LEG_JOINT_NAMES = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]
WHEEL_RADIUS = 0.091
TARGET_BASE_HEIGHT = 0.49
MIN_BASE_HEIGHT = 0.35
SUPPORT_SPAN_MIN = 0.50
SUPPORT_SPAN_MAX = 0.70
TARGET_LIFT_CLEARANCE = 0.20
LIFT_CLEARANCE_LEVELS = (0.05, 0.10, 0.15, TARGET_LIFT_CLEARANCE)
YAW_RATE_LEVELS = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)
ONLINE_DR_SCALE_LEVELS = (0.30, 0.40, 0.55, 0.70, 0.85, 1.00)
YAW_TRACKING_RATIO_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
YAW_EDGE_TRACKING_RATIO_THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
# Per-direction mean radius while in YAW.  The final 8 cm target matches the
# short-run acceptance criterion; earlier yaw levels use the same physical
# bound so the scale ladder cannot hide circular drift.
FSM_DRIFT_THRESHOLDS = (0.08, 0.08, 0.08, 0.08, 0.08, 0.08)
YAW_REF = 1.00

# Reward-rebalance baseline is stage-dependent. On resume at yaw_limit=0.25,
# a null-yaw policy can start near mean reward 300 because yaw tracking is easy;
# the mean is expected to fall as yaw_limit expands. Compare rewards only at the
# same yaw_limit, never directly with the legacy approximately 345 baseline.
# Startup-only physical-parameter DR remains at its configured range. The
# curriculum scales reset/interval (online) disturbances together with yaw.


@configclass
class VQRWheelActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*_HipX_joint", ".*_HipY_joint", ".*_Knee_joint"], scale=0.25,
        use_default_offset=True, clip=None, preserve_order=True
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[".*_WHEEL"], scale=20.0, use_default_offset=True, clip=None,
        preserve_order=True
    )


@configclass
class VQRWheelRewardsCfg:
    """The explicit reward set for diagonal-support rotate-in-place training."""

    com_support = RewTerm(
        func=mdp.yaw_com_support,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
            ),
            "std": 0.08,
        },
    )
    base_height = RewTerm(
        func=mdp.yaw_base_height_tracking,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": TARGET_BASE_HEIGHT,
            "error_scale": 0.10,
        },
    )
    low_base_height = RewTerm(
        func=mdp.yaw_low_base_height_l1,
        weight=-4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": MIN_BASE_HEIGHT,
            "error_scale": 0.10,
        },
    )
    downward_low_base_velocity = RewTerm(
        func=mdp.yaw_downward_low_base_velocity_l2,
        weight=-8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": MIN_BASE_HEIGHT,
            "height_margin": 0.10,
        },
    )
    support_span_band = RewTerm(
        func=mdp.yaw_support_span_band_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
            ),
            "minimum_span": SUPPORT_SPAN_MIN,
            "maximum_span": SUPPORT_SPAN_MAX,
            "std": 0.05,
        },
    )
    lift_clearance = RewTerm(
        func=mdp.yaw_lift_clearance,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
            ),
            "wheel_radius": WHEEL_RADIUS,
            "target_clearance": LIFT_CLEARANCE_LEVELS[0],
        },
    )
    com_inside_segment = RewTerm(
        func=mdp.yaw_com_inside_support_segment,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
            ),
            "std": 0.05,
        },
    )
    balance = RewTerm(
        func=mdp.yaw_balance,
        weight=2.0,
        params={
            "nominal_roll": 0.0,
            "nominal_pitch": 0.0,
            "std": 0.25,
            "fsm_command_name": "yaw_rate_cmd",
        },
    )
    gated_yaw_tracking = RewTerm(
        func=mdp.yaw_gated_tracking,
        weight=8.0,
        params={
            "command_name": "yaw_rate_cmd",
            "support_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=SUPPORT_WHEEL_NAMES,
                preserve_order=True,
            ),
            "lifted_asset_cfg": SceneEntityCfg(
                "robot",
                body_names=LIFTED_WHEEL_NAMES,
                preserve_order=True,
            ),
            "wheel_radius": WHEEL_RADIUS,
            "target_clearance": LIFT_CLEARANCE_LEVELS[0],
            "std": 0.30,
            "contact_threshold": 1.0,
            "clearance_gate_floor": 0.25,
            "edge_command_fraction": 0.80,
        },
    )
    lateral_slip = RewTerm(
        func=mdp.yaw_lateral_wheel_slip,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=WHEEL_NAMES, preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=WHEEL_NAMES, preserve_order=True
            ),
            "threshold": 1.0,
        },
    )
    rolling_slip = RewTerm(
        func=mdp.yaw_rolling_wheel_slip,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=WHEEL_NAMES, preserve_order=True
            ),
            "body_asset_cfg": SceneEntityCfg(
                "robot", body_names=WHEEL_NAMES, preserve_order=True
            ),
            "joint_asset_cfg": SceneEntityCfg(
                "robot", joint_names=WHEEL_NAMES, preserve_order=True
            ),
            "wheel_radius": WHEEL_RADIUS,
            "command_name": "yaw_rate_cmd",
            "yaw_reference": YAW_REF,
            "threshold": 1.0,
        },
    )
    undesired_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["^(?!(TORSO|.*_WHEEL)$).*"],
                preserve_order=True,
            ),
            "threshold": 1.0,
        },
    )
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
            )
        },
    )
    action_rate = RewTerm(func=mdp.yaw_action_rate_l2, weight=-0.02)
    joint_velocity = RewTerm(
        func=mdp.yaw_joint_velocity_l2,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
            )
        },
    )
    torque = RewTerm(
        func=mdp.yaw_joint_torque_l2,
        weight=-1.0e-4,
        params={
            "command_name": "yaw_rate_cmd",
            "yaw_reference": YAW_REF,
            "minimum_scale": 0.30,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINT_NAMES + WHEEL_NAMES, preserve_order=True
            )
        },
    )
    lifted_wheel_spin = RewTerm(
        func=mdp.yaw_lifted_wheel_spin_l2,
        weight=-0.02,
        params={
            "command_name": "yaw_rate_cmd",
            "yaw_reference": YAW_REF,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LIFTED_WHEEL_NAMES, preserve_order=True
            )
        },
    )
    planar_velocity = RewTerm(func=mdp.yaw_planar_velocity_l2, weight=-1.0)


@configclass
class VQRWheelFSMRewardsCfg:
    """The 22-term reward contract for ``Flat-VQR-Wheel-Yaw-FSM``.

    The first ten terms are the unchanged baseline safety/regularization
    terms.  The next eight terms are state-gated geometry terms, and the last
    four terms provide FSM tracking, potential shaping, drift control, and
    recovery-entry bookkeeping.  POS/NEG geometry is selected through the
    command's ``support_diagonal`` buffer rather than by duplicating reward
    terms.
    """

    # ------------------------------ Group 1: always-on baseline ------------------------------
    balance = RewTerm(
        func=mdp.yaw_balance,
        weight=2.0,
        params={
            "nominal_roll": 0.0,
            "nominal_pitch": 0.0,
            "std": 0.25,
            "fsm_command_name": "yaw_rate_cmd",
        },
    )
    torque = RewTerm(
        func=mdp.yaw_joint_torque_l2,
        weight=-1.0e-4,
        params={
            "command_name": "yaw_rate_cmd",
            "yaw_reference": YAW_REF,
            "minimum_scale": 0.30,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINT_NAMES + WHEEL_NAMES, preserve_order=True
            ),
        },
    )
    action_rate = RewTerm(func=mdp.yaw_action_rate_l2, weight=-0.02)
    joint_velocity = RewTerm(
        func=mdp.yaw_joint_velocity_l2,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
            )
        },
    )
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
            )
        },
    )
    lateral_slip = RewTerm(
        func=mdp.yaw_lateral_wheel_slip,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=WHEEL_NAMES, preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=WHEEL_NAMES, preserve_order=True
            ),
            "threshold": 1.0,
        },
    )
    undesired_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["^(?!(TORSO|.*_WHEEL)$).*"],
                preserve_order=True,
            ),
            "threshold": 1.0,
        },
    )
    planar_velocity = RewTerm(func=mdp.yaw_planar_velocity_l2, weight=-1.0)
    low_base_height = RewTerm(
        func=mdp.yaw_low_base_height_l1,
        weight=-4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": MIN_BASE_HEIGHT,
            "error_scale": 0.10,
        },
    )
    downward_low_base_velocity = RewTerm(
        func=mdp.yaw_downward_low_base_velocity_l2,
        weight=-8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": MIN_BASE_HEIGHT,
            "height_margin": 0.10,
        },
    )

    # ------------------------------ Group 2: FSM-gated geometry ------------------------------
    com_support = RewTerm(
        func=mdp.yaw_com_support,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
            ),
            "asset_cfg_mirror": SceneEntityCfg(
                "robot", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
            ),
            "std": 0.08,
            "fsm_command_name": "yaw_rate_cmd",
        },
    )
    com_inside_segment = RewTerm(
        func=mdp.yaw_com_inside_support_segment,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
            ),
            "asset_cfg_mirror": SceneEntityCfg(
                "robot", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
            ),
            "std": 0.05,
            "fsm_command_name": "yaw_rate_cmd",
        },
    )
    support_span_band = RewTerm(
        func=mdp.yaw_support_span_band_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
            ),
            "asset_cfg_mirror": SceneEntityCfg(
                "robot", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
            ),
            "minimum_span": SUPPORT_SPAN_MIN,
            "maximum_span": SUPPORT_SPAN_MAX,
            "std": 0.05,
            "fsm_command_name": "yaw_rate_cmd",
        },
    )
    lift_clearance = RewTerm(
        func=mdp.yaw_lift_clearance,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
            ),
            "asset_cfg_mirror": SceneEntityCfg(
                "robot", body_names=LIFTED_WHEEL_NAMES_MIRROR, preserve_order=True
            ),
            "wheel_radius": WHEEL_RADIUS,
            "target_clearance": LIFT_CLEARANCE_LEVELS[0],
            "fsm_command_name": "yaw_rate_cmd",
        },
    )
    base_height = RewTerm(
        func=mdp.yaw_base_height_tracking,
        weight=0.49,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": TARGET_BASE_HEIGHT,
            "error_scale": 0.10,
            "fsm_command_name": "yaw_rate_cmd",
        },
    )
    lifted_wheel_spin = RewTerm(
        func=mdp.yaw_lifted_wheel_spin_l2,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LIFTED_WHEEL_NAMES, preserve_order=True
            ),
            "asset_cfg_mirror": SceneEntityCfg(
                "robot", joint_names=LIFTED_WHEEL_NAMES_MIRROR, preserve_order=True
            ),
            "command_name": "yaw_rate_cmd",
            "yaw_reference": YAW_REF,
            "fsm_command_name": "yaw_rate_cmd",
        },
    )
    rolling_slip = RewTerm(
        func=mdp.yaw_rolling_wheel_slip,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
            ),
            "sensor_cfg_mirror": SceneEntityCfg(
                "contact_forces", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
            ),
            "body_asset_cfg": SceneEntityCfg(
                "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
            ),
            "body_asset_cfg_mirror": SceneEntityCfg(
                "robot", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
            ),
            "joint_asset_cfg": SceneEntityCfg(
                "robot", joint_names=SUPPORT_WHEEL_NAMES, preserve_order=True
            ),
            "joint_asset_cfg_mirror": SceneEntityCfg(
                "robot", joint_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
            ),
            "wheel_radius": WHEEL_RADIUS,
            "command_name": "yaw_rate_cmd",
            "yaw_reference": YAW_REF,
            "threshold": 1.0,
            "fsm_command_name": "yaw_rate_cmd",
        },
    )
    four_stand_stability = RewTerm(
        func=mdp.four_stand_stability,
        weight=3.0,
        params={
            "fsm_command_name": "yaw_rate_cmd",
            "target_height": TARGET_BASE_HEIGHT,
            "height_std": 0.08,
            "attitude_std": 0.25,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    return_to_four_landing = RewTerm(
        func=mdp.ReturnToFourLanding,
        weight=20.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
            ),
            "asset_cfg_mirror": SceneEntityCfg(
                "robot", body_names=LIFTED_WHEEL_NAMES_MIRROR, preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
            ),
            "sensor_cfg_mirror": SceneEntityCfg(
                "contact_forces", body_names=LIFTED_WHEEL_NAMES_MIRROR, preserve_order=True
            ),
            "wheel_radius": WHEEL_RADIUS,
            "fsm_command_name": "yaw_rate_cmd",
            "contact_threshold": 1.0,
        },
    )
    four_stand_ready_bonus = RewTerm(
        func=mdp.four_stand_ready_bonus,
        weight=20.0,
        params={"fsm_command_name": "yaw_rate_cmd"},
    )

    # ------------------------------ Group 3: phase/curriculum terms ------------------------------
    fsm_gated_tracking = RewTerm(
        func=mdp.fsm_gated_tracking,
        weight=8.0,
        params={
            "command_name": "yaw_rate_cmd",
            "fsm_command_name": "yaw_rate_cmd",
            "support_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
            ),
            "support_sensor_cfg_mirror": SceneEntityCfg(
                "contact_forces", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
            ),
            "lifted_asset_cfg": SceneEntityCfg(
                "robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
            ),
            "lifted_asset_cfg_mirror": SceneEntityCfg(
                "robot", body_names=LIFTED_WHEEL_NAMES_MIRROR, preserve_order=True
            ),
            "wheel_radius": WHEEL_RADIUS,
            "target_clearance": LIFT_CLEARANCE_LEVELS[0],
            "std": 0.30,
            "contact_threshold": 1.0,
            "clearance_gate_floor": 0.0,
            "edge_command_fraction": 0.80,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    transition_progress = RewTerm(
        func=mdp.TransitionProgress,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
            ),
            "asset_cfg_mirror": SceneEntityCfg(
                "robot", body_names=LIFTED_WHEEL_NAMES_MIRROR, preserve_order=True
            ),
            "wheel_radius": WHEEL_RADIUS,
            "target_clearance": LIFT_CLEARANCE_LEVELS[0],
            "fsm_command_name": "yaw_rate_cmd",
            "gamma": 0.99,
        },
    )
    spin_center_drift = RewTerm(
        func=mdp.spin_center_drift,
        weight=0.0,  # Phase A; curriculum raises this to -0.5/-2.0 in B/C.
        params={
            "fsm_command_name": "yaw_rate_cmd",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    safe_recovery_entry = RewTerm(
        func=mdp.safe_recovery_entry,
        weight=0.0,  # Termination semantics in phases A/B; -2.0 in phase C.
        params={"fsm_command_name": "yaw_rate_cmd"},
    )


@configclass
class VQRWheelYawCurriculumCfg:
    """Lift first, then jointly progress yaw command and online DR."""

    task_levels = CurrTerm(
        func=mdp.yaw_task_levels,
        params={
            "command_name": "yaw_rate_cmd",
            "clearance_levels": LIFT_CLEARANCE_LEVELS,
            "yaw_rate_levels": YAW_RATE_LEVELS,
            "dr_scale_levels": ONLINE_DR_SCALE_LEVELS,
            "tracking_ratio_thresholds": YAW_TRACKING_RATIO_THRESHOLDS,
            "edge_tracking_ratio_thresholds": YAW_EDGE_TRACKING_RATIO_THRESHOLDS,
            "lift_reward_name": "lift_clearance",
            "balance_reward_name": "balance",
            "yaw_reward_name": "gated_yaw_tracking",
            "torso_contact_termination_name": "torso_contact",
            "minimum_base_height": MIN_BASE_HEIGHT,
            "support_threshold": 0.85,
            "lift_progress_threshold": 0.80,
            "balance_threshold": 0.75,
            "yaw_threshold": 0.65,
            "min_evaluated_episodes": 2048,
            "required_success_rate": 0.85,
            "required_consecutive_windows": 3,
            "min_clearance_stage_steps": 1000,
            "min_yaw_stage_steps": 6000,
        },
    )


@configclass
class VQRWheelFSMCurriculumCfg(VQRWheelYawCurriculumCfg):
    """Three-phase FSM curriculum with independent POS/NEG certification."""

    task_levels = CurrTerm(
        func=mdp.yaw_fsm_task_levels,
        params={
            "command_name": "yaw_rate_cmd",
            "clearance_levels": LIFT_CLEARANCE_LEVELS,
            "yaw_rate_levels": YAW_RATE_LEVELS,
            "dr_scale_levels": ONLINE_DR_SCALE_LEVELS,
            "tracking_ratio_thresholds": YAW_TRACKING_RATIO_THRESHOLDS,
            "edge_tracking_ratio_thresholds": YAW_EDGE_TRACKING_RATIO_THRESHOLDS,
            "lift_reward_name": "lift_clearance",
            "balance_reward_name": "balance",
            "yaw_reward_name": "fsm_gated_tracking",
            "transition_reward_name": "transition_progress",
            "spin_center_drift_reward_name": "spin_center_drift",
            "safe_recovery_reward_name": "safe_recovery_entry",
            "torso_contact_termination_name": "torso_contact",
            "minimum_base_height": MIN_BASE_HEIGHT,
            "support_threshold": 0.85,
            "lift_progress_threshold": 0.80,
            "balance_threshold": 0.75,
            "yaw_threshold": 0.65,
            "min_evaluated_episodes": 2048,
            "required_success_rate": 0.85,
            "required_consecutive_windows": 3,
            "min_clearance_stage_steps": 1000,
            "min_yaw_stage_steps": 6000,
            # §7: both directions need their own evidence; one good diagonal
            # must not pull the other across a curriculum boundary.
            "min_directional_episodes": 1024,
            "transition_success_threshold": 0.85,
            "drift_thresholds": FSM_DRIFT_THRESHOLDS,
            # 150 PPO iterations at the standard 24-step rollout.
            "reward_ramp_steps": 3600,
        },
    )


@configclass
class VQRWheelYawTerminationsCfg(TerminationsCfg):
    """Yaw-task failures; wheel contact is intentionally not terminal."""

    torso_contact = DoneTerm(
        func=mdp.TorsoContactWithGrace,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["TORSO"]),
            "threshold": 1.0,
            "grace_period_s": 0.15,
        },
    )


@configclass
class VQRWheelFSMTerminationsCfg(VQRWheelYawTerminationsCfg):
    """Phase-aware unsafe handling plus FSM watchdogs."""

    torso_contact = DoneTerm(
        func=mdp.FSMUnsafeWithGrace,
        params={
            "robot_name": "robot",
            "torso_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["TORSO"], preserve_order=True
            ),
            "threshold": 1.0,
            "grace_period_s": 0.15,
            "minimum_base_height": MIN_BASE_HEIGHT,
            "unsafe_angle_limit": 0.80,
        },
    )

    fsm_transition_timeout = DoneTerm(
        func=mdp.fsm_transition_timeout,
        params={"command_name": "yaw_rate_cmd", "timeout_s": 3.0},
    )
    fsm_return_timeout = DoneTerm(
        func=mdp.fsm_return_timeout,
        params={"command_name": "yaw_rate_cmd", "timeout_s": 2.5},
    )

    swing_contact_timeout = DoneTerm(
        func=mdp.SwingContactTimeout,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=WHEEL_NAMES, preserve_order=True
            ),
            "command_name": "yaw_rate_cmd",
            "threshold": 1.0,
            "timeout_s": 0.20,
        },
    )


@configclass
class VQRWheelRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: VQRWheelActionsCfg = VQRWheelActionsCfg()
    rewards: VQRWheelRewardsCfg = VQRWheelRewardsCfg()
    terminations: VQRWheelYawTerminationsCfg = VQRWheelYawTerminationsCfg()
    curriculum: VQRWheelYawCurriculumCfg = VQRWheelYawCurriculumCfg()

    base_link_name = "TORSO"
    foot_link_name = ".*_WHEEL"

    # fmt: off
    leg_joint_names = [
        "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
        "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
        "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
        "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
    ]
    wheel_joint_names = [
        "FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL",
    ]

    hipx_joint_names = [
        "FL_HipX_joint", "FR_HipX_joint", "HL_HipX_joint", "HR_HipX_joint",
    ]

    hipy_joint_names = [
        "FL_HipY_joint", "FR_HipY_joint", "HL_HipY_joint", "HR_HipY_joint",
    ]

    knee_joint_names = [
        "FL_Knee_joint", "FR_Knee_joint", "HL_Knee_joint", "HR_Knee_joint",
    ]
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = VQRWHEEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # increase observation noise to reduce reliance on precise instantaneous feedback
        # (helps close the sim-to-real gap and reduce high-frequency action jitter)
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.4, n_max=0.4)
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.1, n_max=0.1)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-3.0, n_max=3.0)
        # joint_pos: per-step noise + a per-episode, per-joint constant bias (~encoder/mechanical
        # calibration offset observed on real hardware, up to ~0.1 rad)
        self.observations.policy.joint_pos.noise = NoiseModelWithAdditiveBiasCfg(
            noise_cfg=Unoise(n_min=-0.02, n_max=0.02),
            bias_noise_cfg=Unoise(n_min=-0.1, n_max=0.1),
            sample_bias_per_component=True,
        )

        # ------------------------------Actions------------------------------
        # Wider per-joint residual ranges improve pose discovery.
        self.actions.joint_pos.scale = {
            ".*_HipX_joint": 0.30,
            ".*_HipY_joint": 0.60,
            ".*_Knee_joint": 0.50,
        }
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # # set terrain generation probability to 0 for boxes and stairs
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 1.0
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
        # # scale down the terrains because the robot is small
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_width = 0.8
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.0, 0.05)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.2)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.16)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        # add a flat patch to the terrain mix (proportions are auto-normalized against the others)
        # self.scene.terrain.terrain_generator.sub_terrains["flat"] = MeshPlaneTerrainCfg(proportion=0.1)

        # remove stairs and slopes entirely, keep only flat + light random_rough/boxes
        # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.05)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.05)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        # self.scene.terrain.terrain_generator.sub_terrains["flat"] = MeshPlaneTerrainCfg(proportion=0.1)

        self.events.randomize_rigid_body_material.params["static_friction_range"] = [0.35, 1.5]
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = [0.35, 1.5]
        self.events.randomize_rigid_body_material.params["restitution_range"] = [0.0, 0.7]

        # yaw_task_levels owns reset/interval DR and starts it at yaw stage zero.
        # Startup material/mass/inertia/CoM DR is intentionally unchanged.
        self.events.randomize_apply_external_force_torque.params["force_range"] = (0.0, 0.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (0.0, 0.0)
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (1.0, 1.0)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (1.0, 1.0)
        self.events.randomize_push_robot.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
        }

        # ------------------------------Terminations------------------------------
        # The inherited catch-all term is disabled. The dedicated stateful term
        # terminates TORSO contact after its own reset-relative grace timer.
        self.terminations.illegal_contact = None
        self.terminations.bad_orientation_2 = None

        # ------------------------------Commands------------------------------
        self.commands.yaw_rate_cmd.yaw_rate_range = (-YAW_RATE_LEVELS[0], YAW_RATE_LEVELS[0])


@configclass
class VQRWheelFlatEnvCfg(VQRWheelRoughEnvCfg):
    """Flat-ground VQR wheel task driven by a scalar yaw-rate command."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None


@configclass
class VQRWheelFSMCommandsCfg(CommandsCfg):
    """Use the FSM-aware command while preserving the scalar yaw command."""

    yaw_rate_cmd = mdp.YawFSMCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        yaw_rate_range=(-1.0, 1.0),
        yaw_enter=0.10,
        yaw_exit=0.05,
        yaw_min_dwell=0.20,
        recovery_dwell=0.50,
        support_sensor_cfg=SceneEntityCfg(
            "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
        ),
        support_sensor_cfg_mirror=SceneEntityCfg(
            "contact_forces", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
        ),
        lifted_asset_cfg=SceneEntityCfg(
            "robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
        ),
        lifted_asset_cfg_mirror=SceneEntityCfg(
            "robot", body_names=LIFTED_WHEEL_NAMES_MIRROR, preserve_order=True
        ),
        all_wheel_sensor_cfg=SceneEntityCfg(
            "contact_forces", body_names=WHEEL_NAMES, preserve_order=True
        ),
        torso_sensor_cfg=SceneEntityCfg(
            "contact_forces", body_names=["TORSO"], preserve_order=True
        ),
        target_clearance=LIFT_CLEARANCE_LEVELS[0],
        wheel_radius=WHEEL_RADIUS,
        contact_threshold=1.0,
        clearance_fraction=0.8,
        pose_angle_limit=0.35,
        unsafe_angle_limit=0.80,
        minimum_base_height=MIN_BASE_HEIGHT,
        debug_vis=False,
    )


@configclass
class VQRWheelFSMObservationsCfg(ObservationsCfg):
    """Flat yaw observations with 62D policy and 95D critic inputs."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        fsm_state = ObsTerm(
            func=mdp.fsm_state_one_hot,
            params={"command_name": "yaw_rate_cmd", "num_states": 7},
        )

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        support_wheel_alignment = ObsTerm(
            func=mdp.fsm_support_wheel_alignment,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
                ),
                "asset_cfg_mirror": SceneEntityCfg(
                    "robot", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
                ),
                "command_name": "yaw_rate_cmd",
            },
        )
        com_support_coordinate = ObsTerm(
            func=mdp.fsm_com_support_coordinate,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
                ),
                "asset_cfg_mirror": SceneEntityCfg(
                    "robot", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
                ),
                "command_name": "yaw_rate_cmd",
            },
        )
        fsm_state = ObsTerm(
            func=mdp.fsm_state_one_hot,
            params={"command_name": "yaw_rate_cmd", "num_states": 7},
        )
        fsm_ready_flags = ObsTerm(
            func=mdp.fsm_ready_flags,
            params={"command_name": "yaw_rate_cmd"},
        )
        fsm_state_time = ObsTerm(
            func=mdp.fsm_state_time,
            params={"command_name": "yaw_rate_cmd"},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class VQRWheelFlatEnvFSMCfg(VQRWheelFlatEnvCfg):
    """Flat-VQR-Wheel-Yaw with the FSM, 22-term reward set, and watchdog."""

    commands: VQRWheelFSMCommandsCfg = VQRWheelFSMCommandsCfg()
    observations: VQRWheelFSMObservationsCfg = VQRWheelFSMObservationsCfg()
    rewards: VQRWheelFSMRewardsCfg = VQRWheelFSMRewardsCfg()
    terminations: VQRWheelFSMTerminationsCfg = VQRWheelFSMTerminationsCfg()
    curriculum: VQRWheelFSMCurriculumCfg = VQRWheelFSMCurriculumCfg()
