# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.velocity_yaw_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    TerminationsCfg,
)

##
# Pre-defined configs
##
from rl_training.assets.deeprobotics import VQRWHEEL_CFG  # isort: skip


SUPPORT_WHEEL_NAMES = ["FL_WHEEL", "FR_WHEEL"]
LIFTED_WHEEL_NAMES = ["HR_WHEEL", "HL_WHEEL"]
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
YAW_REF = 1.00

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
        params={"nominal_roll": 0.0, "nominal_pitch": 0.0, "std": 0.25},
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