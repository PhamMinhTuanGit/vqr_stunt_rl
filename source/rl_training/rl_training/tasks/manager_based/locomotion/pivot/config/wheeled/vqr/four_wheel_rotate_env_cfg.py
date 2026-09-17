# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal four-wheel in-place yaw-rate tracking task for VQRWheel."""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.pivot.mdp as mdp

from .balance_env_cfg import ActionsCfg, EventCfg
from .robot_cfg import (
    ARTICULATION_JOINT_NAMES,
    DEFAULT_JOINT_POS,
    LEG_JOINT_NAMES,
    VQR_CFG,
    WHEEL_BODY_NAMES,
)
from .rotate_env_cfg import (
    YAW_COMMAND_NAME,
    RotateObservationsCfg,
    VQRTwoWheelRotateEnvCfg,
    YawRateCommandsCfg,
)


STANDING_JOINT_POSITIONS = [DEFAULT_JOINT_POS[name] for name in ARTICULATION_JOINT_NAMES]
STANDING_LEG_POSITION_MAP = {name: DEFAULT_JOINT_POS[name] for name in LEG_JOINT_NAMES}


@configclass
class FourWheelActionsCfg(ActionsCfg):
    """Keep the existing 12-position + 4-velocity actions, centered on standing."""

    leg_positions = ActionsCfg().leg_positions.replace(offset=STANDING_LEG_POSITION_MAP)


@configclass
class FourWheelEventCfg(EventCfg):
    """Reset directly into the existing four-wheel standing pose."""

    reset_two_wheel = None
    reset_four_wheel = EventTerm(
        func=mdp.reset_four_wheel_standing,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ARTICULATION_JOINT_NAMES, preserve_order=True
            ),
            "nominal_joint_positions": STANDING_JOINT_POSITIONS,
            "leg_joint_count": len(LEG_JOINT_NAMES),
            "root_height": VQR_CFG.init_state.pos[2],
            "joint_position_noise": (0.0, 0.0),
            "leg_velocity_noise": (0.0, 0.0),
            "wheel_velocity_noise": (0.0, 0.0),
            "roll_noise": (0.0, 0.0),
            "pitch_noise": (0.0, 0.0),
            "yaw_range": (0.0, 0.0),
            "angular_velocity_noise": (0.0, 0.0),
            "root_xy_noise": (0.0, 0.0),
        },
    )


@configclass
class FourWheelRewardsCfg:
    """Only the four reward terms needed for four-wheel rotation."""

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=3.0,
        params={"command_name": YAW_COMMAND_NAME, "std": math.sqrt(0.5)},
    )
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": YAW_COMMAND_NAME, "std": math.sqrt(0.5)},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lateral_wheel_slip = RewTerm(
        func=mdp.lateral_wheel_slip,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=WHEEL_BODY_NAMES, preserve_order=True
            ),
            "threshold": 2.0,
        },
    )


@configclass
class FourWheelTerminationsCfg:
    """End an episode as soon as four-wheel support is lost."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    lost_wheel_contact = DoneTerm(
        func=mdp.lost_wheel_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True
            ),
            "threshold": 2.0,
            "grace_period_s": 0.1,
        },
    )


@configclass
class VQRFourWheelRotateEnvCfg(VQRTwoWheelRotateEnvCfg):
    """Rotate in place while retaining contact on all four wheels."""

    observations: RotateObservationsCfg = RotateObservationsCfg()
    actions: FourWheelActionsCfg = FourWheelActionsCfg()
    commands: YawRateCommandsCfg = YawRateCommandsCfg()
    rewards: FourWheelRewardsCfg = FourWheelRewardsCfg()
    terminations: FourWheelTerminationsCfg = FourWheelTerminationsCfg()
    events: FourWheelEventCfg = FourWheelEventCfg()
    curriculum = None
