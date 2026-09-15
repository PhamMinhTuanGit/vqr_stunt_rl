# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""M3 rotation task starting from the audited four-wheel standing pose."""

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.pivot.mdp as mdp
import rl_training.tasks.manager_based.locomotion.pivot.mdp.actions as m3_actions

from .balance_env_cfg import (
    ActionsCfg,
    EventCfg,
    NOMINAL_PITCH,
    NOMINAL_ROLL,
    TerminationsCfg,
    TwoWheelBalanceSceneCfg,
    WHEEL_RADIUS,
)
from .robot_cfg import (
    ARTICULATION_JOINT_NAMES,
    DEFAULT_JOINT_POS,
    LEG_JOINT_NAMES,
    LIFTED_WHEEL_NAMES,
    SUPPORT_WHEEL_NAMES,
    VQR_CFG,
    WHEEL_BODY_NAMES,
)
from .rotate_env_cfg import RotateObservationsCfg, VQRTwoWheelRotateEnvCfg, YawRateCommandsCfg


STANDING_ROOT_HEIGHT = VQR_CFG.init_state.pos[2]
STANDING_JOINT_POSITIONS = [DEFAULT_JOINT_POS[name] for name in ARTICULATION_JOINT_NAMES]
STANDING_LEG_POSITION_MAP = {name: DEFAULT_JOINT_POS[name] for name in LEG_JOINT_NAMES}
LIFTED_HIP_Y_JOINT_NAMES = [LEG_JOINT_NAMES[5], LEG_JOINT_NAMES[6]]
LIFTED_KNEE_JOINT_NAMES = [LEG_JOINT_NAMES[9], LEG_JOINT_NAMES[10]]
M3_LEG_ACTION_SCALE = {name: 0.18 for name in LEG_JOINT_NAMES}
M3_LEG_ACTION_SCALE.update({name: 0.40 for name in LIFTED_HIP_Y_JOINT_NAMES})
M3_LEG_ACTION_SCALE.update({name: 0.75 for name in LIFTED_KNEE_JOINT_NAMES})
_M1_SCENE_TEMPLATE = TwoWheelBalanceSceneCfg(num_envs=1, env_spacing=2.5)


def _place_term_after(group, term_name: str, preceding_term_name: str):
    """Keep the documented command ordering in a derived observation group."""
    term = group.__dict__.pop(term_name)
    ordered_items = []
    for name, value in group.__dict__.items():
        ordered_items.append((name, value))
        if name == preceding_term_name:
            ordered_items.append((term_name, term))
    group.__dict__.clear()
    group.__dict__.update(ordered_items)


@configclass
class FourWheelStartSceneCfg(TwoWheelBalanceSceneCfg):
    """Use Isaac Lab's local procedural plane without a remote USD dependency."""

    terrain = _M1_SCENE_TEMPLATE.terrain.replace(
        terrain_type="generator",
        terrain_generator=terrain_gen.TerrainGeneratorCfg(
            seed=0,
            size=(100.0, 100.0),
            border_width=0.0,
            num_rows=1,
            num_cols=1,
            use_cache=False,
            sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
        ),
        use_terrain_origins=False,
    )


@configclass
class FourWheelStartEventCfg(EventCfg):
    """Replace the M1 diagonal reset with a low-noise four-wheel reset."""

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
            "root_height": STANDING_ROOT_HEIGHT,
            "joint_position_noise": (-0.015, 0.015),
            "leg_velocity_noise": (-0.03, 0.03),
            "wheel_velocity_noise": (-0.10, 0.10),
            "roll_noise": (-0.01, 0.01),
            "pitch_noise": (-0.01, 0.01),
            "yaw_range": (-math.pi, math.pi),
            "angular_velocity_noise": (-0.05, 0.05),
            "root_xy_noise": (-0.01, 0.01),
        },
    )


@configclass
class FourToTwoActionsCfg(ActionsCfg):
    """Give only the lifted diagonal enough leg range for the transition."""

    leg_positions = m3_actions.SoftLimitJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=M3_LEG_ACTION_SCALE,
        offset=STANDING_LEG_POSITION_MAP,
        use_default_offset=False,
        preserve_order=True,
        soft_limit_joint_names=LIFTED_KNEE_JOINT_NAMES,
    )


@configclass
class FourToTwoCommandsCfg(YawRateCommandsCfg):
    """M2 yaw command plus the M3 episode-time lift request."""

    lift_cmd = mdp.EpisodeLiftCommandCfg(
        hold_time_s=0.5,
        ramp_end_time_s=2.0,
        wheel_body_names=WHEEL_BODY_NAMES,
        support_wheel_names=SUPPORT_WHEEL_NAMES,
        lifted_wheel_names=LIFTED_WHEEL_NAMES,
        wheel_radius=WHEEL_RADIUS,
        target_clearance=0.05,
    )


@configclass
class FourToTwoObservationsCfg(RotateObservationsCfg):
    """Insert the deployable scalar lift command after the yaw command."""

    @configclass
    class PolicyCfg(RotateObservationsCfg.PolicyCfg):
        lift_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "lift_cmd"},
        )

        def __post_init__(self):
            super().__post_init__()
            _place_term_after(self, "lift_command", "yaw_rate_command")

    @configclass
    class CriticCfg(RotateObservationsCfg.CriticCfg):
        lift_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "lift_cmd"},
        )

        def __post_init__(self):
            super().__post_init__()
            _place_term_after(self, "lift_command", "yaw_rate_command")

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class FourToTwoRewardsCfg:
    """M3 rewards staged by lift progress from four-wheel to diagonal support."""

    four_wheel_support = RewTerm(
        func=mdp.m3_four_wheel_support,
        weight=1.0,
        params={
            "lift_command_name": "lift_cmd",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True
            ),
            "threshold": 1.0,
        },
    )
    support_contact = RewTerm(
        func=mdp.m3_diagonal_support,
        weight=1.0,
        params={
            "lift_command_name": "lift_cmd",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
            ),
            "threshold": 1.0,
        },
    )
    lifted_diagonal = RewTerm(
        func=mdp.m3_lifted_diagonal,
        weight=1.0,
        params={
            "lift_command_name": "lift_cmd",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
            ),
            "wheel_radius": WHEEL_RADIUS,
            "target_clearance": 0.05,
            "threshold": 1.0,
        },
    )
    balance = RewTerm(
        func=mdp.pivot_balance,
        weight=2.0,
        params={"nominal_roll": NOMINAL_ROLL, "nominal_pitch": NOMINAL_PITCH, "std": 0.25},
    )
    yaw_rate_tracking = RewTerm(
        func=mdp.m3_track_yaw_rate,
        weight=1.5,
        params={
            "yaw_command_name": "yaw_rate_cmd",
            "lift_command_name": "lift_cmd",
            "std": 0.30,
        },
    )
    xy_displacement = RewTerm(
        func=mdp.m3_xy_displacement_dead_zone,
        weight=-0.5,
        params={"dead_zone": 0.10},
    )
    planar_velocity = RewTerm(func=mdp.pivot_planar_drift, weight=-0.5)
    angular_stability = RewTerm(func=mdp.pivot_angular_stability, weight=-0.10)
    action_rate = RewTerm(func=mdp.pivot_action_rate_l2, weight=-0.01)
    leg_effort = RewTerm(
        func=mdp.pivot_leg_effort_l2,
        weight=-1.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
            )
        },
    )


@configclass
class FourToTwoTerminationsCfg(TerminationsCfg):
    """Retain M1 safety terms but measure M3 drift from its reset pose."""

    drift = DoneTerm(
        func=mdp.m3_drifted_from_reset,
        params={"maximum_distance": 0.50},
    )


@configclass
class VQRFourToTwoWheelRotateEnvCfg(VQRTwoWheelRotateEnvCfg):
    """M3: learn the transition from four-wheel standing before M2 rotation."""

    scene: FourWheelStartSceneCfg = FourWheelStartSceneCfg(num_envs=16, env_spacing=2.5)
    events: FourWheelStartEventCfg = FourWheelStartEventCfg()
    commands: FourToTwoCommandsCfg = FourToTwoCommandsCfg()
    observations: FourToTwoObservationsCfg = FourToTwoObservationsCfg()
    actions: FourToTwoActionsCfg = FourToTwoActionsCfg()
    rewards: FourToTwoRewardsCfg = FourToTwoRewardsCfg()
    terminations: FourToTwoTerminationsCfg = FourToTwoTerminationsCfg()
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
