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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.pivot.mdp as mdp
import rl_training.tasks.manager_based.locomotion.pivot.mdp.actions as m3_actions
from ....mdp import to_transition as transition
from ....mdp.to_reference import load_leg_reference

from .balance_env_cfg import (
    ActionsCfg,
    EventCfg,
    TerminationsCfg,
    TwoWheelBalanceSceneCfg,
)
from .robot_cfg import (
    ARTICULATION_JOINT_NAMES,
    DEFAULT_JOINT_POS,
    LEG_JOINT_NAMES,
    VQR_CFG,
    WHEEL_BODY_NAMES,
)
from .rotate_env_cfg import RotateObservationsCfg, VQRTwoWheelRotateEnvCfg


STANDING_ROOT_HEIGHT = VQR_CFG.init_state.pos[2]
STANDING_JOINT_POSITIONS = [DEFAULT_JOINT_POS[name] for name in ARTICULATION_JOINT_NAMES]
STANDING_LEG_POSITION_MAP = {name: DEFAULT_JOINT_POS[name] for name in LEG_JOINT_NAMES}
# Standing-centered policy actions must reach every TO joint without a moving
# action offset/controller. Read by name; add room for learned dynamic poses.
TO_LEG_POSE = load_leg_reference(LEG_JOINT_NAMES)
M3_LEG_ACTION_SCALE = {
    name: max(0.25, abs(TO_LEG_POSE[name] - DEFAULT_JOINT_POS[name]) + 0.20)
    for name in LEG_JOINT_NAMES
}
_M1_SCENE_TEMPLATE = TwoWheelBalanceSceneCfg(num_envs=1, env_spacing=2.5)


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
        func=transition.reset_transition_standing,
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
    push = EventTerm(func=transition.transition_push, mode="interval", interval_range_s=(5.0, 8.0))


@configclass
class FourToTwoActionsCfg(ActionsCfg):
    """Standing-centered leg actions with room to approach TO and rebalance."""

    leg_positions = m3_actions.SoftLimitJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=M3_LEG_ACTION_SCALE,
        offset=STANDING_LEG_POSITION_MAP,
        use_default_offset=False,
        preserve_order=True,
        soft_limit_joint_names=LEG_JOINT_NAMES,
        joint_limit_margin=0.05,
    )


@configclass
class FourToTwoCommandsCfg:
    """One yaw/transition command; no duplicate periodic phase."""

    pivot = transition.TOPivotCommandCfg(
        leg_joint_names=LEG_JOINT_NAMES,
        wheel_body_names=WHEEL_BODY_NAMES,
        standing_positions=STANDING_LEG_POSITION_MAP,
    )


@configclass
class FourToTwoObservationsCfg(RotateObservationsCfg):
    """Expose yaw/lambda once, retaining deployable actor and privileged critic."""

    @configclass
    class PolicyCfg(RotateObservationsCfg.PolicyCfg):
        yaw_rate_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "pivot"},
        )

    @configclass
    class CriticCfg(RotateObservationsCfg.CriticCfg):
        yaw_rate_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "pivot"},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class FourToTwoRewardsCfg:
    """M3 rewards staged by lift progress from four-wheel to diagonal support."""

    balance = RewTerm(
        func=mdp.pivot_balance,
        weight=2.0,
        params={"nominal_roll": 0.0, "nominal_pitch": 0.0, "std": 0.5},
    )
    yaw_rate_tracking = RewTerm(
        func=transition.transition_yaw_reward,
        weight=1.5,
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
    # Replace the old abrupt lift schedule with a continuous 4->2 preference.
    four_wheel_support = RewTerm(func=transition.scheduled_contact_reward, weight=3.0)
    pose_reference = RewTerm(func=transition.to_pose_reward, weight=1.0)
    survival = RewTerm(func=mdp.is_alive, weight=0.3)
    failure = RewTerm(func=mdp.is_terminated, weight=-2.0)
    leg_velocity = RewTerm(
        func=mdp.joint_vel_l2, weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )


@configclass
class FourToTwoTerminationsCfg(TerminationsCfg):
    """Retain M1 safety terms but measure M3 drift from its reset pose."""

    drift = DoneTerm(
        func=mdp.m3_drifted_from_reset,
        params={"maximum_distance": 0.50},
    )


@configclass
class TransitionCurriculumCfg:
    transition_levels = CurrTerm(func=transition.transition_levels)


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
    curriculum: TransitionCurriculumCfg = TransitionCurriculumCfg()
    episode_length_s = 12.0

    def __post_init__(self):
        super().__post_init__()
