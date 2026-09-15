# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal flat-ground environment for balancing VQR on FL and HR wheels."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import rl_training.tasks.manager_based.locomotion.pivot.mdp as mdp

from .robot_cfg import (
    ARTICULATION_JOINT_NAMES,
    BASE_LINK_NAME,
    LEG_JOINT_NAMES,
    LIFTED_WHEEL_NAMES,
    SUPPORT_WHEEL_NAMES,
    VQR_CFG,
    WHEEL_JOINT_NAMES,
)


# The support legs retain the audited standing geometry.  The opposite diagonal
# is folded enough to put each wheel bottom about 0.10 m above the plane.
NOMINAL_LEG_POSITIONS = [
    0.0,
    0.0,
    0.0,
    0.0,
    -0.65,
    -1.00,
    -1.00,
    -0.65,
    1.30,
    2.00,
    2.00,
    1.30,
]
NOMINAL_JOINT_POSITIONS = [*NOMINAL_LEG_POSITIONS, *([0.0] * len(WHEEL_JOINT_NAMES))]
NOMINAL_LEG_POSITION_MAP = dict(zip(LEG_JOINT_NAMES, NOMINAL_LEG_POSITIONS, strict=True))

NOMINAL_ROOT_HEIGHT = 0.450
NOMINAL_ROLL = 0.0
NOMINAL_PITCH = 0.0
WHEEL_RADIUS = 0.091


@configclass
class TwoWheelBalanceSceneCfg(InteractiveSceneCfg):
    """Flat plane, the audited VQR articulation, and one contact sensor."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = VQR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class ActionsCfg:
    """Twelve leg position residuals followed by four wheel velocities."""

    leg_positions = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.18,
        offset=NOMINAL_LEG_POSITION_MAP,
        use_default_offset=False,
        preserve_order=True,
    )
    wheel_velocities = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=4.0,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Deployable policy observations and a separate privileged critic group."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        leg_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        leg_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.10, n_max=0.10),
            scale=0.05,
        )
        wheel_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.10, n_max=0.10),
            scale=0.05,
        )
        previous_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        leg_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
        )
        leg_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
            scale=0.05,
        )
        wheel_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True)},
            scale=0.05,
        )
        previous_action = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        support_contact = ObsTerm(
            func=mdp.wheel_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
                ),
                "threshold": 1.0,
            },
        )
        lifted_contact = ObsTerm(
            func=mdp.wheel_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
                ),
                "threshold": 1.0,
            },
        )
        lifted_clearance = ObsTerm(
            func=mdp.wheel_clearance,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True),
                "wheel_radius": WHEEL_RADIUS,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Only the dedicated low-noise two-wheel reset is enabled for M1."""

    reset_two_wheel = EventTerm(
        func=mdp.reset_two_wheel_diagonal,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ARTICULATION_JOINT_NAMES, preserve_order=True
            ),
            "nominal_joint_positions": NOMINAL_JOINT_POSITIONS,
            "leg_joint_count": len(LEG_JOINT_NAMES),
            "root_height": NOMINAL_ROOT_HEIGHT,
            "nominal_roll": NOMINAL_ROLL,
            "nominal_pitch": NOMINAL_PITCH,
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
class RewardsCfg:
    """Seven task-specific rewards; no locomotion reward set is inherited."""

    support_contact = RewTerm(
        func=mdp.pivot_support_contact,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
            ),
            "threshold": 1.0,
        },
    )
    lifted_diagonal = RewTerm(
        func=mdp.pivot_lifted_wheels,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True),
            "wheel_radius": WHEEL_RADIUS,
            "minimum_clearance": 0.05,
            "threshold": 1.0,
        },
    )
    balance = RewTerm(
        func=mdp.pivot_balance,
        weight=2.0,
        params={
            "nominal_roll": NOMINAL_ROLL,
            "nominal_pitch": NOMINAL_PITCH,
            "std": 0.25,
        },
    )
    angular_stability = RewTerm(func=mdp.pivot_angular_stability, weight=-0.10)
    planar_drift = RewTerm(func=mdp.pivot_planar_drift, weight=-0.50)
    action_rate = RewTerm(func=mdp.pivot_action_rate_l2, weight=-0.01)
    leg_effort = RewTerm(
        func=mdp.pivot_leg_effort_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )


@configclass
class TerminationsCfg:
    """Failures that are clearly outside the recoverable balance region."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.pivot_base_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[BASE_LINK_NAME]),
            "threshold": 1.0,
        },
    )
    inverted = DoneTerm(func=mdp.pivot_inverted)
    tilt_limit = DoneTerm(
        func=mdp.pivot_tilt_limit,
        params={
            "nominal_roll": NOMINAL_ROLL,
            "nominal_pitch": NOMINAL_PITCH,
            "roll_limit": 1.05,
            "pitch_limit": 1.05,
        },
    )
    base_height = DoneTerm(func=mdp.pivot_base_height_below, params={"minimum_height": 0.22})
    drift = DoneTerm(func=mdp.pivot_drifted_away, params={"maximum_distance": 1.0})


@configclass
class VQRTwoWheelBalanceEnvCfg(ManagerBasedRLEnvCfg):
    """M1 balance task configuration."""

    decimation = 4
    episode_length_s = 10.0
    sim = sim_utils.SimulationCfg(dt=0.005, render_interval=decimation, device="cuda:0")
    scene = TwoWheelBalanceSceneCfg(num_envs=16, env_spacing=2.5)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = None
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    curriculum = None

    def __post_init__(self):
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.physx.gpu_max_rigid_patch_count = 2**19
