# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Four-mode pivot environment configuration (spec sections 7-9).

GROUND -> REAR_UP -> BALANCE -> LAND (RECOVER via pi_recover), single policy.
Reward weights follow the section-7 table. Heavy DR/curriculum are disabled
until the ManagerBased baseline is validated.
"""

from __future__ import annotations

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

from .physical_params import VQR_PHYSICS as PHYS
from .robot_cfg import (
    ARTICULATION_JOINT_NAMES,
    BASE_LINK_NAME,
    LEG_JOINT_NAMES,
    VQR_CFG,
    WHEEL_BODY_NAMES,
    WHEEL_JOINT_NAMES,
)

# Mode ids (supervisor_gate / commands).
GROUND = 0
REAR_UP = 1
BALANCE = 2
LAND = 3
FRONT_WHEEL_NAMES = WHEEL_BODY_NAMES[:2]
REAR_SUPPORT_WHEEL_NAMES = WHEEL_BODY_NAMES[2:]


##
# Scene
##


@configclass
class PivotSceneCfg(InteractiveSceneCfg):
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


##
# Actions: 12 leg PD residuals (prior + 0.3*scale) + 4 wheel torques (I2/I3)
##


@configclass
class PivotActionsCfg:
    leg_residuals = mdp.PriorResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
        command_name="pivot_mode",
        residual_scale=0.3,
        enforce_soft_limits=True,
    )
    wheel_torques = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=PHYS.wheel_peak_torque,
        preserve_order=True,
    )


##
# Commands: 7-D mode one-hot(4) | omega_z* | delta_theta* | tuck*
##


@configclass
class PivotCommandsCfg:
    pivot_mode = mdp.PivotModeCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        omega_z_range=(-PHYS.omega_z_limit_balance, PHYS.omega_z_limit_balance),
        omega_z_limit=PHYS.omega_z_limit_balance,
        ground_omega_z_limit=PHYS.omega_z_limit_ground,
        delta_theta_range=(-PHYS.delta_theta_command_limit, PHYS.delta_theta_command_limit),
        delta_theta_limit=PHYS.delta_theta_command_limit,
        rel_standing_envs=0.1,
        ground_duration_s=2.0,
        rear_up_duration_s=3.0,
        balance_duration_s=10.0,
    )


##
# Observations: policy group + privileged critic group
##


@configclass
class PivotPolicyObsCfg(ObsGroup):
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.02, n_max=0.02)
    )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02)
    )
    leg_joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    leg_joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        noise=Unoise(n_min=-0.15, n_max=0.15),
        scale=0.05,
    )
    wheel_joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
        noise=Unoise(n_min=-0.10, n_max=0.10),
        scale=0.05,
    )
    previous_action = ObsTerm(func=mdp.last_action)
    balance_signals = ObsTerm(
        func=mdp.balance_signals,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=REAR_SUPPORT_WHEEL_NAMES, preserve_order=True
            )
        },
    )
    wheel_contact = ObsTerm(
        func=mdp.wheel_contact_flags,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True
            )
        },
    )
    pivot_command = ObsTerm(
        func=mdp.pivot_command_obs, params={"command_name": "pivot_mode"}
    )
    balance_history = ObsTerm(
        func=mdp.balance_signals,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=REAR_SUPPORT_WHEEL_NAMES,
                preserve_order=True,
            ),
        },
        history_length=5,
        flatten_history_dim=True,
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class PivotCriticObsCfg(ObsGroup):
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    leg_joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    leg_joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        scale=0.05,
    )
    wheel_joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
        scale=0.05,
    )
    previous_action = ObsTerm(func=mdp.last_action)
    balance_signals = ObsTerm(
        func=mdp.balance_signals,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=REAR_SUPPORT_WHEEL_NAMES, preserve_order=True
            )
        },
    )
    wheel_contact = ObsTerm(
        func=mdp.wheel_contact_flags,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True
            )
        },
    )
    pivot_command = ObsTerm(
        func=mdp.pivot_command_obs, params={"command_name": "pivot_mode"}
    )
    balance_history = ObsTerm(
        func=mdp.balance_signals,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=REAR_SUPPORT_WHEEL_NAMES, preserve_order=True
            )
        },
        history_length=5,
        flatten_history_dim=True,
    )
    # Privileged terms (section 7).
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    contact_force = ObsTerm(
        func=mdp.contact_forces_term,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True
            ),
            "threshold": 1.0,
        },
    )
    com_offset = ObsTerm(func=mdp.com_offset_term)
    action_delay = ObsTerm(func=mdp.action_delay_term)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PivotObservationsCfg:
    policy: PivotPolicyObsCfg = PivotPolicyObsCfg()
    critic: PivotCriticObsCfg = PivotCriticObsCfg()


##
# Rewards (section 7 table): mode-gated terms with exact weights
##


@configclass
class PivotRewardsCfg:
    # omega_z tracking: GROUND 1.0, BALANCE 1.5
    omega_z = RewTerm(
        func=mdp.omega_z_tracking,
        weight=1.0,
        params={"command_name": "pivot_mode", "std": 0.5},
    )
    # exp(-xi^2/sigma^2): REAR_UP 1.5, BALANCE 2.5
    capture_point = RewTerm(
        func=mdp.capture_point_reward,
        weight=1.0,
        params={
            "command_name": "pivot_mode",
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=REAR_SUPPORT_WHEEL_NAMES, preserve_order=True
            ),
            "std": 0.12,
        },
    )
    # anchor rear-axle midpoint: 1.0/0.5/1.0/0.5
    anchor = RewTerm(
        func=mdp.anchor_midpoint_reward,
        weight=1.0,
        params={
            "command_name": "pivot_mode",
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=REAR_SUPPORT_WHEEL_NAMES, preserve_order=True
            ),
        },
    )
    # front wheels clear: REAR_UP 1.5, BALANCE 1.0
    front_wheels_off = RewTerm(
        func=mdp.front_wheels_off_ground,
        weight=1.0,
        params={
            "command_name": "pivot_mode",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=FRONT_WHEEL_NAMES, preserve_order=True
            ),
        },
    )
    # roll -> 0: 0.5/1.0/1.0/1.0
    roll_flat = RewTerm(
        func=mdp.roll_flat_reward,
        weight=1.0,
        params={"command_name": "pivot_mode"},
    )
    # leg-prior/tuck tracking through REAR_UP and BALANCE: 0.5
    tuck_tracking = RewTerm(
        func=mdp.tuck_tracking,
        weight=1.0,
        params={
            "command_name": "pivot_mode",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
            ),
        },
    )
    # instantaneous wheel torque beyond 3 Nm, mode weighted
    thermal = RewTerm(
        func=mdp.thermal_excess_penalty,
        weight=-1.0,
        params={
            "command_name": "pivot_mode",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True
            ),
        },
    )
    # LAND peak normal force: 2.0 (budget 645 N = 2mg)
    landing_fn_peak = RewTerm(
        func=mdp.landing_peak_force,
        weight=-1.0,
        params={
            "command_name": "pivot_mode",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True
            ),
        },
    )
    # I5: pitch beyond theta* x3 in REAR_UP/BALANCE
    backflip = RewTerm(
        func=mdp.backflip_excess_penalty,
        weight=-1.0,
        params={"command_name": "pivot_mode"},
    )
    # smoothness
    action_rate = RewTerm(func=mdp.pivot_action_rate_l2, weight=-0.015)
    joint_accel = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )


##
# Terminations (section 9): fall terminates early -> grace 0.5 s later stages
##


@configclass
class PivotTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[BASE_LINK_NAME]),
            "threshold": 1.0,
        },
    )
    fallen = DoneTerm(func=mdp.pivot_fallen)
    tilt = DoneTerm(
        func=mdp.pivot_tilt_exceeded,
        params={
            "roll_limit": 0.6,
            "pitch_limit_margin": 0.35,
            "command_name": "pivot_mode",
        },
    )
    drift = DoneTerm(
        func=mdp.pivot_drift_exceeded,
        params={
            "maximum_drift": 0.20,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=REAR_SUPPORT_WHEEL_NAMES, preserve_order=True
            ),
        },
    )


##
# Events: simple standing reset; heavy DR/push deferred
##


@configclass
class PivotEventCfg:
    pivot_reset = EventTerm(
        func=mdp.pivot_reset_distribution,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ARTICULATION_JOINT_NAMES, preserve_order=True
            )
        },
    )
    # TODO: Re-enable DR/push only after the ManagerBased baseline smoke test
    # is stable and each perturbation has an isolated validation.
    randomize_friction = None
    randomize_mass = None
    randomize_com = None
    randomize_actuator_gains = None
    push_robot = None


##
# Environment assembly: 200 Hz physics, 50 Hz control (decimation 4)
##


@configclass
class PivotEnvCfg(ManagerBasedRLEnvCfg):
    scene: PivotSceneCfg = PivotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: PivotObservationsCfg = PivotObservationsCfg()
    actions: PivotActionsCfg = PivotActionsCfg()
    commands: PivotCommandsCfg = PivotCommandsCfg()
    rewards: PivotRewardsCfg = PivotRewardsCfg()
    terminations: PivotTerminationsCfg = PivotTerminationsCfg()
    events: PivotEventCfg = PivotEventCfg()
    # TODO: Redesign the old curriculum around existing ManagerTerm metrics.
    curriculum = None

    decimation = 4
    episode_length_s = 20.0
    sim = sim_utils.SimulationCfg(dt=PHYS.physics_dt, render_interval=decimation)

    def __post_init__(self):
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.physx.gpu_max_rigid_patch_count = 2**19
        # Wheel effort 20 -> 24 Nm peak (spec section 2).
        self.scene.robot.actuators["wheels"].effort_limit = PHYS.wheel_peak_torque


@configclass
class PivotEnvCfg_PLAY(PivotEnvCfg):
    """Play variant: 16 envs, no corruption, no push."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
