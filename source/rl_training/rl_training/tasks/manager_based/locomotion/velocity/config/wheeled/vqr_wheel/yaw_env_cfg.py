# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.velocity_yaw_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
)

##
# Pre-defined configs
##
from rl_training.assets.deeprobotics import VQRWHEEL_CFG  # isort: skip


SUPPORT_WHEEL_NAMES = ["FL_WHEEL", "HR_WHEEL"]
LIFTED_WHEEL_NAMES = ["FR_WHEEL", "HL_WHEEL"]
WHEEL_NAMES = ["FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL"]
LEG_JOINT_NAMES = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]
WHEEL_RADIUS = 0.091
TARGET_LIFT_CLEARANCE = 0.05


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
    """The explicit fourteen-term reward set for diagonal-support yaw training."""

    com_support = RewTerm(
        func=mdp.yaw_com_support,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
            ),
            "std": 0.08,
        },
    )
    support_contact = RewTerm(
        func=mdp.yaw_support_contact,
        weight=4.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
            ),
            "threshold": 1.0,
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
            "target_clearance": TARGET_LIFT_CLEARANCE,
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
        weight=3.0,
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
            "target_clearance": TARGET_LIFT_CLEARANCE,
            "std": 0.30,
            "contact_threshold": 1.0,

            # NEW
            "clearance_gate_floor": 0.25,
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
            "threshold": 1.0,
        },
    )
    undesired_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["^(?!.*_WHEEL).*"], preserve_order=True
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
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINT_NAMES + WHEEL_NAMES, preserve_order=True
            )
        },
    )
    lifted_wheel_spin = RewTerm(
        func=mdp.yaw_lifted_wheel_spin_l2,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LIFTED_WHEEL_NAMES, preserve_order=True
            )
        },
    )


@configclass
class VQRWheelRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: VQRWheelActionsCfg = VQRWheelActionsCfg()
    rewards: VQRWheelRewardsCfg = VQRWheelRewardsCfg()

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
        # reduce action scale
        self.actions.joint_pos.scale = {".*_HipX_joint": 0.125, "^(?!.*_HipX_joint).*": 0.25}
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
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.0, 0.0),
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

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.illegal_contact = None
        self.terminations.bad_orientation_2 = None

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels = None

        # ------------------------------Commands------------------------------
        self.commands.yaw_rate_cmd.yaw_rate_range = (-1.0, 1.0)


@configclass
class VQRWheelFlatEnvCfg(VQRWheelRoughEnvCfg):
    """Flat-ground VQR wheel task driven by a scalar yaw-rate command."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None
