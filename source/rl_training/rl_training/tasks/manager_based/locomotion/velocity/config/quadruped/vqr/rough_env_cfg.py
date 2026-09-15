# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
# from isaaclab.sensors.ray_caster import GridPatternCfg
##
# Pre-defined configs
##
from rl_training.assets.deeprobotics import VQR_CFG  # isort: skip


@configclass
class VQRRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "TORSO"
    foot_link_name = ".*_FOOT"
    # fmt: off
    joint_names = [
        "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
        "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
        "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
        "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
    ]

    link_names = [
       'TORSO', 
       'FL_HIP', 'FR_HIP', 'HL_HIP', 'HR_HIP', 
       'FL_THIGH', 'FR_THIGH', 'HL_THIGH', 'HR_THIGH', 
       'FL_SHANK', 'FR_SHANK', 'HL_SHANK', 'HR_SHANK', 
       'FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT',
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
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = VQR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner.pattern_cfg.resolution = 0.07 #  = GridPatternCfg(resolution=0.07, size=[1.6, 1.0]),

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel = None # type: ignore
        self.observations.policy.height_scan = None # type: ignore
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_HipX_joint": 0.125, "^(?!.*_HipX_joint).*": 0.25}
        # The VQR HipX range is +-45 deg vs +-30 deg on Lite3, so with an unbounded clip the policy
        # can abduct 0.79 rad before joint_pos_limits fires. Cap the HipX *action* at +-3.2 * 0.125
        # = +-0.4 rad so splaying stops being free, while leaving the pitch joints unclipped.
        self.actions.joint_pos.clip = {".*_HipX_joint": (-3.2, 3.2), "^(?!.*_HipX_joint).*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

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


        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = self.link_names # [self.base_link_name]
        # self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_com_positions.params["asset_cfg"].body_names = self.base_link_name # [self.base_link_name]
        # self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = self.joint_names

        # set terrain generation probability to 0 for boxes and stairs
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 1.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
        # scale down the terrains because the robot is small
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_width = 0.8
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.0, 0.01)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------Rewards------------------------------
        self.rewards.action_rate_l2.weight = -0.1 #-0.02
        # self.rewards.smoothness_2.weight = -0.0075

        self.rewards.base_height_l2.weight = -50.0
        # Nominal root height at the default pose is 0.3516 m. Keep the target just below it: this
        # term's job is to hold the trunk at the design height, NOT to demand a squat.
        # (0.27 was tried and produced knee-walking: reaching it needs knee ~2.09 in a proper
        # stance, but the policy could get there for free by pitching the whole leg 45 deg through
        # HipY -- which carried no penalty -- dropping the knee joint to z = 0.015 m, below the
        # ankle. Measured base_z settled at 0.252 with 1.5 non-foot links on the ground.)
        self.rewards.base_height_l2.params["target_height"] = 0.34
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.feet_air_time_lin_xy.weight = 5.0 # 5.0
        self.rewards.feet_air_time_lin_xy.params["threshold"] = 0.5
        self.rewards.feet_air_time_lin_xy.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_x_neg.weight = 0.0 # 5.0
        self.rewards.feet_air_time_x_neg.params["threshold"] = 0.5
        self.rewards.feet_air_time_x_neg.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_ang_z.weight = 5.0 # 5.0
        self.rewards.feet_air_time_ang_z.params["threshold"] = 0.5
        self.rewards.feet_air_time_ang_z.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -0.0 # -8.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        # Back to the Lite3 value: at -0.05 a wide, sliding, shuffling stance was almost free.
        # -0.2 then produced a near-static policy at 3000 iter (error_vel_xy flat ~1.1 m/s from
        # iter 800 onward, time_out~100%): combined with the other VQR-only contact penalties
        # below, exploring an actual gait got punished harder than on Lite3, so the policy
        # settled for standing still. Split the difference while retuning.
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.foot_impact_velocity.weight = -2.0 # -10.0
        self.rewards.foot_impact_velocity.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.foot_impact_velocity.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.stand_still.weight = -0.5 # -1.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.stand_still.params["command_threshold"] = 0.1
        self.rewards.feet_height_body.weight = -0.0 # -2.5
        self.rewards.feet_height_body.params["target_height"] = -0.35
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = -0.0 # -0.2
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.contact_forces.weight = -1e-1 # -2e-2
        # The 100 N default is sized for Lite3 (117 N total weight): even a two-foot trot stance
        # only loads each foot to 59 N, so the term never fires. VQR weighs 276 N, so a two-foot
        # stance is 138 N/foot -> 38 N over the threshold on EACH support foot, while a four-foot
        # stance at 69 N/foot stays free. That is a direct, permanent bribe to keep all four feet
        # down and shuffle instead of trotting. Rescaled by weight: 100 * 276/117 = 235 N.
        self.rewards.contact_forces.params["threshold"] = 235.0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Both were doubled vs Lite3. On a robot whose roll gyration radius is 0.68x its stance
        # half-width (Lite3: 0.42x), the cheapest way to satisfy a heavy flatness/vertical penalty
        # is to splay the hips and stop moving. Back to the Lite3 weights.
        self.rewards.lin_vel_z_l2.weight = -10.0 #-2.0
        self.rewards.ang_vel_xy_l2.weight = -0.25 # -0.05

        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # At -0.5 the measured knee-walking gait paid only 0.76/step for having 1.5 links on the
        # ground, which was cheaper than what it saved in base_height + torque + power + action
        # rate. -2.0 makes that trade clearly negative (3.0/step at the same contact count), but
        # at 3000 iter it (with feet_slide and hipx_joint_pos_penalty also over Lite3's ratio)
        # correlated with a near-static policy instead. Ease back toward Lite3's -0.5 while
        # retuning; revisit knee-walking if it reappears.
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]

        self.rewards.joint_torques_l2.weight = -2.5e-4
        self.rewards.joint_acc_l2.weight = -1e-8
        # Second, name-based handle on abduction (L1 sum over the 4 HipX joints, so the marginal
        # cost is 4x the weight per rad). Together with hipx_joint_pos_penalty this is ~6/rad.
        self.rewards.joint_deviation_l1.weight = -0.5
        self.rewards.joint_deviation_l1.params["asset_cfg"].joint_names = [".*HipX.*"]
        self.rewards.joint_power.weight = -8e-4
        self.rewards.flat_orientation_l2.weight = -10.0

        # add the following rewards to improve the gait
        self.rewards.feet_gait.weight = 0.5
        self.rewards.feet_gait.params["synced_feet_pair_names"] = [
            ["FL_FOOT", "HR_FOOT"],
            ["FR_FOOT", "HL_FOOT"]
        ]

        # Turned down 10x. With the shipped params (gait_span = 0.0, stance_span = 0.0) the
        # reference foot NEVER moves fore/aft in the body frame - it is a march-in-place target -
        # so any real stride destroys it. In the validated Lite3 run this term only earns 0.037/2.0
        # (inert, because Lite3's posture never matches the reference either), and in the VQR
        # knee-walking run it earned 0.005. Fixing the posture woke it up to 0.857, i.e. the policy
        # started paying ~0.83/step for walking, and it stopped walking.
        # To use it properly instead of muting it: stance_span = 1.0 (50% duty) and
        # gait_span = +stride/2 (~0.106 m for 1 m/s at cycle_time 0.425 s), then raise the weight.
        self.rewards.phase_foot_trajectory_exp.weight = 0.2
        self.rewards.phase_foot_trajectory_exp.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FL_(HipX|HipY|Knee).*", "HR_(HipX|HipY|Knee).*"],
            ["FR_(HipX|HipY|Knee).*", "HL_(HipX|HipY|Knee).*"],
        ]

        self.rewards.joint_pos_limits.weight = -5.0
        # self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]

        # added rewards
        # joint_pos_penalty returns linalg.norm(deviation), i.e. it is ~linear: -0.4 buys only
        # 0.4 reward per rad of collective abduction while track_lin_vel_xy_exp is worth 4.0.
        # Lite3 got away with it because its hard limit is +-0.523; VQR's is +-0.785.
        # -2.0 (5x Lite3) tracked with turning/side-stepping getting worse over training
        # (error_vel_yaw rose from 0.46 to 0.88 over 3000 iter) -- both need hip abduction, which
        # this increasingly suppressed. -1.0 keeps it stricter than Lite3 for the wider range
        # while giving turning/strafing room to be learned.
        self.rewards.hipx_joint_pos_penalty.weight = -1.0
        self.rewards.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipx_joint_names
        # HipY was the *only* completely free DOF, and that is exactly how the policy reached a low
        # trunk without bending the knee: pitch the whole leg instead of folding it. Small weight
        # (marginal cost ~1.0 per rad of collective deviation) - enough to break the tie towards
        # the nominal leg pitch without shortening the swing. Main knob if the stride gets short.
        # Dialled back from -0.5: HipY is the main stride joint and the log showed it deviating
        # only 0.09 rad/joint (a 1 m/s trot needs ~0.3-0.4). Anti-kneeling is already covered by
        # base_height 0.34 + undesired_contacts -2.0 + the illegal_contact termination, so this
        # only needs to be a tie-breaker. Lite3 runs it at 0.
        self.rewards.hipy_joint_pos_penalty.weight = -0.1
        self.rewards.hipy_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipy_joint_names
        self.rewards.knee_joint_pos_penalty.weight = -2
        self.rewards.knee_joint_pos_penalty.params["asset_cfg"].joint_names = self.knee_joint_names


        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "VQRRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # Hard stop on kneeling. Restricted to TORSO + THIGH: in the default pose the thigh
        # collider bottom sits 0.207 m above ground, so this can only fire when the robot has
        # actually collapsed onto its legs. The SHANK is deliberately left out - its collider is
        # only ~0.05 m clear during a normal crouched trot - it is handled by undesired_contacts.
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = ["TORSO", ".*_THIGH"]
        # self.terminations.bad_orientation_2 = None

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels = None

        # ------------------------------Commands------------------------------
        # 2.0 m/s was geometrically unreachable and poisoned the whole optimisation: with
        # phase_foot_trajectory_exp's cycle_time = 0.425 s it asks for a 0.85 m stride, while the
        # 0.425 m leg at a symmetric stance can produce at most ~0.35 m. The tracking term then
        # saturates near zero everywhere and the policy falls back on the cheap terms - stand wide,
        # keep the trunk flat, do not lift the feet. Raise this again only after the trot is clean
        # (and shorten cycle_time if you want >1.2 m/s).
        self.commands.base_velocity.ranges.lin_vel_x = (-1.2, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)