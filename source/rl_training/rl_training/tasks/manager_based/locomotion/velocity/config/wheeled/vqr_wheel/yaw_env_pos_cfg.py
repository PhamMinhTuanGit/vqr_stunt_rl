"""Neutral/positive command-conditioned yaw task; no FSM or scripted transition."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg

from rl_training.tasks.manager_based.locomotion.velocity.mdp import yaw_pos_commands as pos_commands
from rl_training.tasks.manager_based.locomotion.velocity.mdp import yaw_pos_curriculums as pos_curriculums
from rl_training.tasks.manager_based.locomotion.velocity.mdp import yaw_pos_rewards as pos_rewards
from rl_training.tasks.manager_based.locomotion.velocity.mdp import wheel_contact_kinematics
from rl_training.tasks.manager_based.locomotion.velocity.mdp import yaw_pos_actions as pos_actions
from rl_training.tasks.manager_based.locomotion.velocity.mdp.yaw_pos_asset import inspect_pos_asset
from rl_training.tasks.manager_based.locomotion.velocity.velocity_yaw_env_cfg import CommandsCfg

from .yaw_env_cfg import (
    LEG_JOINT_NAMES,
    LIFT_CLEARANCE_LEVELS,
    MIN_BASE_HEIGHT,
    ONLINE_DR_SCALE_LEVELS,
    SUPPORT_WHEEL_NAMES,
    LIFTED_WHEEL_NAMES,
    TARGET_BASE_HEIGHT,
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
SUPPORT_LINE_SIGMA = 0.20
SUPPORT_Y_MIN_SEPARATION = 0.20
SUPPORT_Y_COLLAPSE_SCALE = 0.05
POS_YAW_HEIGHT = 0.455
POS_YAW_RATE_LEVELS = YAW_RATE_LEVELS + (1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
POS_DR_SCALE_LEVELS = ONLINE_DR_SCALE_LEVELS + (1.0,) * 6
POS_TRACKING_RATIO_THRESHOLDS = YAW_TRACKING_RATIO_THRESHOLDS + (0.55,) * 6
POS_EDGE_TRACKING_RATIO_THRESHOLDS = YAW_EDGE_TRACKING_RATIO_THRESHOLDS + (0.45,) * 6


@configclass
class VQRWheelYawPosActionsCfg:
    joint_pos = pos_actions.LimitedJointPositionActionCfg(
        asset_name="robot", joint_names=LEG_JOINT_NAMES, scale=0.5,
        use_default_offset=True, preserve_order=True,
    )
    joint_vel = pos_actions.LimitedJointVelocityActionCfg(
        asset_name="robot", joint_names=ALL_WHEELS, scale=5.0,
        use_default_offset=True, preserve_order=True,
    )


def configure_pos_asset(env_cfg):
    """Use one USD-derived reference for reset, actions, observations and export."""
    robot = env_cfg.scene.robot.copy()
    profile = inspect_pos_asset(robot.spawn.usd_path, target_height=TARGET_BASE_HEIGHT)
    robot.init_state.joint_pos = dict(zip(profile["leg_joint_names"], profile["default_leg_positions"]))
    robot.init_state.joint_pos.update({name: 0.0 for name in profile["wheel_joint_names"]})
    robot.init_state.pos = (0.0, 0.0, TARGET_BASE_HEIGHT)
    env_cfg.scene.robot = robot
    position = env_cfg.actions.joint_pos
    position.clip = dict(zip(profile["leg_joint_names"], map(tuple, profile["leg_position_limits"])))
    position.wheel_center_offsets = profile["wheel_center_offsets"]
    position.wheel_half_widths = profile["wheel_half_widths"]
    position.wheel_motor_axis_signs = profile["wheel_motor_axis_signs"]
    env_cfg.actions.joint_vel.clip = {".*": (-58.9, 58.9)}
    for term in vars(env_cfg.rewards).values():
        if hasattr(term, "params") and "wheel_radius" in term.params:
            term.params["wheel_radius"] = profile["wheel_radius"]
    for group in (env_cfg.observations.policy, env_cfg.observations.critic):
        group.actions.func = pos_actions.applied_target_history
        group.actions.params = {}
    noise = env_cfg.observations.policy
    noise.base_ang_vel.noise = Unoise(n_min=-0.1, n_max=0.1)
    noise.projected_gravity.noise = Unoise(n_min=-0.02, n_max=0.02)
    noise.joint_vel.noise = Unoise(n_min=-0.3, n_max=0.3)
    noise.joint_pos.noise = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=Unoise(n_min=-0.01, n_max=0.01),
        bias_noise_cfg=Unoise(n_min=-0.02, n_max=0.02), sample_bias_per_component=True,
    )
    profile.update(
        task="Flat-VQR-Wheel-Yaw-POS", actor_observations=55, critic_observations=83, actions=16,
        actor_hidden_dims=[512, 256, 128], activation="elu", empirical_normalization=False,
        action_scales=[0.30, 0.60, 0.50] * 4 + [5.0] * 4,
        action_offsets=profile["default_leg_positions"] + [0.0] * 4,
        history="normalized_applied_targets", history_slice=[39, 55],
        physics_dt=env_cfg.sim.dt, decimation=env_cfg.decimation,
        leg_target_velocity=position.max_target_velocity, leg_target_acceleration=position.max_target_acceleration,
        wheel_target_velocity=env_cfg.actions.joint_vel.max_target_velocity,
        wheel_target_acceleration=env_cfg.actions.joint_vel.max_target_acceleration,
        neutral_height=TARGET_BASE_HEIGHT, yaw_height=POS_YAW_HEIGHT,
        minimum_support_separation=SUPPORT_Y_MIN_SEPARATION,
        yaw_rate_levels=list(POS_YAW_RATE_LEVELS), command_deadband=YAW_DEADBAND,
        observation_layout=[
            {"name": "base_ang_vel_b", "slice": [0, 3], "scale": 0.25},
            {"name": "projected_gravity_b", "slice": [3, 6], "scale": 1.0},
            {"name": "yaw_command", "slice": [6, 7], "scale": 1.0},
            {"name": "leg_position_minus_reference", "slice": [7, 19], "scale": 1.0},
            {"name": "wheel_position_zero", "slice": [19, 23], "scale": 1.0},
            {"name": "joint_velocity", "slice": [23, 39], "scale": 0.05},
            {"name": "normalized_applied_targets", "slice": [39, 55], "scale": 1.0},
        ],
    )
    env_cfg.yaw_pos_contract = profile


@configclass
class VQRWheelYawPosCommandsCfg(CommandsCfg):
    yaw_rate_cmd = pos_commands.YawPosCommandCfg(
        asset_name="robot", resampling_time_range=(4.0, 6.0),
        yaw_rate_range=(0.0, YAW_RATE_LEVELS[0]), deadband=YAW_DEADBAND,
        neutral_probability=0.30, debug_vis=False,
        use_ground_heading_rate=True,
    )


@configclass
class VQRWheelYawPosRewardsCfg(VQRWheelRewardsCfg):
    """POS differential rolling and neutral holding with baseline safety terms."""

    base_height_deficit = RewTerm(
        func=pos_rewards.yaw_pos_base_height_deficit_l2, weight=-2.0,
        params={"command_name": "yaw_rate_cmd", "minimum_height": TARGET_BASE_HEIGHT - 0.03,
                "active_minimum_height": POS_YAW_HEIGHT - 0.015, "deadband": YAW_DEADBAND,
                "error_scale": 0.05, "low_speed_yaw": 0.5, "low_speed_multiplier": 1.0},
    )
    hipx_deviation = RewTerm(
        func=pos_rewards.yaw_pos_hipx_deviation_l2, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HipX_joint"], preserve_order=True),
                "std": 0.15, "free_angle": 0.50},
    )
    com_support = RewTerm(
        func=pos_rewards.yaw_pos_com_support, weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=POS_SUPPORT_WHEELS, preserve_order=True),
                "std": 0.08, "command_name": "yaw_rate_cmd", "deadband": YAW_DEADBAND},
    )
    # Disable the inherited Euclidean span band; align each support about CoM.
    support_span_band = None
    support_line = RewTerm(
        func=pos_rewards.yaw_pos_support_line, weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=POS_SUPPORT_WHEELS, preserve_order=True),
                "sigma": SUPPORT_LINE_SIGMA, "command_name": "yaw_rate_cmd", "deadband": YAW_DEADBAND},
    )
    support_y_collapse = RewTerm(
        func=pos_rewards.yaw_pos_support_y_collapse_l2, weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=POS_SUPPORT_WHEELS, preserve_order=True),
                "minimum_separation": SUPPORT_Y_MIN_SEPARATION,
                "separation_scale": SUPPORT_Y_COLLAPSE_SCALE,
                "command_name": "yaw_rate_cmd", "deadband": YAW_DEADBAND},
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
    heading_support = RewTerm(
        func=pos_rewards.yaw_pos_heading_support, weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=POS_SUPPORT_WHEELS, preserve_order=True),
                "scale": 0.27, "command_name": "yaw_rate_cmd", "deadband": YAW_DEADBAND},
    )
    gated_yaw_tracking = RewTerm(
        func=pos_rewards.yaw_pos_gated_tracking, weight=6.0,
        params={"command_name": "yaw_rate_cmd",
                "support_sensor_cfg": SceneEntityCfg("contact_forces", body_names=POS_SUPPORT_WHEELS, preserve_order=True),
                "lifted_asset_cfg": SceneEntityCfg("robot", body_names=POS_LIFTED_WHEELS, preserve_order=True),
                "wheel_radius": WHEEL_RADIUS, "target_clearance": LIFT_CLEARANCE_LEVELS[0],
                "std": 0.20, "neutral_std": 0.30, "deadband": YAW_DEADBAND, "contact_threshold": 1.0,
                "support_asset_cfg": SceneEntityCfg("robot", body_names=POS_SUPPORT_WHEELS, preserve_order=True),
                "support_joint_cfg": SceneEntityCfg("robot", joint_names=POS_SUPPORT_WHEELS, preserve_order=True),
                "clearance_gate_floor": 0.25, "edge_command_fraction": 0.80,
                "tracking_relative_std": 0.20, "tracking_min_std": 0.04},
    )
    body_angular_xy = RewTerm(
        func=pos_rewards.yaw_pos_body_angular_xy_l2, weight=-0.1,
        params={"command_name": "yaw_rate_cmd", "deadband": YAW_DEADBAND},
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
    differential_rolling = RewTerm(
        func=pos_rewards.yaw_pos_differential_rolling, weight=4.0,
        params={"support_asset_cfg": SceneEntityCfg("robot", body_names=POS_SUPPORT_WHEELS, preserve_order=True),
                "support_joint_cfg": SceneEntityCfg("robot", joint_names=POS_SUPPORT_WHEELS, preserve_order=True),
                "support_sensor_cfg": SceneEntityCfg("contact_forces", body_names=POS_SUPPORT_WHEELS, preserve_order=True),
                "command_name": "yaw_rate_cmd", "deadband": YAW_DEADBAND, "wheel_radius": WHEEL_RADIUS,
                "support_parent_cfg": SceneEntityCfg("robot", body_names=["FL_SHANK", "HR_SHANK"], preserve_order=True),
                "relative_std": 0.25, "speed_std": 0.015, "certification_speed_std": 0.05,
                "motor_tracking_weight": 1.0, "minimum_speed_fraction": 0.5, "settle_time": 0.5},
    )
    planar_velocity = RewTerm(
        func=pos_rewards.yaw_pos_active_translation, weight=-2.0,
        params={"command_name": "yaw_rate_cmd", "deadband": YAW_DEADBAND,
                "speed_scale": 0.10, "drift_scale": 0.05, "settle_time": 0.5},
    )
    neutral_velocity = RewTerm(
        func=pos_rewards.yaw_pos_neutral_velocity, weight=3.0,
        params={"command_name": "yaw_rate_cmd", "deadband": YAW_DEADBAND, "std": 0.05},
    )
    neutral_position = RewTerm(
        func=pos_rewards.yaw_pos_neutral_position, weight=2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=ALL_WHEELS, preserve_order=True),
                "command_name": "yaw_rate_cmd", "deadband": YAW_DEADBAND, "std": 0.05,
                "contact_dwell": 0.2, "speed_threshold": 0.03, "drift_threshold": 0.05},
    )

    def __post_init__(self):
        self.base_height.func = pos_rewards.yaw_pos_base_height_tracking
        self.base_height.weight = 6.0
        self.base_height.params.update(command_name="yaw_rate_cmd", deadband=YAW_DEADBAND,
                                       active_target_height=POS_YAW_HEIGHT)
        self.action_rate.func = pos_actions.applied_target_rate_l2
        self.lateral_slip.weight = -4.0
        self.rolling_slip.weight = -2.0
        self.rolling_slip.func = pos_rewards.yaw_pos_rolling_wheel_slip
        self.rolling_slip.params["deadband"] = YAW_DEADBAND
        self.rolling_slip.params["minimum_scale"] = 0.25


@configclass
class VQRWheelYawPosCurriculumCfg:
    task_levels = CurrTerm(
        func=pos_curriculums.yaw_pos_task_levels,
        params={"command_name": "yaw_rate_cmd", "clearance_levels": LIFT_CLEARANCE_LEVELS,
                "yaw_rate_levels": POS_YAW_RATE_LEVELS, "dr_scale_levels": POS_DR_SCALE_LEVELS,
                "tracking_ratio_thresholds": POS_TRACKING_RATIO_THRESHOLDS,
                "edge_tracking_ratio_thresholds": POS_EDGE_TRACKING_RATIO_THRESHOLDS,
                "lift_reward_name": "lift_clearance", "balance_reward_name": "balance",
                "yaw_reward_name": "gated_yaw_tracking", "transition_reward_name": None,
                "torso_contact_termination_name": "torso_contact", "minimum_base_height": MIN_BASE_HEIGHT,
                "support_threshold": 0.85, "lift_progress_threshold": 0.80,
                "balance_threshold": 0.75, "yaw_threshold": 0.65,
                "min_evaluated_episodes": 2048, "required_success_rate": 0.85,
                "required_consecutive_windows": 3, "min_clearance_stage_steps": 1000,
                "min_yaw_stage_steps": 6000, "certify_behavior": True,
                "behavior_pass_threshold": 0.90},
    )


@configclass
class VQRWheelFlatEnvPOSCfg(VQRWheelFlatEnvCfg):
    actions: VQRWheelYawPosActionsCfg = VQRWheelYawPosActionsCfg()
    commands: VQRWheelYawPosCommandsCfg = VQRWheelYawPosCommandsCfg()
    rewards: VQRWheelYawPosRewardsCfg = VQRWheelYawPosRewardsCfg()
    curriculum: VQRWheelYawPosCurriculumCfg = VQRWheelYawPosCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.critic.rolling_lateral_contact_velocity.func = (
            wheel_contact_kinematics.rolling_lateral_contact_velocity
        )
        # The baseline post-init initializes a symmetric range; this task is positive-only.
        self.commands.yaw_rate_cmd.yaw_rate_range = (0.0, YAW_RATE_LEVELS[0])
        configure_pos_asset(self)
