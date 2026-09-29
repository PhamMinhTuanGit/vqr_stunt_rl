"""Flat VQR yaw task with one certified maneuver per episode."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .yaw_env_fsm_cfg import (
    VQRWheelFlatEnvFSMCfg,
    VQRWheelFSMCommandsCfg,
    VQRWheelFSMRewardsCfg,
    VQRWheelFSMTerminationsCfg,
)


CLEARANCE_LEVELS = (0.02, 0.03, 0.05)
YAW_RATE_LEVELS = (0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00)
ONLINE_DR_SCALE_LEVELS = (0.30, 0.30, 0.40, 0.55, 0.70, 0.85, 1.00)
TRACKING_RATIO_THRESHOLDS = (0.30, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)


@configclass
class VQRWheelStagedCommandsCfg(VQRWheelFSMCommandsCfg):
    yaw_rate_cmd: mdp.StagedYawCommandCfg = mdp.StagedYawCommandCfg(
        asset_name="robot",
        resampling_time_range=(100.0, 100.0),
        yaw_rate_range=(-0.15, 0.15),
        yaw_rate_limit=0.15,
        target_clearance=CLEARANCE_LEVELS[0],
        initial_hold_s=0.5,
        yaw_duration_s=4.0,
        yaw_enter=0.10,
        yaw_exit=0.05,
        yaw_min_dwell=0.20,
        four_stand_ready_dwell=0.20,
        phase0_hold_s=1.0,
        recovery_dwell=0.50,
    )

    def __post_init__(self):
        # Preserve the six ordered sensor/body selections from the FSM task.
        baseline = VQRWheelFSMCommandsCfg().yaw_rate_cmd
        for field in (
            "support_sensor_cfg", "support_sensor_cfg_mirror", "lifted_asset_cfg",
            "lifted_asset_cfg_mirror", "all_wheel_sensor_cfg", "torso_sensor_cfg",
        ):
            setattr(self.yaw_rate_cmd, field, getattr(baseline, field))
        self.yaw_rate_cmd.wheel_radius = baseline.wheel_radius
        self.yaw_rate_cmd.contact_threshold = baseline.contact_threshold
        self.yaw_rate_cmd.clearance_fraction = baseline.clearance_fraction
        self.yaw_rate_cmd.pose_angle_limit = baseline.pose_angle_limit
        self.yaw_rate_cmd.unsafe_angle_limit = baseline.unsafe_angle_limit
        self.yaw_rate_cmd.minimum_base_height = baseline.minimum_base_height


@configclass
class VQRWheelStagedRewardsCfg(VQRWheelFSMRewardsCfg):
    staged_four_contact = RewTerm(
        func=mdp.staged_four_contact, weight=2.0,
        params={"command_name": "yaw_rate_cmd"},
    )
    staged_four_dwell_bonus = RewTerm(
        func=mdp.staged_four_dwell_bonus, weight=5.0,
        params={"command_name": "yaw_rate_cmd"},
    )
    staged_milestone_bonus = RewTerm(
        func=mdp.staged_milestone_bonus, weight=0.0,
        params={"command_name": "yaw_rate_cmd"},
    )
    fsm_failure = RewTerm(func=mdp.staged_failure_cost, weight=-60.0)


@configclass
class VQRWheelStagedTerminationsCfg(VQRWheelFSMTerminationsCfg):
    staged_complete = DoneTerm(
        func=mdp.staged_complete, params={"command_name": "yaw_rate_cmd"},
    )
    staged_promotion_reset = DoneTerm(func=mdp.staged_promotion_reset, time_out=True)


@configclass
class VQRWheelStagedCurriculumCfg:
    task_levels = CurrTerm(
        func=mdp.staged_yaw_task_levels,
        params={
            "command_name": "yaw_rate_cmd",
            "clearance_levels": CLEARANCE_LEVELS,
            "yaw_rate_levels": YAW_RATE_LEVELS,
            "dr_scale_levels": ONLINE_DR_SCALE_LEVELS,
            "tracking_ratio_thresholds": TRACKING_RATIO_THRESHOLDS,
            "hold_window": 2048,
            "directional_window": 1024,
            "required_windows": 3,
        },
    )


@configclass
class VQRWheelFlatEnvFSMStagedCfg(VQRWheelFlatEnvFSMCfg):
    commands: VQRWheelStagedCommandsCfg = VQRWheelStagedCommandsCfg()
    rewards: VQRWheelStagedRewardsCfg = VQRWheelStagedRewardsCfg()
    terminations: VQRWheelStagedTerminationsCfg = VQRWheelStagedTerminationsCfg()
    curriculum: VQRWheelStagedCurriculumCfg = VQRWheelStagedCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.yaw_rate_cmd.yaw_rate_range = (-0.15, 0.15)
        # The Rough VQR asset supplies z=0.45, HipX=0, HipY=-0.65,
        # Knee=1.3, and zero default joint velocities. Reset it exactly.
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
        self.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-3.14, 3.14),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.randomize_apply_external_force_torque.params["force_range"] = (0.0, 0.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (0.0, 0.0)
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (1.0, 1.0)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (1.0, 1.0)
        self.events.randomize_push_robot.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0),
        }
        self.rewards.fsm_gated_tracking.weight = 0.0
        self.rewards.transition_progress.weight = 0.0
        self.rewards.return_to_four_landing.weight = 0.0
        self.rewards.four_stand_ready_bonus.weight = 0.0
        self.rewards.spin_center_drift.weight = 0.0
        self.rewards.safe_recovery_entry.weight = 0.0
