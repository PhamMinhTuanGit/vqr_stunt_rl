"""Transfer the base two-wheel POS skill; closed deadband requests FOUR_STAND."""

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.mdp import yaw_pos_rewards as neutral
from rl_training.tasks.manager_based.locomotion.velocity.mdp import yaw_pos_transfer_rewards as transfer
from rl_training.tasks.manager_based.locomotion.velocity.mdp.yaw_pos_transfer_commands import YawPosTransferCommandCfg
from rl_training.tasks.manager_based.locomotion.velocity.mdp.yaw_pos_transfer_curriculums import yaw_pos_transfer_task_levels
from rl_training.tasks.manager_based.locomotion.velocity.velocity_yaw_env_cfg import CommandsCfg

from .yaw_env_cfg import (
    LEG_JOINT_NAMES, WHEEL_NAMES, LIFTED_WHEEL_NAMES, WHEEL_RADIUS,
    YAW_RATE_LEVELS, LIFT_CLEARANCE_LEVELS, VQRWheelFlatEnvCfg,
    VQRWheelRewardsCfg, VQRWheelYawCurriculumCfg,
)

POSITIVE_SPECIFIC = (
    "com_support", "base_height", "support_span_band", "lift_clearance",
    "com_inside_segment", "balance", "gated_yaw_tracking", "lifted_wheel_spin",
)
NEUTRAL_SPECIFIC = (
    "neutral_landing_progress", "four_stand_pose", "four_wheel_contact", "neutral_yaw_tracking",
)
GLOBAL = (
    "low_base_height", "downward_low_base_velocity", "lateral_slip", "rolling_slip",
    "action_rate", "undesired_contact", "joint_limits", "joint_velocity", "torque", "planar_velocity",
)
REWARD_CLASSIFICATION = {
    **dict.fromkeys(POSITIVE_SPECIFIC, "POSITIVE_SPECIFIC"),
    **dict.fromkeys(NEUTRAL_SPECIFIC, "NEUTRAL_SPECIFIC"),
    **dict.fromkeys(GLOBAL, "GLOBAL"),
}


@configclass
class VQRWheelYawPosTransferCommandsCfg(CommandsCfg):
    # 15% exact zero, 15% uniform deadband, 70% positive; preserve total neutral mass.
    yaw_rate_cmd = YawPosTransferCommandCfg(
        asset_name="robot", resampling_time_range=(4.0, 6.0),
        yaw_rate_range=(0.0, YAW_RATE_LEVELS[0]), deadband=transfer.DEADBAND,
        neutral_probability=0.30, debug_vis=False,
    )


@configclass
class VQRWheelYawPosTransferRewardsCfg(VQRWheelRewardsCfg):
    # Positive terms inherit every base parameter and weight. __post_init__
    # only replaces their callable with a signature-preserving mode wrapper.
    neutral_landing_progress = RewTerm(
        func=neutral.yaw_pos_neutral_landing_progress, weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True),
                "wheel_radius": WHEEL_RADIUS, "target_clearance": LIFT_CLEARANCE_LEVELS[0],
                "command_name": transfer.COMMAND_NAME, "deadband": transfer.DEADBAND},
    )
    four_stand_pose = RewTerm(
        func=neutral.yaw_pos_four_stand_pose, weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
                "command_name": transfer.COMMAND_NAME, "deadband": transfer.DEADBAND, "std": 0.25},
    )
    four_wheel_contact = RewTerm(
        func=neutral.yaw_pos_four_wheel_contact, weight=3.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_NAMES, preserve_order=True),
                "command_name": transfer.COMMAND_NAME, "deadband": transfer.DEADBAND, "threshold": 1.0},
    )
    neutral_yaw_tracking = RewTerm(
        func=transfer.neutral_yaw_tracking, weight=8.0,
        params={"std": 0.30, "command_name": transfer.COMMAND_NAME, "deadband": transfer.DEADBAND},
    )

    def __post_init__(self):
        for name in POSITIVE_SPECIFIC:
            term = getattr(self, name)
            term.func = getattr(transfer, f"positive_{term.func.__name__}")


@configclass
class VQRWheelYawPosTransferCurriculumCfg(VQRWheelYawCurriculumCfg):
    def __post_init__(self):
        self.task_levels.func = yaw_pos_transfer_task_levels
        self.task_levels.params["active_only"] = True
        # New neutral-only gate; positive thresholds/stage tables are inherited.
        self.task_levels.params["neutral_planar_speed_threshold"] = 0.10


@configclass
class VQRWheelYawPosTransferEnvCfg(VQRWheelFlatEnvCfg):
    commands: VQRWheelYawPosTransferCommandsCfg = VQRWheelYawPosTransferCommandsCfg()
    rewards: VQRWheelYawPosTransferRewardsCfg = VQRWheelYawPosTransferRewardsCfg()
    curriculum: VQRWheelYawPosTransferCurriculumCfg = VQRWheelYawPosTransferCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.yaw_rate_cmd.yaw_rate_range = (0.0, YAW_RATE_LEVELS[0])
