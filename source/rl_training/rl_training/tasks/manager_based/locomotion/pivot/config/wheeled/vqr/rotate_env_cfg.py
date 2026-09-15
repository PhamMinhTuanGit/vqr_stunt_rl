# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""M2 yaw-rotation task built directly on the M1 two-wheel balance task."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.pivot.mdp as mdp

from .balance_env_cfg import ObservationsCfg, RewardsCfg, VQRTwoWheelBalanceEnvCfg


YAW_COMMAND_NAME = "yaw_rate_cmd"
YAW_RATE_LEVELS = (0.3, 0.6, 1.0, 1.5)


@configclass
class YawRateCommandsCfg:
    """The sole M2 command: body-frame yaw rate."""

    yaw_rate_cmd = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        rel_standing_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-YAW_RATE_LEVELS[0], YAW_RATE_LEVELS[0]),
            heading=None,
        ),
    )


@configclass
class RotateObservationsCfg(ObservationsCfg):
    """M1 observations plus the scalar yaw-rate command."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        yaw_rate_command = ObsTerm(
            func=mdp.yaw_rate_command,
            params={"command_name": YAW_COMMAND_NAME},
        )

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        yaw_rate_command = ObsTerm(
            func=mdp.yaw_rate_command,
            params={"command_name": YAW_COMMAND_NAME},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RotateRewardsCfg(RewardsCfg):
    """The complete M1 reward set plus yaw-rate tracking."""

    yaw_rate_tracking = RewTerm(
        func=mdp.pivot_track_yaw_rate,
        weight=1.5,
        params={"command_name": YAW_COMMAND_NAME, "std": 0.30},
    )


@configclass
class YawRateCurriculumCfg:
    """Performance-gated progression through four symmetric yaw ranges."""

    yaw_rate_levels = CurrTerm(
        func=mdp.pivot_yaw_rate_levels,
        params={
            "command_name": YAW_COMMAND_NAME,
            "levels": YAW_RATE_LEVELS,
            "support_reward_name": "support_contact",
            "lifted_reward_name": "lifted_diagonal",
            "balance_reward_name": "balance",
            "yaw_reward_name": "yaw_rate_tracking",
            "support_threshold": 0.85,
            "lifted_threshold": 0.85,
            "balance_threshold": 0.75,
            "yaw_threshold": 0.70,
            "min_evaluated_episodes": 256,
            "required_success_rate": 0.80,
        },
    )


@configclass
class VQRTwoWheelRotateEnvCfg(VQRTwoWheelBalanceEnvCfg):
    """M2: retain the M1 diagonal balance task and add controlled yaw rotation."""

    observations: RotateObservationsCfg = RotateObservationsCfg()
    commands: YawRateCommandsCfg = YawRateCommandsCfg()
    rewards: RotateRewardsCfg = RotateRewardsCfg()
    curriculum: YawRateCurriculumCfg = YawRateCurriculumCfg()
