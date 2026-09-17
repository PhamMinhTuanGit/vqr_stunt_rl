"""Wheeled VQR model and environment configurations for pivot tasks."""

import gymnasium as gym

from . import agents

from .robot_cfg import (
    ARTICULATION_BODY_NAMES,
    ARTICULATION_JOINT_NAMES,
    BASE_LINK_NAME,
    CONTROLLED_JOINT_NAMES,
    LEG_JOINT_NAMES,
    LIFTED_WHEEL_NAMES,
    SUPPORT_WHEEL_NAMES,
    VQR_CFG,
    WHEEL_BODY_NAMES,
    WHEEL_JOINT_NAMES,
)

__all__ = [
    "ARTICULATION_BODY_NAMES",
    "ARTICULATION_JOINT_NAMES",
    "BASE_LINK_NAME",
    "CONTROLLED_JOINT_NAMES",
    "LEG_JOINT_NAMES",
    "LIFTED_WHEEL_NAMES",
    "SUPPORT_WHEEL_NAMES",
    "VQR_CFG",
    "WHEEL_BODY_NAMES",
    "WHEEL_JOINT_NAMES",
]


gym.register(
    id="Pivot-TwoWheelBalance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg:VQRTwoWheelBalanceEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:VQRTwoWheelBalancePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Pivot-TwoWheelRotate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rotate_env_cfg:VQRTwoWheelRotateEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:VQRTwoWheelRotatePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Pivot-VQR-FourWheel-Rotate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.four_wheel_rotate_env_cfg:VQRFourWheelRotateEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:VQRTwoWheelRotatePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Pivot-FourToTwoWheelRotate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.four_to_two_rotate_env_cfg:VQRFourToTwoWheelRotateEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:VQRFourToTwoWheelRotatePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Pivot-FourWheelRotate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.four_wheel_rotate_env_cfg:VQRFourWheelRotateEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:VQRFourWheelRotatePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Pivot-VQR-Train-v0",
    entry_point="rl_training.tasks.manager_based.locomotion.pivot.pivot_env:PivotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pivot_env_cfg:PivotEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PivotVQRPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Pivot-VQR-Play-v0",
    entry_point="rl_training.tasks.manager_based.locomotion.pivot.pivot_env:PivotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pivot_env_cfg:PivotEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PivotVQRPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Recover-VQR-v0",
    entry_point="rl_training.tasks.manager_based.locomotion.pivot.pivot_env:PivotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recover_env_cfg:RecoverEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PivotVQRRecoverPPORunnerCfg"
        ),
    },
)