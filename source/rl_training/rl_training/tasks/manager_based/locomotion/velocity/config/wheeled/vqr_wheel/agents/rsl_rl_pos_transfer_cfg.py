"""Base PPO settings with an independent transfer experiment namespace."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import VQRWheelYawFlatPPORunnerCfg


@configclass
class VQRWheelYawPosTransferPPORunnerCfg(VQRWheelYawFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "vqr_wheel_yaw_flat_pos_transfer"
        self.resume = False
