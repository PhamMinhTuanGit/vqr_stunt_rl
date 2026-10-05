"""Noise models whose live scale follows the isolated skill robustness stage."""

import torch
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseModel, NoiseModelCfg, AdditiveUniformNoiseCfg


class SkillNoiseModel(NoiseModel):
    def __init__(self, cfg, num_envs, device):
        super().__init__(cfg, num_envs, device)
        self.scale = 0.0
        self.bias = None

    def set_scale(self, scale):
        if not 0 <= scale <= 1:
            raise ValueError("Skill noise scale must be in [0, 1].")
        self.scale = float(scale)

    def reset(self, env_ids=None):
        if self.bias is not None:
            ids = slice(None) if env_ids is None else env_ids
            self.bias[ids] = 2 * torch.rand_like(self.bias[ids]) - 1

    def __call__(self, data):
        cfg = self._noise_model_cfg
        bounds = torch.full_like(data[0], cfg.bound * self.scale)
        if cfg.joint_names:
            if len(cfg.joint_names) != data.shape[-1]:
                raise ValueError("Skill velocity noise joint order/dimension mismatch.")
            wheels = data.new_tensor([name.endswith("_WHEEL") for name in cfg.joint_names], dtype=torch.bool)
            bounds[wheels] = 0.5
        result = data + (2 * torch.rand_like(data) - 1) * bounds
        if cfg.bias_bound:
            if self.bias is None:
                self.bias = torch.empty_like(data)
                self.reset()
            result = result + self.bias * cfg.bias_bound * self.scale
        return result


@configclass
class SkillNoiseCfg(NoiseModelCfg):
    class_type = SkillNoiseModel
    noise_cfg = AdditiveUniformNoiseCfg(n_min=0.0, n_max=0.0)
    bound: float = 0.0
    bias_bound: float = 0.0
    joint_names: tuple[str, ...] = ()
