# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms used only by the four-to-two-wheel pivot task."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass


class SoftLimitJointPositionAction(JointPositionAction):
    """Clamp selected processed position targets to runtime soft joint limits."""

    cfg: "SoftLimitJointPositionActionCfg"

    def __init__(self, cfg: "SoftLimitJointPositionActionCfg", env):
        super().__init__(cfg, env)
        missing_names = set(cfg.soft_limit_joint_names).difference(self._joint_names)
        if missing_names:
            raise ValueError(f"Soft-limit joints are not controlled by this action term: {missing_names}.")
        self._soft_limit_action_ids = [
            self._joint_names.index(name) for name in cfg.soft_limit_joint_names
        ]
        self._soft_limit_asset_ids = [
            self._joint_ids[index] for index in self._soft_limit_action_ids
        ]

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        limits = self._asset.data.soft_joint_pos_limits[:, self._soft_limit_asset_ids]
        if self.cfg.joint_limit_margin is not None:
            limits = self._asset.data.joint_pos_limits[:, self._soft_limit_asset_ids].clone()
            limits[..., 0] += self.cfg.joint_limit_margin
            limits[..., 1] -= self.cfg.joint_limit_margin
        self._processed_actions[:, self._soft_limit_action_ids] = torch.clamp(
            self._processed_actions[:, self._soft_limit_action_ids],
            min=limits[..., 0],
            max=limits[..., 1],
        )


@configclass
class SoftLimitJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for selective runtime soft-limit clamping."""

    class_type: type = SoftLimitJointPositionAction
    soft_limit_joint_names: list[str] = MISSING
    # Optional absolute safety margin; asset/actuator limits stay unchanged.
    joint_limit_margin: float | None = None
