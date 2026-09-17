# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Custom leg action terms shared by the pivot tasks."""

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


class PriorResidualJointPositionAction(JointPositionAction):
    """Leg targets = q_prior(mode, tuck*) + 0.3 * residual (invariant I3).

    The residual scale is fixed at 0.3 by the specification: the policy only
    nudges around the mode-interpolated prior, never steers the whole pose.
    """

    cfg: "PriorResidualJointPositionActionCfg"

    def __init__(self, cfg: "PriorResidualJointPositionActionCfg", env):
        super().__init__(cfg, env)
        # Keep JointAction._scale intact: it represents cfg.scale and is used
        # by the base action processing/IO descriptor.
        self._residual_scale = cfg.residual_scale
        self._command_name = cfg.command_name
        self._joint_names_ordered = list(self._joint_names)

    @property
    def prior_command(self):
        return self._env.command_manager.get_term(self._command_name)

    def compute(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        # Let JointAction store raw actions and apply cfg scale/offset/clip.
        super().process_actions(actions)

        # Pull the mode and tuck from the pivot command term.
        term = self.prior_command
        from .leg_prior import q_prior  # local import avoids a cycle

        prior = q_prior(
            term.mode,
            term.tuck_command,
            self._joint_names_ordered,
            getattr(self.cfg, "reference_path", None),
        ).to(device=self.device, dtype=self._processed_actions.dtype)
        # Fold the prior in as a dynamic offset and shrink the already-scaled
        # policy residual by the pivot residual scale.
        self._processed_actions = prior + self._residual_scale * self._processed_actions
        # Optionally clamp to runtime soft limits like the soft-limit term.
        if self.cfg.enforce_soft_limits:
            limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
            self._processed_actions = torch.clamp(
                self._processed_actions, min=limits[..., 0], max=limits[..., 1]
            )


@configclass
class PriorResidualJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the q_prior + scaled-residual leg action."""

    class_type: type = PriorResidualJointPositionAction
    command_name: str = MISSING
    # Spec-fixed residual scale (I3); do not tune without re-auditing.
    residual_scale: float = 0.3
    # Base scale of JointPositionActionCfg applies on top of the residual scale.
    enforce_soft_limits: bool = True
    reference_path: str | None = None
