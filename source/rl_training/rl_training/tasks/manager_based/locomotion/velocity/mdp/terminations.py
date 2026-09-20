# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific termination terms for velocity environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TorsoContactWithGrace(ManagerTermBase):
    """Terminate selected torso contact after a reset-relative grace period.

    The elapsed-time buffer is owned by this term and reset through the
    termination manager. It deliberately does not use ``episode_length_buf``,
    which may be randomized at startup by an RL runner.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._elapsed_s = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._elapsed_s.zero_()
        else:
            self._elapsed_s[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        threshold: float,
        grace_period_s: float,
    ) -> torch.Tensor:
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative.")
        if grace_period_s < 0.0:
            raise ValueError("grace_period_s must be non-negative.")

        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        force_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
        torso_contact = torch.linalg.vector_norm(force_history, dim=-1).amax(dim=1).amax(dim=1) > threshold
        grace_finished = self._elapsed_s + 1.0e-6 >= grace_period_s
        terminated = grace_finished & torso_contact
        self._elapsed_s += env.step_dt
        return terminated
