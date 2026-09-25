# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific termination terms for velocity environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .fsm import VQRFsmState
from .observations import yaw_fsm_unsafe

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


class FSMUnsafeWithGrace(ManagerTermBase):
    """Phase-aware unsafe termination using the live FSM safety predicate.

    Phases A/B terminate after the short reset grace.  Phase C intentionally
    never terminates for this predicate: the command FSM receives the same
    unsafe signal and enters ``SAFE_RECOVERY`` instead.  This class never
    reads ``command.unsafe`` because that buffer can be one simulation step
    old when the termination manager runs.
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
        robot_name: str,
        torso_sensor_cfg: SceneEntityCfg,
        threshold: float,
        grace_period_s: float,
        minimum_base_height: float,
        unsafe_angle_limit: float,
    ) -> torch.Tensor:
        if grace_period_s < 0.0:
            raise ValueError("grace_period_s must be non-negative.")
        unsafe = yaw_fsm_unsafe(
            env,
            robot_name=robot_name,
            torso_sensor_cfg=torso_sensor_cfg,
            minimum_base_height=minimum_base_height,
            unsafe_angle_limit=unsafe_angle_limit,
            contact_threshold=threshold,
        )
        phase_value = getattr(env, "_yaw_fsm_task_curriculum_phase", 0)
        phase = int(phase_value.item()) if torch.is_tensor(phase_value) else int(phase_value)
        grace_finished = self._elapsed_s + 1.0e-6 >= grace_period_s
        self._elapsed_s += env.step_dt
        if phase >= 2:
            return torch.zeros_like(unsafe, dtype=torch.bool)
        return grace_finished & unsafe


class SwingContactTimeout(ManagerTermBase):
    """Terminate persistent swing-wheel contact during either YAW state.

    Contact timers are independent for the two swing wheels.  They are reset
    as soon as the FSM leaves YAW or that wheel loses contact, making the
    timeout a continuous-contact condition rather than an episode counter.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._contact_time = torch.zeros(env.num_envs, 2, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._contact_time.zero_()
        else:
            self._contact_time[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        command_name: str = "yaw_rate_cmd",
        threshold: float = 1.0,
        timeout_s: float = 0.20,
    ) -> torch.Tensor:
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative.")
        if timeout_s < 0.0:
            raise ValueError("timeout_s must be non-negative.")
        command = env.command_manager.get_term(command_name)
        if not hasattr(command, "fsm_state") or not hasattr(command, "support_diagonal"):
            raise TypeError(
                f"Command '{command_name}' does not expose FSM state/diagonal buffers."
            )
        state = torch.as_tensor(command.fsm_state, device=env.device)
        diagonal = torch.as_tensor(command.support_diagonal, device=env.device)
        if state.shape != (env.num_envs,) or diagonal.shape != (env.num_envs,):
            raise ValueError("FSM state and support_diagonal must have shape (num_envs,).")

        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        force = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
        if force.ndim != 3 or force.shape[1] != 4:
            raise ValueError(
                "SwingContactTimeout requires four wheels ordered FL, FR, HL, HR."
            )
        contact = torch.linalg.vector_norm(force, dim=-1) > threshold
        # POS supports FL+HR, therefore swings FR+HL. NEG is its mirror.
        pos_swing = contact[:, (1, 2)]
        neg_swing = contact[:, (0, 3)]
        swing_contact = torch.where((diagonal == 1).unsqueeze(1), pos_swing, neg_swing)
        in_yaw = (state == int(VQRFsmState.YAW_POS)) | (state == int(VQRFsmState.YAW_NEG))
        active_contact = swing_contact & in_yaw.unsqueeze(1)
        self._contact_time.copy_(torch.where(
            active_contact,
            self._contact_time + env.step_dt,
            torch.zeros_like(self._contact_time),
        ))
        return (self._contact_time >= timeout_s).any(dim=1)


def fsm_transition_timeout(
    env: ManagerBasedRLEnv,
    command_name: str = "yaw_rate_cmd",
    timeout_s: float = 3.0,
) -> torch.Tensor:
    """Terminate environments stuck in either FSM transition state.

    ``state_time`` is owned by the command term and is reset with the FSM, so
    this term remains independent of randomized episode lengths and command
    resampling.  All non-transition states return ``False``.
    """
    if timeout_s < 0.0:
        raise ValueError("timeout_s must be non-negative.")

    command_term = env.command_manager.get_term(command_name)
    missing = [
        name
        for name in ("fsm_state", "state_time")
        if not hasattr(command_term, name)
    ]
    if missing:
        raise TypeError(
            f"Command '{command_name}' does not expose FSM buffers {missing}; "
            "use YawFSMCommand for the FSM task."
        )

    device = getattr(env, "device", None)
    state = torch.as_tensor(getattr(command_term, "fsm_state"), device=device)
    state_time = torch.as_tensor(getattr(command_term, "state_time"), device=device)
    expected_shape = (env.num_envs,)
    if state.shape != expected_shape or state_time.shape != expected_shape:
        raise ValueError(
            "FSM state and state_time must each have shape "
            f"{expected_shape}; got {tuple(state.shape)} and {tuple(state_time.shape)}"
        )

    in_transition = (state == int(VQRFsmState.TRANSITION_POS)) | (
        state == int(VQRFsmState.TRANSITION_NEG)
    )
    return in_transition & (state_time >= timeout_s)


def fsm_return_timeout(
    env: ManagerBasedRLEnv,
    command_name: str = "yaw_rate_cmd",
    timeout_s: float = 2.5,
) -> torch.Tensor:
    """Terminate an episode that cannot regain four-wheel stance in time."""
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive.")

    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term, "fsm_state") or not hasattr(command_term, "state_time"):
        raise TypeError(f"Command '{command_name}' does not expose FSM state/time buffers.")
    state = torch.as_tensor(command_term.fsm_state, device=env.device)
    state_time = torch.as_tensor(command_term.state_time, device=env.device)
    if state.shape != (env.num_envs,) or state_time.shape != (env.num_envs,):
        raise ValueError("FSM state and state_time must each have shape (num_envs,).")
    return (state == int(VQRFsmState.RETURN_TO_4)) & (state_time >= timeout_s)
