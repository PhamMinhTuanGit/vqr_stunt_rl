# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from .fsm import VQRFsmState, VQRYawFSM

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

    cfg: mdp.UniformThresholdVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Additional metrics for TensorBoard.
        self.metrics["base_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["knee_pos"] = torch.zeros(self.num_envs, device=self.device)
        self._metric_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        knee_joint_ids = self.robot.find_joints(".*[Kk]nee.*")[0]
        self._knee_joint_ids = torch.tensor(knee_joint_ids, dtype=torch.long, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)

        extras = {}
        for metric_name, metric_value in self.metrics.items():
            if metric_name in {"base_z", "knee_pos"}:
                step_count = torch.clamp(self._metric_step_counter[env_ids].float(), min=1.0)
                extras[metric_name] = torch.mean(metric_value[env_ids] / step_count).item()
            else:
                extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            metric_value[env_ids] = 0.0

        self._metric_step_counter[env_ids] = 0
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return extras

    def _update_metrics(self):
        super()._update_metrics()

        # 1) base_z metric: root_pos_w[:, 2]
        base_z = self.robot.data.root_pos_w[:, 2]

        # 2) knee_pos metric: same formulation as joint_pos_penalty for knee joints
        cmd = torch.linalg.norm(self.vel_command_b, dim=1)
        body_vel = torch.linalg.norm(self.robot.data.root_lin_vel_b[:, :2], dim=1)

        if self._knee_joint_ids.numel() > 0:
            running_reward = torch.linalg.norm(
                self.robot.data.joint_pos[:, self._knee_joint_ids]
                - self.robot.data.default_joint_pos[:, self._knee_joint_ids],
                dim=1,
            )
        else:
            running_reward = torch.zeros(self.num_envs, device=self.device)

        knee_pos = torch.where(
            torch.logical_or(cmd > 0.1, body_vel > 0.5),
            running_reward,
            5.0 * running_reward,
        )

        self.metrics["base_z"] += base_z
        self.metrics["knee_pos"] += knee_pos
        self._metric_step_counter += 1

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """


class YawRateCommand(CommandTerm):
    """Sample target yaw rate [rad/s].

    command shape: (num_envs, 1)
    command[:, 0] = yaw_rate_cmd
    """

    cfg: "YawRateCommandCfg"

    def __init__(self, cfg: "YawRateCommandCfg", env):
        super().__init__(cfg, env)

        self.robot = env.scene[cfg.asset_name]

        self._command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["error_yaw_rate"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new yaw-rate target."""
        # Advanced indexing returns a copy, so calling ``uniform_`` directly on
        # ``self._command[env_ids, 0]`` leaves the command buffer unchanged.
        sampled = torch.empty_like(self._command[env_ids, 0]).uniform_(*self.cfg.yaw_rate_range)
        self._command[env_ids, 0] = sampled

    def _update_command(self):
        # Command is constant until next resampling.
        pass

    def _update_metrics(self):
        wz = self.robot.data.root_ang_vel_b[:, 2]

        self.metrics["error_yaw_rate"] = torch.abs(wz - self._command[:, 0])


@configclass
class YawRateCommandCfg(CommandTermCfg):
    """Configuration for a scalar body-frame yaw-rate command."""

    class_type: type = YawRateCommand
    asset_name: str = "robot"
    yaw_rate_range: tuple[float, float] = (-1.0, 1.0)


class YawFSMCommand(YawRateCommand):
    """Yaw-rate command coupled to the per-environment yaw FSM.

    ``VQRYawFSM`` is deliberately kept as the single source of truth for the
    transition rules.  The readiness predicates are buffers rather than
    callbacks so they can be computed from batched Isaac Lab sensor data by
    the task and assigned without creating one Python callback per environment.

    The command itself remains ``(num_envs, 1)`` and is therefore compatible
    with existing yaw observations and rewards.
    """

    cfg: "YawFSMCommandCfg"

    def __init__(self, cfg: "YawFSMCommandCfg", env):
        super().__init__(cfg, env)

        self._fsm = [
            VQRYawFSM(yaw_enter=cfg.yaw_enter, yaw_exit=cfg.yaw_exit)
            for _ in range(self.num_envs)
        ]
        self._fsm_state = torch.full(
            (self.num_envs,), int(VQRFsmState.FOUR_STAND), dtype=torch.int64, device=self.device
        )

        # These are public on purpose: task-specific sensor code can update
        # all predicates in one batched operation before the command update.
        self.positive_pose_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.negative_pose_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.four_stand_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.unsafe = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def fsm_state(self) -> torch.Tensor:
        """Current FSM state for every environment as ``VQRFsmState`` values."""
        return self._fsm_state

    def set_fsm_inputs(
        self,
        positive_pose_ready: torch.Tensor,
        negative_pose_ready: torch.Tensor,
        four_stand_ready: torch.Tensor,
        unsafe: torch.Tensor,
    ) -> None:
        """Set the batched readiness predicates consumed on the next update."""
        values = (
            (positive_pose_ready, self.positive_pose_ready),
            (negative_pose_ready, self.negative_pose_ready),
            (four_stand_ready, self.four_stand_ready),
            (unsafe, self.unsafe),
        )
        for value, target in values:
            if value.shape != target.shape:
                raise ValueError(f"FSM predicate must have shape {tuple(target.shape)}, got {tuple(value.shape)}")
            target.copy_(value.to(device=self.device, dtype=torch.bool))

    def _reset_fsm(self, env_ids: Sequence[int] | slice) -> None:
        ids = range(self.num_envs) if isinstance(env_ids, slice) else env_ids
        for env_id in ids:
            self._fsm[int(env_id)].state = VQRFsmState.FOUR_STAND
        self._fsm_state[env_ids] = int(VQRFsmState.FOUR_STAND)

        self.positive_pose_ready[env_ids] = False
        self.negative_pose_ready[env_ids] = False
        self.four_stand_ready[env_ids] = False
        self.unsafe[env_ids] = False

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self._reset_fsm(env_ids)

    def _update_command(self):
        """Apply exactly the transition ordering defined by ``fsm.py``."""
        yaw_cmd = self._command[:, 0].detach().cpu().tolist()
        positive = self.positive_pose_ready.detach().cpu().tolist()
        negative = self.negative_pose_ready.detach().cpu().tolist()
        four_stand = self.four_stand_ready.detach().cpu().tolist()
        unsafe = self.unsafe.detach().cpu().tolist()

        for env_id, fsm in enumerate(self._fsm):
            self._fsm_state[env_id] = int(
                fsm.update(yaw_cmd[env_id], positive[env_id], negative[env_id], four_stand[env_id], unsafe[env_id])
            )


@configclass
class YawFSMCommandCfg(YawRateCommandCfg):
    """Configuration for a yaw-rate command driven by ``VQRYawFSM``."""

    class_type: type = YawFSMCommand
    yaw_enter: float = 0.10
    yaw_exit: float = 0.05
