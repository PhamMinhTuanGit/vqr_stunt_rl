# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import math 
from typing import TYPE_CHECKING, Sequence, Any
from dataclasses import MISSING 
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

MISSING: Any = MISSING

class EpisodeLiftCommand(CommandTerm):
    """Expose a scheduled lift request without writing actions or robot state."""

    cfg: "EpisodeLiftCommandCfg"

    def __init__(self, cfg: "EpisodeLiftCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if not 0.0 <= cfg.hold_time_s < cfg.ramp_end_time_s:
            raise ValueError("Expected 0 <= hold_time_s < ramp_end_time_s for lift command.")
        self._command = torch.zeros((self.num_envs, 1), device=self.device)
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._contact_sensor: ContactSensor = env.scene.sensors[cfg.contact_sensor_name]
        self._wheel_contact_ids = self._contact_sensor.find_bodies(
            cfg.wheel_body_names, preserve_order=True
        )[0]
        self._support_contact_ids = self._contact_sensor.find_bodies(
            cfg.support_wheel_names, preserve_order=True
        )[0]
        self._lifted_contact_ids = self._contact_sensor.find_bodies(
            cfg.lifted_wheel_names, preserve_order=True
        )[0]
        self._lifted_body_ids = self._robot.find_bodies(
            cfg.lifted_wheel_names, preserve_order=True
        )[0]
        if len(self._wheel_contact_ids) != 4 or len(self._lifted_contact_ids) != 2:
            raise ValueError("M3 metrics require four ordered wheels and two lifted wheels.")

        metric_names = (
            "fl_contact",
            "fr_contact",
            "hl_contact",
            "hr_contact",
            "fr_clearance",
            "hl_clearance",
            "four_wheel_support_score",
            "fl_hr_support_score",
            "fr_lifted_score",
            "hl_lifted_score",
            "minimum_lifted_score",
            "lift_cmd",
            "yaw_rate_cmd",
            "actual_yaw_rate",
            "yaw_error",
            "xy_displacement",
            "planar_velocity",
        )
        for name in metric_names:
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        self._metric_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def __str__(self) -> str:
        return (
            "EpisodeLiftCommand:\n"
            f"\tHold time: {self.cfg.hold_time_s} s\n"
            f"\tRamp end: {self.cfg.ramp_end_time_s} s"
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Log per-episode metric means and restart the lift schedule."""
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        step_count = torch.clamp(self._metric_step_counter[env_ids].float(), min=1.0)
        extras = {
            name: torch.mean(values[env_ids] / step_count).item()
            for name, values in self.metrics.items()
        }
        for values in self.metrics.values():
            values[env_ids] = 0.0
        self._metric_step_counter[env_ids] = 0
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return extras

    def _update_metrics(self):
        wheel_forces = self._contact_sensor.data.net_forces_w[:, self._wheel_contact_ids]
        wheel_contacts = (
            torch.linalg.vector_norm(wheel_forces, dim=-1) > self.cfg.contact_threshold
        ).float()
        support_forces = self._contact_sensor.data.net_forces_w[:, self._support_contact_ids]
        support_contacts = (
            torch.linalg.vector_norm(support_forces, dim=-1) > self.cfg.contact_threshold
        ).float()
        lifted_forces = self._contact_sensor.data.net_forces_w[:, self._lifted_contact_ids]
        lifted_contacts = (
            torch.linalg.vector_norm(lifted_forces, dim=-1) > self.cfg.contact_threshold
        )

        lifted_height = self._robot.data.body_pos_w[:, self._lifted_body_ids, 2]
        ground_height = self._env.scene.env_origins[:, 2].unsqueeze(-1)
        lifted_clearance = lifted_height - ground_height - self.cfg.wheel_radius
        clearance_score = torch.clamp(
            lifted_clearance / self.cfg.target_clearance, min=0.0, max=1.0
        )
        lifted_scores = 0.5 * (~lifted_contacts).float() + 0.5 * clearance_score

        yaw_command = self._env.command_manager.get_command(self.cfg.yaw_command_name)[:, 2]
        actual_yaw_rate = self._robot.data.root_ang_vel_b[:, 2]
        if hasattr(self._env, "_pivot_reset_root_xy"):
            xy_displacement = torch.linalg.vector_norm(
                self._robot.data.root_pos_w[:, :2] - self._env._pivot_reset_root_xy,
                dim=1,
            )
        else:
            xy_displacement = torch.zeros(self.num_envs, device=self.device)
        planar_velocity = torch.linalg.vector_norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)

        for index, name in enumerate(("fl_contact", "fr_contact", "hl_contact", "hr_contact")):
            self.metrics[name] += wheel_contacts[:, index]
        self.metrics["fr_clearance"] += lifted_clearance[:, 0]
        self.metrics["hl_clearance"] += lifted_clearance[:, 1]
        self.metrics["four_wheel_support_score"] += wheel_contacts.mean(dim=1)
        self.metrics["fl_hr_support_score"] += support_contacts.mean(dim=1)
        self.metrics["fr_lifted_score"] += lifted_scores[:, 0]
        self.metrics["hl_lifted_score"] += lifted_scores[:, 1]
        self.metrics["minimum_lifted_score"] += lifted_scores.amin(dim=1)
        self.metrics["lift_cmd"] += self._command[:, 0]
        self.metrics["yaw_rate_cmd"] += yaw_command
        self.metrics["actual_yaw_rate"] += actual_yaw_rate
        self.metrics["yaw_error"] += torch.abs(actual_yaw_rate - yaw_command)
        self.metrics["xy_displacement"] += xy_displacement
        self.metrics["planar_velocity"] += planar_velocity
        self._metric_step_counter += 1

    def _resample_command(self, env_ids: Sequence[int]):
        self._command[env_ids] = 0.0

    def _update_command(self):
        episode_time = self._env.episode_length_buf.float() * self._env.step_dt
        ramp_duration = self.cfg.ramp_end_time_s - self.cfg.hold_time_s
        self._command[:, 0] = torch.clamp(
            (episode_time - self.cfg.hold_time_s) / ramp_duration,
            min=0.0,
            max=1.0,
        )


@configclass
class EpisodeLiftCommandCfg(CommandTermCfg):
    """Configuration for the M3 episode-time lift schedule."""

    class_type: type = EpisodeLiftCommand
    hold_time_s: float = 0.5
    ramp_end_time_s: float = 2.0
    asset_name: str = "robot"
    contact_sensor_name: str = "contact_forces"
    wheel_body_names: list[str] = MISSING
    support_wheel_names: list[str] = MISSING
    lifted_wheel_names: list[str] = MISSING
    yaw_command_name: str = "yaw_rate_cmd"
    wheel_radius: float = MISSING
    target_clearance: float = 0.05
    contact_threshold: float = 1.0

    def __post_init__(self):
        # CommandManager samples this interval with torch.uniform_, which does
        # not accept infinity.  This finite interval is far beyond any episode
        # duration, so the term is reset only when the environment resets.
        self.resampling_time_range = (1.0e6, 1.0e6)


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
#Phần command cho task pivot 
#
class PivotCommand(CommandTerm):
    """Lệnh: [yaw_rate*, pitch*, sin(phi), cos(phi)]"""
    cfg: "PivotCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env) #Khởi tạo các biến cần thiết 
        self.robot = env.scene[cfg.asset_name] 
        self._cmd  = torch.zeros(self.num_envs, 4, device=self.device)
        self._freq = torch.zeros(self.num_envs, device=self.device)
        self._phase = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_yaw_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"]    = torch.zeros(self.num_envs, device=self.device)

    def __str__(self):
        return f"PivotCommand: yaw{self.cfg.ang_vel_z} pitch{self.cfg.pitch_target}"

    @property
    def command(self) -> torch.Tensor:
        return self._cmd                       # (N, 4) (4,4) 

    @property
    def pitch_cmd(self) -> torch.Tensor:
        return self._cmd[:, 1]

    def _resample_command(self, env_ids):
        n = len(env_ids)
        r = lambda rng: torch.empty(n, device=self.device).uniform_(*rng)
        self._cmd[env_ids, 0] = r(self.cfg.ang_vel_z)
        self._cmd[env_ids, 1] = r(self.cfg.pitch_target)
        self._freq[env_ids]   = r(self.cfg.gesture_freq)
        self._phase[env_ids]  = torch.rand(n, device=self.device) * 2 * math.pi
        # một phần env yêu cầu đứng yên (yaw=0) để giữ kỹ năng thăng bằng thuần
        zero = torch.rand(n, device=self.device) < self.cfg.rel_standing_envs
        self._cmd[env_ids[zero], 0] = 0.0

    def _update_command(self):
        self._phase = (self._phase + 2 * math.pi * self._freq * self._env.step_dt) % (2 * math.pi) #(phi_k+1 = phi_k + 2*pi*f*dt)%(2*pi) chioa lấy phần dư của 2 pi lấy pha hiện tại
        self._cmd[:, 2] = torch.sin(self._phase)
        self._cmd[:, 3] = torch.cos(self._phase)

    def _update_metrics(self):
        self.metrics["error_yaw_rate"] += torch.abs(
            self._cmd[:, 0] - self.robot.data.root_ang_vel_b[:, 2]) / self._env.max_episode_length
        g = self.robot.data.projected_gravity_b
        pitch = torch.atan2(-g[:, 0], -g[:, 2])
        self.metrics["error_pitch"] += torch.abs(self._cmd[:, 1] - pitch) / self._env.max_episode_length

    def _set_debug_vis_impl(self, debug_vis):   # bỏ qua cho gọn
        pass


@configclass
class PivotCommandCfg(CommandTermCfg):
    class_type: type = PivotCommand
    asset_name: str = MISSING
    ang_vel_z: tuple[float, float]    = (-3.0, 3.0)
    pitch_target: tuple[float, float] = (0.9, 1.2)
    gesture_freq: tuple[float, float] = (0.5, 1.5)
    rel_standing_envs: float = 0.2


# ---------------------------------------------------------------------------
# Four-mode command (spec section 7): mode one-hot(4) | omega_z* | delta_theta* | tuck*
# ---------------------------------------------------------------------------

GROUND = 0
REAR_UP = 1
BALANCE = 2
LAND = 3
NUM_MODES = 4
COMMAND_DIM = NUM_MODES + 3  # 8
OMEGA_Z_IDX = 4
DELTA_THETA_IDX = 5
TUCK_IDX = 6

class PivotModeCommand(CommandTerm):
    """Four-mode pivot command: mode one-hot(4) | omega_z* | delta_theta* | tuck*.

    The supervisor (environment) owns the mode schedule; this term samples
    the continuous setpoints around it and exposes the 8-dim command vector.
    """

    def __init__(self, cfg: "PivotModeCommandCfg", env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self._cmd = torch.zeros(self.num_envs, COMMAND_DIM, device=self.device)
        self._mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._omega_z_ramp = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_omega_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_delta_theta"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self):
        return f"PivotModeCommand(mode={self.cfg.mode_schedule})"

    @property
    def command(self) -> torch.Tensor:
        return self._cmd

    @property
    def mode(self) -> torch.Tensor:
        return self._mode

    @property
    def omega_z_command(self) -> torch.Tensor:
        return self._cmd[:, NUM_MODES]

    @property
    def delta_theta_command(self) -> torch.Tensor:
        return self._cmd[:, NUM_MODES + 1]

    @property
    def tuck_command(self) -> torch.Tensor:
        return self._cmd[:, NUM_MODES + 2]

    # -- supervisor hooks -------------------------------------------------
    def set_mode(self, env_ids: torch.Tensor, modes: torch.Tensor) -> None:
        """Supervisor (L0-L3 arbitration) may override the scheduled mode."""
        self._mode[env_ids] = modes
        one_hot = torch.nn.functional.one_hot(self._mode, NUM_MODES).to(self._cmd.dtype)
        self._cmd[env_ids, :NUM_MODES] = one_hot[env_ids]

    def _resample_command(self, env_ids):
        n = len(env_ids)
        device = self.device
        r = lambda rng: torch.empty(n, device=device).uniform_(*rng)
        # Continuous setpoints; the mode stays whatever the schedule/supervisor set.
        # omega_z* <= |6| BALANCE / |3| GROUND (invariant I10) via clamp at write.
        self._cmd[env_ids, NUM_MODES] = r(self.cfg.omega_z_range).clamp_(
            -self.cfg.omega_z_limit, self.cfg.omega_z_limit
        )
        # delta_theta* in +/-8 deg around theta*(omega_z) (section 7).
        self._cmd[env_ids, NUM_MODES + 1] = r(self.cfg.delta_theta_range).clamp_(
            -self.cfg.delta_theta_limit, self.cfg.delta_theta_limit
        )
        # tuck* in [0, 1]: 0 = open stance, 1 = fully tucked.
        self._cmd[env_ids, NUM_MODES + 2] = r((0.0, 1.0))
        # Keep the one-hot consistent with resampled modes.
        one_hot = torch.nn.functional.one_hot(self._mode[env_ids], NUM_MODES).to(self._cmd.dtype)
        self._cmd[env_ids, :NUM_MODES] = one_hot
        # A share of envs gets a pure hold (omega_z* = 0) to preserve the static
        # balance skill (S2 gate: r_xi > 0.7 for 20 s).
        hold = torch.rand(n, device=device) < self.cfg.rel_standing_envs
        self._cmd[env_ids[hold], NUM_MODES] = 0.

    def _update_command(self):
        self._update_mode()

        one_hot = torch.nn.functional.one_hot(
            self._mode,
            NUM_MODES,
        ).to(self._cmd.dtype)

        self._cmd[:, :NUM_MODES] = one_hot

        # I10
        omega = self._cmd[:, OMEGA_Z_IDX]

        ground = self._mode == GROUND
        balance = self._mode == BALANCE

        omega[ground] = torch.clamp(
            omega[ground], -3.0, 3.0
        )

        omega[balance] = torch.clamp(
            omega[balance], -6.0, 6.0
        )

        self._cmd[:, OMEGA_Z_IDX] = omega

    def _update_metrics(self):
        self.metrics["error_omega_z"] += torch.abs(
            self._cmd[:, NUM_MODES] - self.robot.data.root_ang_vel_b[:, 2]
        ) / self._env.max_episode_length
        # delta_theta error versus realized pitch offset around theta*: reads
        # the cache when available (I8) and falls back to raw attitude otherwise.
        cache = getattr(self._env.unwrapped, "pivot_cache", None)
        if cache is not None and getattr(cache, "_updated", False):
            realized = cache.pitch - cache.theta_star
        else:
            g = self.robot.data.projected_gravity_b
            pitch = torch.atan2(g[:, 0], -g[:, 2])
            realized = pitch  # theta* term unavailable pre-cache; coarse error only
        self.metrics["error_delta_theta"] += torch.abs(
            self._cmd[:, NUM_MODES + 1] - realized
        ) / self._env.max_episode_length

    def _set_debug_vis_impl(self, debug_vis):
        pass
    
    def _update_mode(self):
        t = self._env.episode_length_buf * self._env.step_dt

        ground_end = self.cfg.ground_duration_s

        rear_up_end = (
            ground_end
            + self.cfg.rear_up_duration_s
        )

        balance_end = (
            rear_up_end
            + self.cfg.balance_duration_s
        )

        mode = torch.full_like(self._mode, LAND)

        mode = torch.where(
            t < balance_end,
            BALANCE,
            mode,
        )

        mode = torch.where(
            t < rear_up_end,
            REAR_UP,
            mode,
        )

        mode = torch.where(
            t < ground_end,
            GROUND,
            mode,
        )

        self._mode[:] = mode


@configclass
class PivotModeCommandCfg(CommandTermCfg):
    class_type: type = PivotModeCommand
    asset_name: str = MISSING
    # |omega_z*| capped per mode at runtime (I10): 3 GROUND, 6 BALANCE.
    omega_z_range: tuple[float, float] = (-6.0, 6.0)
    omega_z_limit: float = 6.0
    delta_theta_range: tuple[float, float] = (-0.1396, 0.906)
    delta_theta_limit: float = 0.906
    rel_standing_envs: float = 0.2
    rear_up_duration_s = 
    mode_schedule: tuple[str, ...] = ("GROUND", "REAR_UP", "BALANCE", "LAND")

