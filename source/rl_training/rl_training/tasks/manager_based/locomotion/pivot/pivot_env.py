# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""PivotEnv: ManagerBasedRLEnv extension owning the PivotStateCache (I8).

``_pre_physics_step`` computes zeta/zeta_dot/xi, the thermal EMA, clearance,
contacts, and theta*(omega_z_cmd) ONCE; every observation/reward/termination
term only reads the cache afterwards.  The mode schedule (GROUND -> REAR_UP ->
BALANCE -> LAND, RECOVER handled by pi_recover) runs through the pure
``supervisor_gate`` so train and deploy share one decision path (I9).
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .state.cache import PivotStateCache
from .state.hard_state_buffer import HardStateBuffer
from .mdp.supervisor_gate import RateLimiter, mode_transition

WHEEL_BODY_ORDER = ("FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL")
REAR_WHEELS = ("HL_WHEEL", "HR_WHEEL")
FRONT_WHEELS = ("FL_WHEEL", "FR_WHEEL")


class _HistoryStack:
    """Ring buffer of the last H (zeta, zeta_dot, xi) frames (I8 reader)."""

    def __init__(self, num_envs: int, device, length: int = 5):
        self.length = length
        self.buffer = torch.zeros(num_envs, length, 3, device=device)
        self.index = 0
        self.filled = 0

    def push(self, frame: torch.Tensor) -> None:
        slot = self.index % self.length
        self.buffer[:, slot] = frame
        self.index += 1
        self.filled = min(self.filled + 1, self.length)

    def concat(self, current: torch.Tensor) -> torch.Tensor:
        """(N, 3*H): oldest -> newest, newest = the passed current frame."""
        frames = []
        for k in range(self.filled):
            slot = (self.index - 1 - k) % self.length
            frames.append(self.buffer[:, slot])
        frames.reverse()
        frames.append(current)
        while len(frames) < self.length:
            frames.insert(0, frames[0])  # pad at the oldest end
        return torch.cat(frames[-self.length:], dim=-1)

    def reset(self, env_ids) -> None:
        self.buffer[env_ids] = 0.0
        if len(env_ids) == getattr(self.buffer, "shape", [0])[0]:
            self.index = 0
            self.filled = 0


class PivotEnv(ManagerBasedRLEnv):
    """Four-mode pivot environment with a per-step state cache."""

    def __init__(self, cfg, render_mode: int | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        robot = self.scene["robot"]
        self.pivot_cache = PivotStateCache(
            self.num_envs, device=self.device, mu_hat=cfg.mu_hat_default
        )
        self._pivot_cmd_term = self._find_pivot_command_term()

        # Body indices for the rear/front wheel pairs (anchor/drift/clearance).
        body_names = list(robot.data.body_names)
        self.pivot_rear_wheel_ids = torch.tensor(
            [body_names.index(name) for name in REAR_WHEELS], device=self.device
        )
        self.pivot_front_wheel_ids = torch.tensor(
            [body_names.index(name) for name in FRONT_WHEELS], device=self.device
        )
        self.pivot_wheel_body_ids = torch.tensor(
            [body_names.index(name) for name in WHEEL_BODY_ORDER], device=self.device
        )
        # Wheel joint indices for applied-torque readback.
        joint_names = list(robot.data.joint_names)
        self.pivot_wheel_joint_ids = torch.tensor(
            [joint_names.index(name) for name in ("FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL")],
            device=self.device,
        )

        # Supervisor state (shared decision path with deployment, I9).
        self._pivot_stage = 0
        self.pivot_rate_limiter = RateLimiter(
            velocity_limit=6.0, dt=self.step_dt, num_envs=self.num_envs, device=self.device
        )
        self.pivot_hard_states = self.pivot_cache.hard_states
        self.pivot_history = _HistoryStack(
            self.num_envs, self.device, length=cfg.history_length
        )
        self.pivot_clearance = None  # analytic capsule clearance, bound lazily

    # ------------------------------------------------------------------
    def _find_pivot_command_term(self):
        for name in getattr(self.cfg.commands, "__dict__", {}).keys():
            if name.startswith("_"):
                continue
            try:
                term = self.command_manager.get_term(name)
            except Exception:
                continue
            if type(term).__name__ == "PivotModeCommand":
                return term
        return None

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Refresh every cached quantity once per control step (I8)."""
        robot = self.scene["robot"]
        cache = self.pivot_cache

        com_pos_w = robot.data.body_com_pos_w[:, robot.root_idx]
        com_vel_w = robot.data.body_com_vel_w[:, robot.root_idx]
        rear_contact = robot.data.body_pos_w[:, self.pivot_rear_wheel_ids, :].mean(dim=1)
        heading = self._heading_w(robot)

        sensor = self.scene.sensors["contact_forces"]
        forces = sensor.data.net_forces_w_history[-1][:, self.pivot_wheel_body_ids]
        flags = (torch.linalg.vector_norm(forces, dim=-1) > 1.0).float()
        wheel_torques = robot.data.applied_torque[:, self.pivot_wheel_joint_ids]

        cmd_term = self._pivot_cmd_term
        omega_z_cmd = (
            cmd_term.omega_z_command
            if cmd_term is not None
            else torch.zeros(self.num_envs, device=self.device)
        )

        cache.update(
            com_pos_w=com_pos_w,
            com_vel_w=com_vel_w,
            rear_contact_pos_w=rear_contact,
            heading_w=heading,
            proj_gravity_b=robot.data.projected_gravity_b,
            wheel_torques=wheel_torques,
            wheel_contact_flags=flags,
            omega_z=robot.data.root_ang_vel_b[:, 2],
            omega_z_command=omega_z_cmd,
        )

        # Supervisor: rate-limit omega_z*, then arbitrate the mode (L0-L3).
        limited = self.pivot_rate_limiter(omega_z_cmd)
        mode = (
            cmd_term.mode
            if cmd_term is not None
            else torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        )
        new_mode = mode_transition(
            mode=mode,
            xi=cache.xi,
            xi_safe=cache.xi_safe,
            xi_max=cache.xi_max,
            ema_wheel_torque=cache.ema_wheel_torque.mean(dim=-1),
            thermal_limit=cache.physics.wheel_continuous_torque,
        )
        if cmd_term is not None:
            all_ids = torch.arange(self.num_envs, device=self.device)
            cmd_term.set_mode(all_ids, new_mode)
            cmd_term._cmd[:, NUM_MODES] = limited  # write rate-limited omega_z*

        # Balance-core history for the 5-frame observation stack.
        self.pivot_history.push(
            torch.stack([cache.zeta, cache.zeta_dot, cache.xi], dim=-1)
        )

        # Hard-state buffer for the 15% reset bucket (S6+).
        self.pivot_hard_states.update(
            root_state=robot.data.root_state_w,
            joint_pos=robot.data.joint_pos,
            joint_vel=robot.data.joint_vel,
        )

    # ------------------------------------------------------------------
    def _heading_w(self, robot) -> torch.Tensor:
        """Horizontal forward unit vector per env from the root quaternion."""
        quat = robot.data.root_quat_w  # (N, 4) wxyz
        forward = torch.zeros(self.num_envs, 3, device=self.device)
        forward[:, 0] = 1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2)
        forward[:, 1] = 2.0 * (quat[:, 1] * quat[:, 2] + quat[:, 0] * quat[:, 3])
        return torch.nn.functional.normalize(forward, dim=-1)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.pivot_cache.reset(env_ids)
        self.pivot_history.reset(env_ids)
        self.pivot_rate_limiter.reset(env_ids)
        self._pivot_cmd_term_cmd_reset(env_ids)

    def _pivot_cmd_term_cmd_reset(self, env_ids):
        if self._pivot_cmd_term is not None:
            self._pivot_cmd_term.reset(env_ids)


# Mode-slot count shared with commands.py (one-hot block size).
NUM_MODES = 4
