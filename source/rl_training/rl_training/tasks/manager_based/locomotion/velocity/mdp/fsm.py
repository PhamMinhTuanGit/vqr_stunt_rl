from __future__ import annotations

import torch

from enum import IntEnum
from typing import Sequence

try:
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils import configclass

    from .commands import YawRateCommand, YawRateCommandCfg
except ImportError:  # Keep scalar/vector FSM unit tests independent of Isaac Lab.
    def configclass(cls):
        return cls

    class YawRateCommand:
        pass

    class YawRateCommandCfg:
        pass

    class SceneEntityCfg:
        pass

__all__ = [
    "VQRFsmState",
    "VQRYawFSM",
    "YawFSMVectorized",
    "YawFSMCommand",
    "YawFSMCommandCfg",
    "select_swing_wheel_contact",
]


class VQRFsmState(IntEnum):
    FOUR_STAND = 0
    TRANSITION_POS = 1
    YAW_POS = 2
    TRANSITION_NEG = 3
    YAW_NEG = 4
    RETURN_TO_4 = 5
    SAFE_RECOVERY = 6


def select_swing_wheel_contact(
    positive_support_contact: torch.Tensor,
    negative_support_contact: torch.Tensor,
    support_diagonal: torch.Tensor,
) -> torch.Tensor:
    """Select whether any active swing wheel is in contact.

    The POS support pair is FL+HR, so its swing pair is the NEG support pair
    FR+HL; the NEG branch is the exact mirror.  Inputs retain the two
    per-wheel contact flags so an individual swing-wheel contact is not lost
    through a prior ``prod`` reduction.
    """
    if positive_support_contact.shape != negative_support_contact.shape:
        raise ValueError("POS/NEG contact tensors must have the same shape.")
    if positive_support_contact.ndim != 2 or positive_support_contact.shape[1] != 2:
        raise ValueError("Support contact tensors must have shape (num_envs, 2).")
    if support_diagonal.shape != positive_support_contact.shape[:1]:
        raise ValueError("support_diagonal must have shape (num_envs,).")

    pos_swing_contact = negative_support_contact.to(dtype=torch.bool).any(dim=1)
    neg_swing_contact = positive_support_contact.to(dtype=torch.bool).any(dim=1)
    return torch.where(
        support_diagonal == 1,
        pos_swing_contact,
        torch.where(
            support_diagonal == -1,
            neg_swing_contact,
            torch.zeros_like(pos_swing_contact),
        ),
    )


class VQRYawFSM:
    def __init__(
        self,
        yaw_enter=0.10,
        yaw_exit=0.05,
        dt=0.02,
        yaw_min_dwell=0.20,
        recovery_dwell=0.5,
    ):
        self.yaw_enter = yaw_enter
        self.yaw_exit = yaw_exit
        self.dt = dt
        self.yaw_min_dwell = yaw_min_dwell
        self.recovery_dwell = recovery_dwell
        self.state = VQRFsmState.FOUR_STAND
        self.state_time = 0.0
        self._recovery_safe_time = 0.0

    def update(
        self,
        yaw_cmd: float,
        positive_pose_ready: bool,
        negative_pose_ready: bool,
        four_stand_ready: bool,
        unsafe: bool,
    ):
        previous = self.state
        if unsafe:
            self.state = VQRFsmState.SAFE_RECOVERY
            self._recovery_safe_time = 0.0
        elif self.state == VQRFsmState.SAFE_RECOVERY:
            if four_stand_ready:
                self._recovery_safe_time += self.dt
            else:
                self._recovery_safe_time = 0.0

            if self._recovery_safe_time >= self.recovery_dwell:
                self.state = VQRFsmState.FOUR_STAND
                self._recovery_safe_time = 0.0

        elif self.state == VQRFsmState.FOUR_STAND:
            if yaw_cmd > self.yaw_enter:
                self.state = VQRFsmState.TRANSITION_POS

            elif yaw_cmd < -self.yaw_enter:
                self.state = VQRFsmState.TRANSITION_NEG

        elif self.state == VQRFsmState.TRANSITION_POS:
            if yaw_cmd < self.yaw_exit:
                self.state = VQRFsmState.RETURN_TO_4

            elif positive_pose_ready:
                self.state = VQRFsmState.YAW_POS

        elif self.state == VQRFsmState.YAW_POS:
            if self.state_time >= self.yaw_min_dwell and yaw_cmd < self.yaw_exit:
                self.state = VQRFsmState.RETURN_TO_4

        elif self.state == VQRFsmState.TRANSITION_NEG:
            if yaw_cmd > -self.yaw_exit:
                self.state = VQRFsmState.RETURN_TO_4

            elif negative_pose_ready:
                self.state = VQRFsmState.YAW_NEG

        elif self.state == VQRFsmState.YAW_NEG:
            if self.state_time >= self.yaw_min_dwell and yaw_cmd > -self.yaw_exit:
                self.state = VQRFsmState.RETURN_TO_4

        elif self.state == VQRFsmState.RETURN_TO_4:
            if four_stand_ready:
                self.state = VQRFsmState.FOUR_STAND

        self.state_time = 0.0 if self.state != previous else self.state_time + self.dt
        return self.state


class YawFSMVectorized:
    """Run the yaw FSM for a batch of environments using torch tensors only.

    The transition predicates and all persistent state live on ``device``.
    ``update`` evaluates every environment with tensor masks and never
    synchronizes one environment at a time.

    Args:
        num_envs: Number of independent FSM instances in the batch.
        device: Device on which the FSM buffers are allocated.
        yaw_enter: Positive command threshold that starts a transition.
        yaw_exit: Hysteresis threshold used to abort a transition.
        dt: Duration represented by one update, in seconds.
        yaw_min_dwell: Minimum duration in a YAW state before a command-driven
            exit to ``RETURN_TO_4`` is accepted. Unsafe always bypasses it.
        recovery_dwell: Continuous safe/four-wheel-ready time required to
            leave ``SAFE_RECOVERY``.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str = "cpu",
        yaw_enter: float = 0.10,
        yaw_exit: float = 0.05,
        dt: float = 0.02,
        yaw_min_dwell: float = 0.20,
        recovery_dwell: float = 0.5,
    ):
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}")

        self.device = torch.device(device)
        # Keep the hysteresis comparisons at Python-scalar precision.  This
        # matches the scalar oracle when a float32 command is materialized as
        # a Python value, including right on a threshold.
        self.yaw_enter = torch.as_tensor(yaw_enter, dtype=torch.float64, device=self.device)
        self.yaw_exit = torch.as_tensor(yaw_exit, dtype=torch.float64, device=self.device)
        self.dt = torch.as_tensor(dt, dtype=torch.float32, device=self.device)
        self.yaw_min_dwell = torch.as_tensor(
            yaw_min_dwell, dtype=torch.float32, device=self.device
        )
        self.recovery_dwell = torch.as_tensor(
            recovery_dwell, dtype=torch.float32, device=self.device
        )

        self.fsm_state = torch.full(
            (self.num_envs,),
            int(VQRFsmState.FOUR_STAND),
            dtype=torch.long,
            device=self.device,
        )
        self.support_diagonal = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.state_time = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.just_switched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._recovery_safe_time = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    @property
    def state(self) -> torch.Tensor:
        """Alias for ``fsm_state`` matching the scalar FSM interface."""
        return self.fsm_state

    @state.setter
    def state(self, value: torch.Tensor) -> None:
        value = self._as_env_vector(value, name="state", dtype=torch.long)
        self.fsm_state.copy_(value)

    def _as_env_vector(
        self,
        value: torch.Tensor,
        *,
        name: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert one input to a tensor vector without leaving ``device``."""
        tensor = torch.as_tensor(value, device=self.device).to(dtype=dtype)
        if tensor.ndim == 0:
            tensor = tensor.expand(self.num_envs)
        elif tensor.shape == (self.num_envs, 1):
            tensor = tensor[:, 0]
        elif tensor.shape != (self.num_envs,):
            raise ValueError(
                f"{name} must have shape ({self.num_envs},) or "
                f"({self.num_envs}, 1), got {tuple(tensor.shape)}"
            )
        return tensor

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset selected environments to ``FOUR_STAND``."""
        if env_ids is None:
            index = slice(None)
        elif isinstance(env_ids, slice):
            index = env_ids
        else:
            index = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            if index.ndim == 0:
                index = index.unsqueeze(0)

        self.fsm_state[index] = int(VQRFsmState.FOUR_STAND)
        self.support_diagonal[index] = 0
        self.state_time[index] = 0.0
        self.just_switched[index] = False
        self._recovery_safe_time[index] = 0.0

    def update(
        self,
        yaw_cmd: torch.Tensor,
        positive_pose_ready: torch.Tensor,
        negative_pose_ready: torch.Tensor,
        four_stand_ready: torch.Tensor,
        unsafe: torch.Tensor,
    ) -> torch.Tensor:
        """Advance every FSM by one update and return the live state buffer.

        All predicates are evaluated from the state at the start of this
        update.  In particular, an environment in ``RETURN_TO_4`` can only
        enter ``FOUR_STAND`` here; its yaw command is considered on the next
        update, preventing a direct ``RETURN_TO_4 -> TRANSITION_*`` edge.
        """
        yaw = self._as_env_vector(yaw_cmd, name="yaw_cmd", dtype=torch.float64)
        positive_ready = self._as_env_vector(
            positive_pose_ready, name="positive_pose_ready", dtype=torch.bool
        )
        negative_ready = self._as_env_vector(
            negative_pose_ready, name="negative_pose_ready", dtype=torch.bool
        )
        four_ready = self._as_env_vector(
            four_stand_ready, name="four_stand_ready", dtype=torch.bool
        )
        unsafe_mask = self._as_env_vector(unsafe, name="unsafe", dtype=torch.bool)

        previous = self.fsm_state
        next_state = previous.clone()

        # Safety has priority over all normal transitions.
        next_state = torch.where(
            unsafe_mask,
            torch.full_like(previous, int(VQRFsmState.SAFE_RECOVERY)),
            next_state,
        )
        active = ~unsafe_mask

        four_stand = previous == int(VQRFsmState.FOUR_STAND)
        transition_pos = previous == int(VQRFsmState.TRANSITION_POS)
        yaw_pos = previous == int(VQRFsmState.YAW_POS)
        transition_neg = previous == int(VQRFsmState.TRANSITION_NEG)
        yaw_neg = previous == int(VQRFsmState.YAW_NEG)
        return_to_four = previous == int(VQRFsmState.RETURN_TO_4)
        safe_recovery = previous == int(VQRFsmState.SAFE_RECOVERY)

        enter_pos = active & four_stand & (yaw > self.yaw_enter)
        enter_neg = active & four_stand & (yaw < -self.yaw_enter)
        next_state = torch.where(
            enter_pos,
            torch.full_like(previous, int(VQRFsmState.TRANSITION_POS)),
            next_state,
        )
        next_state = torch.where(
            enter_neg,
            torch.full_like(previous, int(VQRFsmState.TRANSITION_NEG)),
            next_state,
        )

        # TRANSITION may always abort.  YAW exits are intentionally held for a
        # short dwell so a resampled/noisy command cannot immediately undo a
        # completed lift.  The unsafe branch above bypasses this dwell.
        abort_pos = active & (
            (transition_pos & (yaw < self.yaw_exit))
            | (yaw_pos & (self.state_time >= self.yaw_min_dwell) & (yaw < self.yaw_exit))
        )
        abort_neg = active & (
            (transition_neg & (yaw > -self.yaw_exit))
            | (yaw_neg & (self.state_time >= self.yaw_min_dwell) & (yaw > -self.yaw_exit))
        )
        next_state = torch.where(
            abort_pos | abort_neg,
            torch.full_like(previous, int(VQRFsmState.RETURN_TO_4)),
            next_state,
        )

        ready_pos = active & transition_pos & ~abort_pos & positive_ready
        ready_neg = active & transition_neg & ~abort_neg & negative_ready
        next_state = torch.where(
            ready_pos,
            torch.full_like(previous, int(VQRFsmState.YAW_POS)),
            next_state,
        )
        next_state = torch.where(
            ready_neg,
            torch.full_like(previous, int(VQRFsmState.YAW_NEG)),
            next_state,
        )

        # RETURN_TO_4 always lands in FOUR_STAND; it cannot skip this state.
        return_ready = active & return_to_four & four_ready
        next_state = torch.where(
            return_ready,
            torch.full_like(previous, int(VQRFsmState.FOUR_STAND)),
            next_state,
        )

        # SAFE_RECOVERY requires a continuous safe/four-wheel-ready dwell.
        recovery_ready = safe_recovery & ~unsafe_mask & four_ready
        recovery_time = torch.where(
            recovery_ready,
            self._recovery_safe_time + self.dt,
            torch.zeros_like(self._recovery_safe_time),
        )
        recover = recovery_ready & (recovery_time >= self.recovery_dwell)
        next_state = torch.where(
            recover,
            torch.full_like(previous, int(VQRFsmState.FOUR_STAND)),
            next_state,
        )
        self._recovery_safe_time.copy_(recovery_time)
        self._recovery_safe_time.masked_fill_(recover, 0.0)

        switched = next_state != previous

        # Remember the support diagonal throughout TRANSITION/YAW/RETURN.
        diagonal = self.support_diagonal
        diagonal = torch.where(
            enter_pos | ready_pos,
            torch.ones_like(diagonal),
            diagonal,
        )
        diagonal = torch.where(
            enter_neg | ready_neg,
            -torch.ones_like(diagonal),
            diagonal,
        )
        diagonal = torch.where(
            next_state == int(VQRFsmState.FOUR_STAND),
            torch.zeros_like(diagonal),
            diagonal,
        )

        self.fsm_state.copy_(next_state)
        self.support_diagonal.copy_(diagonal)
        self.just_switched.copy_(switched)
        self.state_time.copy_(
            torch.where(
                switched,
                torch.zeros_like(self.state_time),
                self.state_time + self.dt,
            )
        )

        return self.fsm_state


class YawFSMCommand(YawRateCommand):
    """Yaw-rate command coupled to the per-environment yaw FSM.

    ``VQRYawFSM`` is deliberately kept as the single source of truth for the
    transition rules.  The readiness predicates are buffers populated from
    batched Isaac Lab sensor data, without creating one Python callback per
    environment.  ``set_fsm_inputs`` remains available for task-specific
    callers that provide their own predicate implementation.

    The command itself remains ``(num_envs, 1)`` and is therefore compatible
    with existing yaw observations and rewards.
    """

    cfg: "YawFSMCommandCfg"

    def __init__(self, cfg: "YawFSMCommandCfg", env):
        super().__init__(cfg, env)

        self._fsm = YawFSMVectorized(
            self.num_envs,
            device=self.device,
            yaw_enter=cfg.yaw_enter,
            yaw_exit=cfg.yaw_exit,
            dt=env.step_dt,
            yaw_min_dwell=cfg.yaw_min_dwell,
            recovery_dwell=cfg.recovery_dwell,
        )
        self._fsm_state = self._fsm.fsm_state

        # These are public on purpose: task-specific sensor code can update
        # all predicates in one batched operation before the command update.
        self.positive_pose_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.negative_pose_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.four_stand_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.unsafe = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.support_diagonal = self._fsm.support_diagonal
        self.state_time = self._fsm.state_time
        self.just_switched = self._fsm.just_switched
        # This anchor belongs to the command, rather than the FSM helper or a
        # reward-local cache.  The drift reward must consume this exact buffer.
        self.yaw_entry_pos = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)

        # SceneEntityCfg resolution is intentionally lazy.  Command managers
        # can be constructed before Isaac Lab starts the simulation, while the
        # body/sensor views needed by resolve() only exist after play begins.
        self._fsm_sensor_cfg_names = (
            "support_sensor_cfg",
            "support_sensor_cfg_mirror",
            "lifted_asset_cfg",
            "lifted_asset_cfg_mirror",
            "all_wheel_sensor_cfg",
            "torso_sensor_cfg",
        )
        configured = [getattr(cfg, name, None) for name in self._fsm_sensor_cfg_names]
        if any(value is None for value in configured) and any(value is not None for value in configured):
            missing = [name for name, value in zip(self._fsm_sensor_cfg_names, configured) if value is None]
            raise ValueError(
                "YawFSMCommand requires all sensor/body configs when predicate wiring is enabled; "
                f"missing {missing}."
            )
        self._fsm_predicates_enabled = all(value is not None for value in configured)
        self._fsm_scene_entities_resolved = False

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
        self._fsm.reset(env_ids)
        self.yaw_entry_pos[env_ids] = 0.0

        self.positive_pose_ready[env_ids] = False
        self.negative_pose_ready[env_ids] = False
        self.four_stand_ready[env_ids] = False
        self.unsafe[env_ids] = False

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset command metrics and FSM state for the selected environments."""
        extras = super().reset(env_ids)
        self._reset_fsm(slice(None) if env_ids is None else env_ids)
        return extras

    def _step_fsm(self) -> None:
        self._update_fsm_predicates()
        self._fsm.update(
            yaw_cmd=self._command[:, 0],
            positive_pose_ready=self.positive_pose_ready,
            negative_pose_ready=self.negative_pose_ready,
            four_stand_ready=self.four_stand_ready,
            unsafe=self.unsafe,
        )
        entering_yaw = self.just_switched & (
            (self.fsm_state == int(VQRFsmState.YAW_POS))
            | (self.fsm_state == int(VQRFsmState.YAW_NEG))
        )
        current_xy = (
            self.robot.data.root_pos_w[:, :2] - self._env.scene.env_origins[:, :2]
        ).to(dtype=self.yaw_entry_pos.dtype)
        self.yaw_entry_pos.copy_(
            torch.where(entering_yaw.unsqueeze(1), current_xy, self.yaw_entry_pos)
        )

    def _update_fsm_predicates(self) -> None:
        """Refresh FSM inputs from the current batched Isaac Lab sensor state."""
        if not self._fsm_predicates_enabled:
            # Preserve the explicit set_fsm_inputs() path for CPU-only FSM
            # tests and callers that provide task-specific predicates.
            return

        configs = [getattr(self.cfg, name) for name in self._fsm_sensor_cfg_names]
        if not self._fsm_scene_entities_resolved:
            for config in configs:
                resolve = getattr(config, "resolve", None)
                if resolve is None:
                    raise TypeError(
                        "YawFSMCommand sensor/body configs must be SceneEntityCfg instances."
                    )
                resolve(self._env.scene)
            self._fsm_scene_entities_resolved = True

        from .observations import yaw_fsm_predicates

        predicates = yaw_fsm_predicates(
            self._env,
            robot_name=self.cfg.asset_name,
            support_sensor_cfg=configs[0],
            support_sensor_cfg_mirror=configs[1],
            lifted_asset_cfg=configs[2],
            lifted_asset_cfg_mirror=configs[3],
            all_wheel_sensor_cfg=configs[4],
            torso_sensor_cfg=configs[5],
            target_clearance=self.cfg.target_clearance,
            wheel_radius=self.cfg.wheel_radius,
            contact_threshold=self.cfg.contact_threshold,
            clearance_fraction=self.cfg.clearance_fraction,
            pose_angle_limit=self.cfg.pose_angle_limit,
            unsafe_angle_limit=self.cfg.unsafe_angle_limit,
            minimum_base_height=self.cfg.minimum_base_height,
        )
        self.set_fsm_inputs(*predicates)

    def _update_command(self):
        """Keep the base command behavior, then advance the FSM once."""
        super()._update_command()
        self._step_fsm()


@configclass
class YawFSMCommandCfg(YawRateCommandCfg):
    """Configuration for a yaw-rate command driven by ``VQRYawFSM``."""

    class_type: type = YawFSMCommand
    yaw_enter: float = 0.10
    yaw_exit: float = 0.05
    yaw_min_dwell: float = 0.20
    recovery_dwell: float = 0.50
    support_sensor_cfg: SceneEntityCfg | None = None
    support_sensor_cfg_mirror: SceneEntityCfg | None = None
    lifted_asset_cfg: SceneEntityCfg | None = None
    lifted_asset_cfg_mirror: SceneEntityCfg | None = None
    all_wheel_sensor_cfg: SceneEntityCfg | None = None
    torso_sensor_cfg: SceneEntityCfg | None = None
    target_clearance: float = 0.20
    wheel_radius: float = 0.091
    contact_threshold: float = 1.0
    clearance_fraction: float = 0.8
    pose_angle_limit: float = 0.35
    unsafe_angle_limit: float = 0.80
    minimum_base_height: float = 0.35
