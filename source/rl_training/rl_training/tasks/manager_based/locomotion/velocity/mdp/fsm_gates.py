"""Shared reward gates for the yaw FSM.

The FSM command owns the state buffers.  This module derives all masks from
those buffers once per environment step so reward terms cannot accidentally
observe different gate values during the same step.
"""

from __future__ import annotations

import torch

from .fsm import VQRFsmState

__all__ = ["fsm_gates"]


_CACHE_STEP_ATTR = "_fsm_gates_cache_step"
_CACHE_ATTR = "_fsm_gates_cache"


def _step_key(env) -> int:
    """Return a stable scalar cache key for Isaac Lab's step counter."""
    step = getattr(env, "common_step_counter", None)
    if step is None:
        raise AttributeError("FSM gates require env.common_step_counter for cache invalidation")

    if torch.is_tensor(step):
        if step.numel() != 1:
            raise ValueError(
                "env.common_step_counter must be scalar, "
                f"got shape {tuple(step.shape)}"
            )
        return int(step.detach().item())
    return int(step)


def _required_tensor(command, name: str, *, device: torch.device) -> torch.Tensor:
    """Read and normalize one required FSM buffer."""
    if not hasattr(command, name):
        raise TypeError(
            f"Command term does not expose '{name}'; "
            "use YawFSMCommand with fsm_gates()."
        )

    value = torch.as_tensor(getattr(command, name), device=device)
    if value.ndim != 1:
        raise ValueError(f"FSM buffer '{name}' must have shape (N,), got {tuple(value.shape)}")
    return value


def fsm_gates(env, command_name: str) -> dict[str, torch.Tensor]:
    """Return all shared hard, soft, and diagonal FSM gates.

    The returned dictionary is cached per ``(env.common_step_counter,
    command_name)``.  All state-derived masks are boolean; ``f_*`` entries
    are float masks suitable for reward multiplication.

    Keys:
        ``b_four``, ``b_trans``, ``b_yaw``, ``b_return``, ``b_safe``:
            Hard state masks.
        ``f_four``, ``f_trans``, ``f_yaw``, ``f_return``, ``f_safe``:
            Float versions of the hard masks.
        ``f_geom``:
            Float mask for geometry terms, which are defined in transition,
            yaw, and return states.
        ``f_stability``:
            Soft four-stand stability gate.  It fades out over the first
            second of a transition and fades in over the first second of a
            return, while remaining active in FOUR_STAND. SAFE_RECOVERY is
            deliberately excluded from every positive FSM reward gate.
        ``diag_pos``, ``diag_neg``:
            Support-diagonal selectors from ``support_diagonal``.
        ``fsm_state``, ``support_diagonal``, ``state_time``, ``tau``,
        ``just_switched``:
            The live command buffers used by stateful reward terms.
    """
    step = _step_key(env)
    cache_step = getattr(env, _CACHE_STEP_ATTR, None)
    cache = getattr(env, _CACHE_ATTR, None)

    if cache_step != step or cache is None:
        cache_step = step
        cache = {}
        setattr(env, _CACHE_STEP_ATTR, cache_step)
        setattr(env, _CACHE_ATTR, cache)

    cached = cache.get(command_name)
    if cached is not None:
        return cached

    command = env.command_manager.get_term(command_name)
    device = torch.device(getattr(env, "device", "cpu"))

    state = _required_tensor(command, "fsm_state", device=device).to(dtype=torch.long)
    support_diagonal = _required_tensor(command, "support_diagonal", device=device).to(dtype=torch.long)
    state_time = _required_tensor(command, "state_time", device=device).to(dtype=torch.float32)
    just_switched = _required_tensor(command, "just_switched", device=device).to(dtype=torch.bool)

    if state.shape != support_diagonal.shape or state.shape != state_time.shape:
        raise ValueError(
            "FSM buffers must share shape (N): "
            f"fsm_state={tuple(state.shape)}, "
            f"support_diagonal={tuple(support_diagonal.shape)}, "
            f"state_time={tuple(state_time.shape)}"
        )
    if just_switched.shape != state.shape:
        raise ValueError(
            "FSM buffer 'just_switched' must share shape with fsm_state: "
            f"{tuple(just_switched.shape)} vs {tuple(state.shape)}"
        )
    expected_num_envs = getattr(env, "num_envs", state.shape[0])
    if state.shape != (int(expected_num_envs),):
        raise ValueError(
            "FSM buffers must match env.num_envs: "
            f"expected ({int(expected_num_envs)},), got {tuple(state.shape)}"
        )

    b_four = state == int(VQRFsmState.FOUR_STAND)
    b_trans = (state == int(VQRFsmState.TRANSITION_POS)) | (
        state == int(VQRFsmState.TRANSITION_NEG)
    )
    b_yaw = (state == int(VQRFsmState.YAW_POS)) | (state == int(VQRFsmState.YAW_NEG))
    b_return = state == int(VQRFsmState.RETURN_TO_4)
    b_safe = state == int(VQRFsmState.SAFE_RECOVERY)

    f_four = b_four.to(dtype=torch.float32)
    f_trans = b_trans.to(dtype=torch.float32)
    f_yaw = b_yaw.to(dtype=torch.float32)
    f_return = b_return.to(dtype=torch.float32)
    f_safe = b_safe.to(dtype=torch.float32)

    tau = torch.clamp(state_time, min=0.0, max=1.0)
    f_stability = f_four + f_trans * (1.0 - tau) + f_return * tau

    diag_pos = support_diagonal == 1
    diag_neg = support_diagonal == -1
    f_geom = (b_trans | b_yaw | b_return).to(dtype=torch.float32)

    gates = {
        "b_four": b_four,
        "b_trans": b_trans,
        "b_yaw": b_yaw,
        "b_return": b_return,
        "b_safe": b_safe,
        "f_four": f_four,
        "f_trans": f_trans,
        "f_yaw": f_yaw,
        "f_return": f_return,
        "f_safe": f_safe,
        "f_geom": f_geom,
        "f_stability": f_stability,
        "diag_pos": diag_pos,
        "diag_neg": diag_neg,
        "fsm_state": state,
        "support_diagonal": support_diagonal,
        "state_time": state_time,
        "tau": tau,
        "just_switched": just_switched,
    }
    cache[command_name] = gates
    return gates
