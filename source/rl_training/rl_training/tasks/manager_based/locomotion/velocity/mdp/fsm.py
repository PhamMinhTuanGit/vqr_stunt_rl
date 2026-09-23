import torch

from enum import IntEnum


class VQRFsmState(IntEnum):
    FOUR_STAND = 0
    TRANSITION_POS = 1
    YAW_POS = 2
    TRANSITION_NEG = 3
    YAW_NEG = 4
    RETURN_TO_4 = 5
    SAFE_RECOVERY = 6


class VQRYawFSM:
    def __init__(
        self,
        yaw_enter=0.10,
        yaw_exit=0.05,
    ):
        self.yaw_enter = yaw_enter
        self.yaw_exit = yaw_exit
        self.state = VQRFsmState.FOUR_STAND

    def update(
        self,
        yaw_cmd: float,
        positive_pose_ready: bool,
        negative_pose_ready: bool,
        four_stand_ready: bool,
        unsafe: bool,
    ):
        if unsafe:
            self.state = VQRFsmState.SAFE_RECOVERY
            return self.state

        if self.state == VQRFsmState.FOUR_STAND:
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
            if yaw_cmd < self.yaw_exit:
                self.state = VQRFsmState.RETURN_TO_4

        elif self.state == VQRFsmState.TRANSITION_NEG:
            if yaw_cmd > -self.yaw_exit:
                self.state = VQRFsmState.RETURN_TO_4

            elif negative_pose_ready:
                self.state = VQRFsmState.YAW_NEG

        elif self.state == VQRFsmState.YAW_NEG:
            if yaw_cmd > -self.yaw_exit:
                self.state = VQRFsmState.RETURN_TO_4

        elif self.state == VQRFsmState.RETURN_TO_4:
            if four_stand_ready:
                if yaw_cmd > self.yaw_enter:
                    self.state = VQRFsmState.TRANSITION_POS

                elif yaw_cmd < -self.yaw_enter:
                    self.state = VQRFsmState.TRANSITION_NEG

                else:
                    self.state = VQRFsmState.FOUR_STAND

        return self.state


def compute_fsm_reward_mask(
    env,
    command_name: str,
    active_states: "tuple[VQRFsmState, ...]",
) -> torch.Tensor:
    """Return a per-environment reward mask that is 1.0 in ``active_states``.

    The mask reads the batched FSM state exposed by ``YawFSMCommand`` and is
    cached per step so that multiple reward terms sharing one gate evaluate
    the FSM only once per environment step.

    Args:
        env: The manager-based environment.
        command_name: Name of the ``YawFSMCommand`` term holding the FSM.
        active_states: FSM states in which the mask is 1.0; it is 0.0
            everywhere else.

    Returns:
        A float tensor of shape ``(num_envs,)`` in ``{0.0, 1.0}``.
    """
    if not active_states:
        raise ValueError("active_states must be non-empty.")

    command = env.command_manager.get_term(command_name)
    states = command.fsm_state  # (num_envs,) int64 tensor

    key = tuple(int(state) for state in sorted(active_states))
    cache_token = (command_name, key)

    # common_step_counter is a scalar and advances once per env step, so all
    # reward terms sharing one gate reuse the same mask within a step.
    step = env.common_step_counter
    if getattr(env, "_fsm_mask_step", None) == step:
        cached = env._fsm_mask_cache.get(cache_token)
        if cached is not None:
            return cached

    active = torch.zeros_like(states, dtype=torch.bool)
    for state in key:
        active |= states == state
    mask = active.to(torch.float32).to(env.device)

    if getattr(env, "_fsm_mask_step", None) == step:
        env._fsm_mask_cache[cache_token] = mask
    else:
        env._fsm_mask_step = step
        env._fsm_mask_cache = {cache_token: mask}

    return mask
