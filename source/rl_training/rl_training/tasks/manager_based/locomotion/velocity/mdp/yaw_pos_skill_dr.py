"""Task-local physical and actuator-delay schedule, applied on environment reset."""

from __future__ import annotations

import math

import torch


def _ids(env, env_ids):
    if isinstance(env_ids, slice) or env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device="cpu")
    return torch.as_tensor(env_ids, dtype=torch.long, device="cpu")


def reset_physical(env, env_ids):
    """Resample at a new robustness stage from captured nominal properties."""
    ids = _ids(env, env_ids)
    if ids.numel() == 0:
        return
    robot = env.scene["robot"]
    view = robot.root_physx_view
    if not hasattr(env, "_yaw_pos_skill_nominal_physics"):
        env._yaw_pos_skill_nominal_physics = {
            "masses": view.get_masses().clone(),
            "inertias": view.get_inertias().clone(),
            "coms": view.get_coms().clone(),
            "materials": view.get_material_properties().clone(),
        }
        env._yaw_pos_skill_applied_physics_stage = torch.full((env.num_envs,), -1, dtype=torch.long)
    stage = int(getattr(getattr(env, "_yaw_pos_skill_state", None), "robustness_stage", 0))
    scale = getattr(getattr(env, "_yaw_pos_skill_state", None), "dr_scale", 0.0)
    if scale == 0:
        ids = ids[env._yaw_pos_skill_applied_physics_stage[ids] != stage]
        if ids.numel() == 0:
            return
    # In robustness stages, resample each episode from the original nominal
    # body properties, while reusing the finite stage-specific material pool.
    nominal = env._yaw_pos_skill_nominal_physics
    masses = view.get_masses().clone()
    base = nominal["masses"][ids].clone()
    body_names = robot.body_names
    base_index = body_names.index("TORSO")
    factors = 1.0 + (2.0 * torch.rand_like(base) - 1.0) * (0.15 * scale)
    masses[ids] = base * factors
    masses[ids, base_index] += (-1.0 + 4.0 * torch.rand(len(ids), device=base.device)) * scale
    if torch.any(masses[ids] <= 0):
        raise RuntimeError("Skill DR would create nonpositive body mass.")
    view.set_masses(masses, ids)

    inertias = view.get_inertias().clone()
    inertias[ids] = nominal["inertias"][ids]
    diag = (0, 4, 8)
    for index in diag:
        original = nominal["inertias"][ids, :, index]
        inertias[ids, :, index] = original * (1 + (2 * torch.rand_like(original) - 1) * 0.15 * scale)
    view.set_inertias(inertias, ids)

    coms = view.get_coms().clone()
    coms[ids] = nominal["coms"][ids]
    offset = torch.empty((len(ids), 1, 3), device=coms.device).uniform_(-1, 1)
    coms[ids, :, :3] += offset * coms.new_tensor((0.03, 0.03, 0.02)) * scale
    view.set_coms(coms, ids)

    materials = view.get_material_properties().clone()
    if scale == 0:
        materials[ids, :, 0:2] = 1.0
        materials[ids, :, 2] = 0.0
    else:
        # PhysX has a finite unique-material budget. Match the existing POS
        # startup randomizer's 1024 buckets, once per robustness level.
        if not hasattr(env, "_yaw_pos_skill_material_buckets"):
            env._yaw_pos_skill_material_buckets = {}
        buckets = env._yaw_pos_skill_material_buckets.get(stage)
        if buckets is None:
            low, high = 1.0 - 0.65 * scale, 1.0 + 0.50 * scale
            buckets = torch.empty((1024, 3), device=materials.device)
            buckets[:, :2] = low + (high - low) * torch.rand((1024, 2), device=materials.device)
            buckets[:, 2] = 0.7 * scale * torch.rand(1024, device=materials.device)
            env._yaw_pos_skill_material_buckets[stage] = buckets
        bucket_ids = torch.randint(0, len(buckets), materials[ids].shape[:-1], device=materials.device)
        materials[ids] = buckets[bucket_ids]
    view.set_material_properties(materials, ids)
    env._yaw_pos_skill_applied_physics_stage[ids] = stage


def reset_delay(env, env_ids):
    """Override sampled DelayedPDActuator lags before the next physics action."""
    ids = _ids(env, env_ids).to(env.device)
    scale = getattr(getattr(env, "_yaw_pos_skill_state", None), "dr_scale", 0.0)
    maximum = 2 + math.floor(6 * scale)
    for actuator in env.scene["robot"].actuators.values():
        for buffer in (actuator.positions_delay_buffer, actuator.velocities_delay_buffer,
                       actuator.efforts_delay_buffer):
            if buffer.max_time_lag > 8 or buffer._history_length < 8:
                raise RuntimeError("Skill delay buffer must support eight steps.")
        lags = torch.randint(2, maximum + 1, (len(ids),), device=env.device, dtype=torch.int)
        actuator.positions_delay_buffer.set_time_lag(lags, ids)
        actuator.velocities_delay_buffer.set_time_lag(lags, ids)
        actuator.efforts_delay_buffer.set_time_lag(lags, ids)
