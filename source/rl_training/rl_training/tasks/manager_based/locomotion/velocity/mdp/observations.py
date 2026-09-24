# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def fsm_state_one_hot(
    env: ManagerBasedEnv,
    command_name: str = "yaw_rate_cmd",
    num_states: int = 7,
) -> torch.Tensor:
    """Return the yaw FSM state as a one-hot observation vector.

    The command term owns the state so the observation always reflects the
    same state machine used by the command/reward logic.  The output shape is
    ``(num_envs, num_states)``.
    """

    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term, "fsm_state"):
        raise TypeError(
            f"Command '{command_name}' does not expose 'fsm_state'; "
            "use YawFSMCommand for the FSM task."
        )

    state = command_term.fsm_state
    if state.ndim != 1 or state.shape[0] != env.num_envs:
        raise ValueError(
            f"FSM state must have shape ({env.num_envs},), got {tuple(state.shape)}"
        )
    if torch.any((state < 0) | (state >= num_states)):
        raise ValueError(f"FSM state values must be in [0, {num_states}), got {state}")

    return F.one_hot(state.to(dtype=torch.long), num_classes=num_states).to(dtype=torch.float32)


def _fsm_vector_observation(command_term, name: str, num_envs: int) -> torch.Tensor:
    """Read one per-environment FSM predicate with a stable shape."""
    if not hasattr(command_term, name):
        raise TypeError(
            f"Command term does not expose '{name}'; "
            "use YawFSMCommand for the FSM task."
        )

    value = torch.as_tensor(getattr(command_term, name))
    if value.ndim != 1 or value.shape[0] != num_envs:
        raise ValueError(
            f"FSM predicate '{name}' must have shape ({num_envs},), "
            f"got {tuple(value.shape)}"
        )
    return value


def fsm_ready_flags(
    env: ManagerBasedEnv,
    command_name: str = "yaw_rate_cmd",
) -> torch.Tensor:
    """Return ``[positive, negative, four-stand, unsafe]`` FSM predicates.

    The flags are critic-only in the FSM task.  They are kept as explicit
    scalar channels instead of being inferred from the one-hot state because
    readiness is useful before a transition is accepted and ``unsafe`` can be
    true while the FSM is still exposing its previous state for one step.
    """
    command_term = env.command_manager.get_term(command_name)
    flags = (
        _fsm_vector_observation(command_term, "positive_pose_ready", env.num_envs),
        _fsm_vector_observation(command_term, "negative_pose_ready", env.num_envs),
        _fsm_vector_observation(command_term, "four_stand_ready", env.num_envs),
        _fsm_vector_observation(command_term, "unsafe", env.num_envs),
    )
    return torch.stack(flags, dim=-1).to(dtype=torch.float32)


def fsm_state_time(
    env: ManagerBasedEnv,
    command_name: str = "yaw_rate_cmd",
) -> torch.Tensor:
    """Return elapsed time in the current FSM state as one critic channel."""
    command_term = env.command_manager.get_term(command_name)
    state_time = _fsm_vector_observation(command_term, "state_time", env.num_envs)
    return state_time.to(dtype=torch.float32).unsqueeze(-1)


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def wheel_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return one binary contact flag per selected wheel body."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

    return (torch.linalg.vector_norm(force, dim=-1) > threshold).float()


def wheel_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float,
) -> torch.Tensor:
    """Return wheel-bottom clearance relative to each environment origin."""

    robot: Articulation = env.scene[asset_cfg.name]
    wheel_height = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    ground_height = env.scene.env_origins[:, 2].unsqueeze(-1)

    return wheel_height - ground_height - wheel_radius


def wheel_normal_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_scale: float = 100.0,
) -> torch.Tensor:
    """Return normalized upward contact force for each selected wheel."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    fz = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]

    return torch.clamp(fz, min=0.0) / force_scale


def yaw_fsm_predicates(
    env: ManagerBasedEnv,
    robot_name: str,
    support_sensor_cfg: SceneEntityCfg,
    support_sensor_cfg_mirror: SceneEntityCfg,
    lifted_asset_cfg: SceneEntityCfg,
    lifted_asset_cfg_mirror: SceneEntityCfg,
    all_wheel_sensor_cfg: SceneEntityCfg,
    torso_sensor_cfg: SceneEntityCfg,
    target_clearance: float,
    wheel_radius: float,
    contact_threshold: float = 1.0,
    clearance_fraction: float = 0.8,
    pose_angle_limit: float = 0.35,
    unsafe_angle_limit: float = 0.80,
    minimum_base_height: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the four batched readiness predicates consumed by ``YawFSMCommand``.

    The body selections are explicit and ordered in the command configuration:
    ``support_sensor_cfg`` is the POS diagonal (FL, HR), while its mirror is
    the NEG diagonal (FR, HL).  This keeps contact and clearance predicates on
    the same canonical mapping used by the FSM rewards.
    """
    if target_clearance < 0.0:
        raise ValueError("target_clearance must be non-negative.")
    if contact_threshold < 0.0:
        raise ValueError("contact_threshold must be non-negative.")
    if wheel_radius < 0.0:
        raise ValueError("wheel_radius must be non-negative.")
    if not 0.0 <= clearance_fraction <= 1.0:
        raise ValueError("clearance_fraction must be in [0, 1].")
    if pose_angle_limit < 0.0 or unsafe_angle_limit < 0.0:
        raise ValueError("Pose angle limits must be non-negative.")

    positive_support_contact = wheel_contact(
        env, support_sensor_cfg, threshold=contact_threshold
    ).bool()
    negative_support_contact = wheel_contact(
        env, support_sensor_cfg_mirror, threshold=contact_threshold
    ).bool()
    four_wheel_contact = wheel_contact(
        env, all_wheel_sensor_cfg, threshold=contact_threshold
    ).bool()
    torso_contact = wheel_contact(
        env, torso_sensor_cfg, threshold=contact_threshold
    ).bool()

    lifted_clearance = wheel_clearance(
        env, lifted_asset_cfg, wheel_radius=wheel_radius
    )
    lifted_clearance_mirror = wheel_clearance(
        env, lifted_asset_cfg_mirror, wheel_radius=wheel_radius
    )

    if positive_support_contact.shape[-1] != 2 or negative_support_contact.shape[-1] != 2:
        raise ValueError("Yaw FSM support contact selections must contain exactly two bodies.")
    if lifted_clearance.shape[-1] != 2 or lifted_clearance_mirror.shape[-1] != 2:
        raise ValueError("Yaw FSM lifted-body selections must contain exactly two bodies.")
    if four_wheel_contact.shape[-1] != 4:
        raise ValueError("Yaw FSM four-wheel contact selection must contain exactly four bodies.")
    if torso_contact.shape[-1] != 1:
        raise ValueError("Yaw FSM torso contact selection must contain exactly one body.")

    robot = env.scene[robot_name]
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    pose_is_safe = (torch.abs(roll) < pose_angle_limit) & (torch.abs(pitch) < pose_angle_limit)

    clearance_threshold = clearance_fraction * target_clearance
    positive_pose_ready = (
        positive_support_contact.all(dim=-1)
        & (lifted_clearance >= clearance_threshold).all(dim=-1)
        & pose_is_safe
    )
    negative_pose_ready = (
        negative_support_contact.all(dim=-1)
        & (lifted_clearance_mirror >= clearance_threshold).all(dim=-1)
        & pose_is_safe
    )
    four_stand_ready = four_wheel_contact.all(dim=-1) & pose_is_safe

    unsafe = yaw_fsm_unsafe(
        env,
        robot_name=robot_name,
        torso_sensor_cfg=torso_sensor_cfg,
        minimum_base_height=minimum_base_height,
        unsafe_angle_limit=unsafe_angle_limit,
        contact_threshold=contact_threshold,
    )

    return positive_pose_ready, negative_pose_ready, four_stand_ready, unsafe


def yaw_fsm_unsafe(
    env: ManagerBasedEnv,
    robot_name: str,
    torso_sensor_cfg: SceneEntityCfg,
    minimum_base_height: float,
    unsafe_angle_limit: float,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Return the single current-state unsafe predicate used by FSM and done.

    Keeping this separate from :func:`yaw_fsm_predicates` is deliberate: a
    termination term must inspect live sensor/pose data, not the command
    buffer from the preceding command update.  Both paths nevertheless use
    exactly the same thresholds and contact interpretation.
    """
    if minimum_base_height < 0.0:
        raise ValueError("minimum_base_height must be non-negative.")
    if unsafe_angle_limit < 0.0:
        raise ValueError("unsafe_angle_limit must be non-negative.")
    if contact_threshold < 0.0:
        raise ValueError("contact_threshold must be non-negative.")

    torso_contact = wheel_contact(
        env, torso_sensor_cfg, threshold=contact_threshold
    ).bool()
    if torso_contact.ndim != 2 or torso_contact.shape[-1] != 1:
        raise ValueError("Yaw FSM torso contact selection must contain exactly one body.")
    robot: Articulation = env.scene[robot_name]
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    base_height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (
        torso_contact[:, 0]
        | (base_height < minimum_base_height)
        | (torch.abs(roll) > unsafe_angle_limit)
        | (torch.abs(pitch) > unsafe_angle_limit)
    )


def base_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return root height relative to the local environment ground origin."""
    robot: Articulation = env.scene[asset_cfg.name]
    return (robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)


def com_support_coordinate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return whole-body CoM coordinates in the two-wheel support frame.

    The selected bodies must be ordered front wheel then hind wheel: FL->HR
    or FR->HL. The output is ``[along_support, lateral_to_support]`` in meters,
    measured from the support-segment midpoint. Positive ``along_support``
    points from the first wheel to the second; positive lateral is the left
    normal of that direction in the world XY plane.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    support_xy = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]
    if support_xy.shape[1] != 2:
        raise ValueError("com_support_coordinate requires exactly two ordered support bodies.")

    support_vector = support_xy[:, 1] - support_xy[:, 0]
    support_length = torch.linalg.vector_norm(
        support_vector, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    tangent = support_vector / support_length
    normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=-1)
    support_midpoint = support_xy.mean(dim=1)

    masses = robot.root_physx_view.get_masses().to(robot.device)
    total_mass = masses.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    com_xy = (
        robot.data.body_com_pos_w[..., :2] * masses.unsqueeze(-1)
    ).sum(dim=1) / total_mass
    relative_com = com_xy - support_midpoint

    along_support = torch.sum(relative_com * tangent, dim=-1)
    lateral_to_support = torch.sum(relative_com * normal, dim=-1)
    return torch.stack((along_support, lateral_to_support), dim=-1)


def _fsm_select_support_geometry(
    env: ManagerBasedRLEnv,
    command_name: str,
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> torch.Tensor:
    """Select the active support diagonal, with zero geometry in four-wheel stand."""
    command = env.command_manager.get_term(command_name)
    diagonal = torch.as_tensor(command.support_diagonal, device=positive.device)
    if diagonal.shape != positive.shape[:1] or negative.shape != positive.shape:
        raise ValueError("FSM support geometry requires matching (num_envs, 2) inputs and diagonal state.")
    return torch.where(
        (diagonal == 1).unsqueeze(-1),
        positive,
        torch.where((diagonal == -1).unsqueeze(-1), negative, torch.zeros_like(positive)),
    )


def fsm_com_support_coordinate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    asset_cfg_mirror: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """CoM in FL->HR for POS or FR->HL for NEG; zero when no diagonal is active."""
    positive = com_support_coordinate(env, asset_cfg)
    negative = com_support_coordinate(env, asset_cfg_mirror)
    return _fsm_select_support_geometry(env, command_name, positive, negative)


def rolling_lateral_contact_velocity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    body_asset_cfg: SceneEntityCfg,
    joint_asset_cfg: SceneEntityCfg,
    wheel_radius: float,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return contacted wheels' local rolling/lateral velocities.

    For each selected wheel, the two values are contact-point
    ``[rolling, lateral]`` velocity.  Rolling uses ``v_local_x + radius * qdot``
    for the VQR wheel joint's local negative-Y axis; lateral uses ``v_local_y``.
    Values are zeroed while that wheel is not in contact.  The flattened output
    preserves the configured wheel/body order.
    """
    robot: Articulation = env.scene[body_asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    selection_lengths = {
        len(body_asset_cfg.body_ids),
        len(joint_asset_cfg.joint_ids),
        len(sensor_cfg.body_ids),
    }
    if len(selection_lengths) != 1:
        raise ValueError("Wheel body, joint, and contact-sensor selections must have equal length.")

    velocity_w = robot.data.body_lin_vel_w[:, body_asset_cfg.body_ids]
    orientation_w = robot.data.body_quat_w[:, body_asset_cfg.body_ids]
    contact_force = torch.linalg.vector_norm(
        sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1
    )
    in_contact = contact_force > threshold
    velocity_local = quat_apply_inverse(
        orientation_w.reshape(-1, 4), velocity_w.reshape(-1, 3)
    ).reshape(velocity_w.shape)
    rolling_contact_velocity = (
        velocity_local[..., 0]
        + wheel_radius * robot.data.joint_vel[:, joint_asset_cfg.joint_ids]
    )
    rolling_lateral_velocity = torch.stack(
        (rolling_contact_velocity, velocity_local[..., 1]), dim=-1
    )
    return (rolling_lateral_velocity * in_contact.unsqueeze(-1)).reshape(env.num_envs, -1)


def support_wheel_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    wheel_axle_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    body_lateral_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Alignment of support-wheel axle with robot lateral axis.

    Output:
        (N, 2)

    1.0 -> perfectly aligned
    0.0 -> perpendicular
    """

    robot: Articulation = env.scene[asset_cfg.name]

    wheel_quat = robot.data.body_quat_w[:, asset_cfg.body_ids, :]

    # Root/body orientation
    body_quat = robot.data.root_quat_w

    # ---------------------------------------------------
    # Wheel axle -> world
    # ---------------------------------------------------
    axle_local = torch.tensor(
        wheel_axle_axis,
        device=env.device,
        dtype=wheel_quat.dtype,
    ).view(1, 1, 3).expand(wheel_quat.shape[0], wheel_quat.shape[1], 3)

    axle_w = quat_apply(
        wheel_quat.reshape(-1, 4),
        axle_local.reshape(-1, 3),
    ).reshape(wheel_quat.shape[0], wheel_quat.shape[1], 3)

    # ---------------------------------------------------
    # Body lateral axis -> world
    # ---------------------------------------------------
    lateral_local = torch.tensor(
        body_lateral_axis,
        device=env.device,
        dtype=body_quat.dtype,
    ).view(1, 3).expand(body_quat.shape[0], 3)

    body_lateral_w = quat_apply(
        body_quat,
        lateral_local,
    )

    # Normalize
    axle_w = axle_w / (torch.linalg.vector_norm(axle_w, dim=-1, keepdim=True) + 1e-6)
    body_lateral_w = body_lateral_w / (
        torch.linalg.vector_norm(body_lateral_w, dim=-1, keepdim=True) + 1e-6
    )

    # (N, 2)
    alignment = torch.abs(torch.sum(axle_w * body_lateral_w.unsqueeze(1), dim=-1))

    return alignment


def fsm_support_wheel_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    asset_cfg_mirror: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Alignment of the active POS or NEG support pair; zero in four-wheel stand."""
    positive = support_wheel_alignment(env, asset_cfg)
    negative = support_wheel_alignment(env, asset_cfg_mirror)
    return _fsm_select_support_geometry(env, command_name, positive, negative)
