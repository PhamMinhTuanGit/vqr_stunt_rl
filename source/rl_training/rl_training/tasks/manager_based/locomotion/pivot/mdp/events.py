# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_two_wheel_diagonal(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    nominal_joint_positions: list[float],
    leg_joint_count: int,
    root_height: float,
    nominal_roll: float,
    nominal_pitch: float,
    joint_position_noise: tuple[float, float],
    leg_velocity_noise: tuple[float, float],
    wheel_velocity_noise: tuple[float, float],
    roll_noise: tuple[float, float],
    pitch_noise: tuple[float, float],
    yaw_range: tuple[float, float],
    angular_velocity_noise: tuple[float, float],
    root_xy_noise: tuple[float, float] = (-0.01, 0.01),
):
    """Reset directly into the fixed FL-HR two-wheel support configuration."""
    asset: Articulation = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    num_joints = len(nominal_joint_positions)

    if isinstance(asset_cfg.joint_ids, slice):
        joint_ids = slice(None)
        selected_joint_count = asset.num_joints
        indexing_env_ids = env_ids
    else:
        joint_ids = asset_cfg.joint_ids
        selected_joint_count = len(joint_ids)
        indexing_env_ids = env_ids[:, None]
    if selected_joint_count != num_joints:
        raise ValueError(
            f"Two-wheel reset expected {num_joints} joints, but SceneEntityCfg resolved {selected_joint_count}."
        )

    joint_pos = torch.tensor(nominal_joint_positions, device=asset.device).repeat(num_envs, 1)
    joint_pos[:, :leg_joint_count] += math_utils.sample_uniform(
        *joint_position_noise, (num_envs, leg_joint_count), asset.device
    )
    joint_limits = asset.data.soft_joint_pos_limits[indexing_env_ids, joint_ids]
    joint_pos.clamp_(joint_limits[..., 0], joint_limits[..., 1])

    joint_vel = torch.zeros((num_envs, num_joints), device=asset.device)
    joint_vel[:, :leg_joint_count] = math_utils.sample_uniform(
        *leg_velocity_noise, (num_envs, leg_joint_count), asset.device
    )
    joint_vel[:, leg_joint_count:] = math_utils.sample_uniform(
        *wheel_velocity_noise, (num_envs, num_joints - leg_joint_count), asset.device
    )

    root_pos = env.scene.env_origins[env_ids].clone()
    root_pos[:, :2] += math_utils.sample_uniform(*root_xy_noise, (num_envs, 2), asset.device)
    root_pos[:, 2] += root_height
    roll = nominal_roll + math_utils.sample_uniform(*roll_noise, (num_envs,), asset.device)
    pitch = nominal_pitch + math_utils.sample_uniform(*pitch_noise, (num_envs,), asset.device)
    yaw = math_utils.sample_uniform(*yaw_range, (num_envs,), asset.device)
    root_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

    root_vel = torch.zeros((num_envs, 6), device=asset.device)
    root_vel[:, 3:] = math_utils.sample_uniform(*angular_velocity_noise, (num_envs, 3), asset.device)

    asset.write_root_pose_to_sim(torch.cat((root_pos, root_quat), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)


def reset_four_wheel_standing(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    nominal_joint_positions: list[float],
    leg_joint_count: int,
    root_height: float,
    joint_position_noise: tuple[float, float],
    leg_velocity_noise: tuple[float, float],
    wheel_velocity_noise: tuple[float, float],
    roll_noise: tuple[float, float],
    pitch_noise: tuple[float, float],
    yaw_range: tuple[float, float],
    angular_velocity_noise: tuple[float, float],
    root_xy_noise: tuple[float, float] = (-0.01, 0.01),
):
    """Reset M3 around the audited four-wheel standing pose."""
    reset_two_wheel_diagonal(
        env=env,
        env_ids=env_ids,
        asset_cfg=asset_cfg,
        nominal_joint_positions=nominal_joint_positions,
        leg_joint_count=leg_joint_count,
        root_height=root_height,
        nominal_roll=0.0,
        nominal_pitch=0.0,
        joint_position_noise=joint_position_noise,
        leg_velocity_noise=leg_velocity_noise,
        wheel_velocity_noise=wheel_velocity_noise,
        roll_noise=roll_noise,
        pitch_noise=pitch_noise,
        yaw_range=yaw_range,
        angular_velocity_noise=angular_velocity_noise,
        root_xy_noise=root_xy_noise,
    )
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_pivot_reset_root_xy"):
        env._pivot_reset_root_xy = torch.zeros((env.num_envs, 2), device=asset.device)
    env._pivot_reset_root_xy[env_ids] = asset.data.root_pos_w[env_ids, :2]


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract and randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        # Assign the randomized values back to the inertia tensor
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_com_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the center of mass (COM) positions for the rigid bodies.

    This function allows randomizing the COM positions of the bodies in the physics simulation. The positions can be
    randomized by adding, scaling, or setting random values sampled from the specified distribution.

    .. tip::
        This function is intended for initialization or offline adjustments, as it modifies physics properties directly.

    Args:
        env (ManagerBasedEnv): The simulation environment.
        env_ids (torch.Tensor | None): Specific environment indices to apply randomization, or None for all environments.
        asset_cfg (SceneEntityCfg): The configuration for the target asset whose COM will be randomized.
        com_distribution_params (tuple[float, float]): Parameters of the distribution (e.g., min and max for uniform).
        operation (Literal["add", "scale", "abs"]): The operation to apply for randomization.
        distribution (Literal["uniform", "log_uniform", "gaussian"]): The distribution to sample random values from.
    """
    # Extract the asset (Articulation or RigidObject)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Resolve environment indices
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Get the current COM offsets (num_assets, num_bodies, 3)
    com_offsets = asset.root_physx_view.get_coms()

    for dim_idx in range(3):  # Randomize x, y, z independently
        randomized_offset = _randomize_prop_by_op(
            com_offsets[:, :, dim_idx],
            com_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

    # Set the randomized COM offsets into the simulation
    asset.root_physx_view.set_coms(com_offsets, env_ids)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None) # type: ignore
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def bad_orientation_2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot") # type: ignore
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return (asset.data.projected_gravity_b[:, 2] > 0) | (asset.data.projected_gravity_b[:, :2].abs() > 0.7).any(-1)
