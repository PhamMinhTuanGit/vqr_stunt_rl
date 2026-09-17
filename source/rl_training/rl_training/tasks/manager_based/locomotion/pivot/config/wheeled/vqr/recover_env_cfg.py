# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""pi_recover environment: righting from the fallen distribution (section 5).

Kept separate from the four-mode task because fallen states are a clearly
disjoint state distribution; reuses the robot articulation, the mdp library,
and the PivotStateCache machinery through PivotEnv.
"""

from __future__ import annotations

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.pivot.mdp as mdp

from .pivot_env_cfg import (
    BASE_LINK_NAME,
    LEG_JOINT_NAMES,
    PivotEventCfg,
    PivotObservationsCfg,
    PivotSceneCfg,
    PivotTerminationsCfg,
    WHEEL_JOINT_NAMES,
)
from .physical_params import VQR_PHYSICS as PHYS


def recover_upright_bonus(env) -> torch.Tensor:
    """Reward rising toward upright: grows as projected gravity hits -z."""
    g = env.scene["robot"].data.projected_gravity_b
    return torch.clamp(-g[:, 2], min=0.0).square()


def recover_contact_restored(env) -> torch.Tensor:
    """Reward all four wheels back on the ground."""
    from ...mdp.observations import _pivot_cache

    cache = _pivot_cache(env)
    return (cache.wheel_contact > 0.5).float().mean(dim=-1)


def recover_joint_smoothness(env) -> torch.Tensor:
    """Penalise wild joint accelerations while righting."""
    robot = env.scene["robot"]
    idx = [robot.data.joint_names.index(n) for n in LEG_JOINT_NAMES] or slice(None)
    return robot.data.joint_acc[:, idx].square().mean(dim=-1)


@configclass
class RecoverEventCfg(PivotEventCfg):
    """Reset directly into the fallen distribution: side-lying/upside-down."""

    pivot_reset = EventTerm(
        func=mdp.pivot_recover_reset,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", preserve_order=True)},
    )
    # No DR on the way up: the fall state distribution is already hard.
    randomize_friction = None
    randomize_mass = None
    randomize_com = None
    randomize_actuator_gains = None
    push_robot = None


@configclass
class RecoverRewardsCfg:
    upright_bonus = RewTerm(func=recover_upright_bonus, weight=1.5)
    contact_restored = RewTerm(func=recover_contact_restored, weight=1.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.015)
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    self_collision = RewTerm(
        func=mdp.self_collision_analytic, weight=-1.0
    )  # I6 stays on


@configclass
class RecoverTerminationsCfg(PivotTerminationsCfg):
    # Righting succeeded once all four wheels contact for stable duration.
    righted = DoneTerm(func=mdp.pivot_recover_righted, time_out=False)
    # Righting failed: still fallen after a full episode.
    still_fallen = DoneTerm(func=mdp.pivot_fallen)
    still_fallen.time_out = True


@configclass
class RecoverEnvCfg(PivotEnvCfg):
    """Reusable four-mode machinery  truncd for the recovery-only policy."""

    events: RecoverEventCfg = RecoverEventCfg()
    rewards: RecoverRewardsCfg = RecoverRewardsCfg()
    terminations: RecoverTerminationsCfg = RecoverTerminationsCfg()
    # π_recover trains at 50 Hz like π_main but on the fallen distribution.
    episode_length_s = 8.0

    def __post_init__(self):
        super().__post_init__()
        # Wheels stay limp (zero torque) during righting: only legs act.
        self.scene.robot.actuators["wheels"].effort_limit = PHYS.wheel_peak_torque
