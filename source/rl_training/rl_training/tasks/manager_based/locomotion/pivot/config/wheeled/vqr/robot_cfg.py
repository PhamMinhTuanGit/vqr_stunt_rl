# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Audited Isaac Lab configuration for the wheeled VQR articulation.

The names and ordering below are taken from the ``VQRWheel.usd`` articulation
as initialized by Isaac Lab/PhysX.  The URDF and USD prims use the same names,
but their declaration/traversal order differs from the runtime DOF order.
They intentionally remain explicit here so later pivot task configuration can
import these constants instead of maintaining its own name patterns.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from rl_training.assets import ISAACLAB_ASSETS_DATA_DIR


VQR_USD_PATH = f"{ISAACLAB_ASSETS_DATA_DIR}/VQRWheel/VQRWheel_usd/VQRWheel.usd"

BASE_LINK_NAME = "TORSO"

# Runtime articulation order within the leg group: HipX, HipY, then knee.
LEG_JOINT_NAMES = [
    "FL_HipX_joint",
    "FR_HipX_joint",
    "HL_HipX_joint",
    "HR_HipX_joint",
    "FL_HipY_joint",
    "FR_HipY_joint",
    "HL_HipY_joint",
    "HR_HipY_joint",
    "FL_Knee_joint",
    "FR_Knee_joint",
    "HL_Knee_joint",
    "HR_Knee_joint",
]

WHEEL_JOINT_NAMES = [
    "FL_WHEEL",
    "FR_WHEEL",
    "HL_WHEEL",
    "HR_WHEEL",
]

WHEEL_BODY_NAMES = [
    "FL_WHEEL",
    "FR_WHEEL",
    "HL_WHEEL",
    "HR_WHEEL",
]

# The initial pivot task always uses this diagonal; it must not switch at run time.
SUPPORT_WHEEL_NAMES = [WHEEL_BODY_NAMES[0], WHEEL_BODY_NAMES[3]]  # FL + HR
LIFTED_WHEEL_NAMES = [WHEEL_BODY_NAMES[1], WHEEL_BODY_NAMES[2]]  # FR + HL

# Isaac Lab/PhysX articulation DOF ordering.
ARTICULATION_JOINT_NAMES = [*LEG_JOINT_NAMES, *WHEEL_JOINT_NAMES]

# Isaac Lab/PhysX articulation link ordering, rooted at the TORSO body.
ARTICULATION_BODY_NAMES = [
    BASE_LINK_NAME,
    "FL_HIP",
    "FR_HIP",
    "HL_HIP",
    "HR_HIP",
    "FL_THIGH",
    "FR_THIGH",
    "HL_THIGH",
    "HR_THIGH",
    "FL_SHANK",
    "FR_SHANK",
    "HL_SHANK",
    "HR_SHANK",
    *WHEEL_BODY_NAMES,
]

CONTROLLED_JOINT_NAMES = ARTICULATION_JOINT_NAMES.copy()


# These conservative values are carried over unchanged from the repository's
# existing VQRWHEEL_CFG.  The URDF supplies the 60 Nm and 14.66 rad/s leg
# limits, but its continuous wheel joints contain no limits; 20 Nm and
# 58.90 rad/s are therefore the safest repository-backed wheel assumptions.
LEG_EFFORT_LIMIT = 60.0
LEG_VELOCITY_LIMIT = 14.66
LEG_STIFFNESS = 80.0
LEG_DAMPING = {
    name: 1.524 if "HipX" in name else 1.431 if "HipY" in name else 0.859
    for name in LEG_JOINT_NAMES
}

WHEEL_EFFORT_LIMIT = 20.0
WHEEL_VELOCITY_LIMIT = 58.90
WHEEL_STIFFNESS = 0.0
WHEEL_DAMPING = 0.6

DEFAULT_JOINT_POS = {
    **{name: 0.0 for name in LEG_JOINT_NAMES if "HipX" in name},
    **{name: -0.65 for name in LEG_JOINT_NAMES if "HipY" in name},
    **{name: 1.3 for name in LEG_JOINT_NAMES if "Knee" in name},
    **{name: 0.0 for name in WHEEL_JOINT_NAMES},
}


VQR_CFG = ArticulationCfg(
    articulation_root_prim_path=f"/{BASE_LINK_NAME}",
    spawn=sim_utils.UsdFileCfg(
        usd_path=VQR_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos=DEFAULT_JOINT_POS.copy(),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=LEG_JOINT_NAMES,
            effort_limit=LEG_EFFORT_LIMIT,
            velocity_limit=LEG_VELOCITY_LIMIT,
            stiffness=LEG_STIFFNESS,
            damping=LEG_DAMPING,
            friction=0.0,
            armature=0.0,
            min_delay=2,
            max_delay=8,
        ),
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=WHEEL_JOINT_NAMES,
            effort_limit=WHEEL_EFFORT_LIMIT,
            velocity_limit=WHEEL_VELOCITY_LIMIT,
            stiffness=WHEEL_STIFFNESS,
            damping=WHEEL_DAMPING,
            friction=0.0,
            armature=0.0025,
            min_delay=2,
            max_delay=8,
        ),
    },
)
