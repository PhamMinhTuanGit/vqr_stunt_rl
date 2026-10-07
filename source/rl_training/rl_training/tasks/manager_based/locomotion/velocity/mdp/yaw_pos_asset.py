"""Read the composed USD and derive the POS reset/deployment contract on CPU.

This module deliberately imports neither Isaac Lab nor Isaac Sim. Joint frames,
limits and collision centers come from the USD actually used for training.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


LEGS = ("FL", "FR", "HL", "HR")
LEG_JOINTS = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in ("HipX", "HipY", "Knee"))
WHEEL_JOINTS = tuple(f"{leg}_WHEEL" for leg in LEGS)
CONTRACT_VERSION = 2


def _rotation(quaternion) -> np.ndarray:
    from pxr import Gf

    # Gf matrices use row vectors; all math below uses column vectors.
    return np.asarray(Gf.Matrix3d(Gf.Quatd(quaternion)), dtype=float).T


def _frame(position, quaternion) -> np.ndarray:
    frame = np.eye(4)
    frame[:3, :3] = _rotation(quaternion)
    frame[:3, 3] = np.asarray(position, dtype=float)
    return frame


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
    return result


def inspect_pos_asset(usd_path: str, target_height: float = 0.49) -> dict:
    """Validate the VQR wheel asset and solve a flat, four-wheel neutral pose.

    HipX stays at zero. HipY/Knee are solved separately per leg so that each
    collision center is below its HipY joint and the tire touches z=0 with the
    torso at target_height. No old URDF dimensions or mass values are reused.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.optimize import least_squares

    path = Path(usd_path).resolve()
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open POS USD: {path}")
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise ValueError("POS USD must use meters.")
    bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if "TORSO" not in bodies:
        raise ValueError("POS USD must contain rigid body TORSO.")
    joints = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(parents) != 1 or len(children) != 1:
            raise ValueError(f"Invalid bodies for {prim.GetName()}")
        axis = np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())]
        joints[prim.GetName()] = dict(
            parent=parents[0].name, child=children[0].name, axis=axis,
            frame0=_frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
            frame1=_frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
            limits=np.deg2rad([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()]),
        )
    missing = set(LEG_JOINTS + WHEEL_JOINTS) - joints.keys()
    if missing:
        raise ValueError(f"POS USD missing joints: {sorted(missing)}")
    cache = UsdGeom.XformCache()
    wheels = {}
    for name in WHEEL_JOINTS:
        body = bodies[name]
        colliders = [p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                     if p.HasAPI(UsdPhysics.CollisionAPI)]
        if len(colliders) != 1 or not colliders[0].IsA(UsdGeom.Cylinder):
            raise ValueError(f"{name} requires one cylindrical tire collider.")
        cylinder = UsdGeom.Cylinder(colliders[0])
        relative = np.asarray(cache.GetLocalToWorldTransform(colliders[0]), dtype=float).T
        relative = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body), dtype=float).T) @ relative
        axes_scale = np.linalg.norm(relative[:3, :3], axis=0)
        if not np.allclose(axes_scale, 1.0, atol=1e-5):
            raise ValueError(f"Scaled tire collider on {name}; bake its scale first.")
        axle = relative[:3, :3] @ np.eye(3)["XYZ".index(cylinder.GetAxisAttr().Get())]
        if not np.allclose(np.abs(axle), [0., 1., 0.], atol=2e-3):
            raise ValueError(f"{name} tire axle must align with wheel-body Y.")
        positive_axis = joints[name]["frame1"][:3, :3] @ joints[name]["axis"]
        if not np.allclose(np.abs(positive_axis), [0., 1., 0.], atol=2e-3):
            raise ValueError(f"{name} joint axis must align with wheel-body Y.")
        wheels[name] = dict(
            center=relative[:3, 3], axle=axle,
            radius=float(cylinder.GetRadiusAttr().Get()),
            half_width=float(cylinder.GetHeightAttr().Get()) / 2.,
            motor_axis_sign=float(np.sign(positive_axis[1])),
        )
    radii = [w["radius"] for w in wheels.values()]
    if not np.allclose(radii, radii[0], atol=1e-6):
        raise ValueError("POS requires equal wheel radii.")

    def forward(leg, angles):
        transform = np.eye(4)
        for joint_name, angle in zip(
            [f"{leg}_{name}_joint" for name in ("HipX", "HipY", "Knee")] + [f"{leg}_WHEEL"],
            [*angles, 0.],
        ):
            joint = joints[joint_name]
            transform = transform @ joint["frame0"] @ _axis_rotation(joint["axis"], angle) @ np.linalg.inv(joint["frame1"])
        wheel = wheels[f"{leg}_WHEEL"]
        center = transform[:3, :3] @ wheel["center"] + transform[:3, 3]
        axle = transform[:3, :3] @ wheel["axle"]
        vertical_extent = wheel["radius"] * np.sqrt(max(0., 1. - axle[2] ** 2)) + wheel["half_width"] * abs(axle[2])
        return center, vertical_extent

    reference, centers, extents = [], [], []
    for leg in LEGS:
        hipx, hipy, knee = [joints[f"{leg}_{name}_joint"] for name in ("HipX", "HipY", "Knee")]
        lower, upper = np.array([hipy["limits"][0], knee["limits"][0]]), np.array([hipy["limits"][1], knee["limits"][1]])
        if not np.all(np.isfinite([lower, upper])) or np.any(lower >= upper):
            raise ValueError(f"Invalid finite leg limits on {leg}")
        hipy_origin = (hipx["frame0"] @ np.linalg.inv(hipx["frame1"]) @ hipy["frame0"])[:3, 3]

        def errors(angles):
            center, extent = forward(leg, [0., *angles])
            return [center[0] - hipy_origin[0], target_height + center[2] - extent]

        solution = least_squares(errors, np.clip([-.5, 1.0], lower + 1e-5, upper - 1e-5), bounds=(lower, upper),
                                 xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if np.max(np.abs(errors(solution.x))) > 1e-5:
            raise ValueError(f"Cannot stand at {target_height} m within {leg} joint limits.")
        reference.extend([0., *solution.x.tolist()])
        centers.append(forward(leg, [0., *solution.x])[0].tolist())
        extents.append(float(forward(leg, [0., *solution.x])[1]))

    layers = sorted((Path(layer.realPath) for layer in stage.GetUsedLayers() if layer.realPath), key=lambda p: str(p))
    digest = hashlib.sha256()
    for layer in layers:
        digest.update(str(layer.relative_to(path.parent) if layer.is_relative_to(path.parent) else layer.name).encode())
        digest.update(layer.read_bytes())
    masses = {name: float(UsdPhysics.MassAPI(body).GetMassAttr().Get() or 0.) for name, body in bodies.items()}
    if any(value <= 0 for value in masses.values()):
        raise ValueError("Every POS rigid body must have a positive authored mass.")
    inertias = {name: list(UsdPhysics.MassAPI(body).GetDiagonalInertiaAttr().Get()) for name, body in bodies.items()}
    if any(not np.all(np.isfinite(value)) or min(value) <= 0 for value in inertias.values()):
        raise ValueError("Every POS rigid body must have positive finite principal inertia.")
    return dict(
        version=CONTRACT_VERSION, usd_path=str(path), usd_sha256=digest.hexdigest(),
        total_mass_kg=sum(masses.values()), body_masses_kg=masses, body_inertias_kg_m2=inertias,
        target_height=target_height, leg_joint_names=list(LEG_JOINTS), wheel_joint_names=list(WHEEL_JOINTS),
        default_leg_positions=reference, leg_position_limits=[joints[name]["limits"].tolist() for name in LEG_JOINTS],
        wheel_radius=radii[0], wheel_center_offsets={name: wheel["center"].tolist() for name, wheel in wheels.items()},
        wheel_half_widths={name: wheel["half_width"] for name, wheel in wheels.items()},
        wheel_motor_axis_signs={name: wheel["motor_axis_sign"] for name, wheel in wheels.items()},
        neutral_wheel_centers=centers, neutral_wheel_vertical_extents=extents,
    )
