#!/usr/bin/env python3
"""Guard the equal-value joint-symmetry convention used by M-TO2S shaping.

The audited URDF places the rear legs so that identical joint values on all
four legs produce a point-symmetric contact pattern (FL + HR ~ 0 and FR + HL ~
0 in the base XY plane). The M-TO2S point-symmetry constraints and the C++
independent validator depend on this convention through JOINT_SYMMETRY_PAIRS
and kJointSymmetryPairs. This test re-derives the convention from the URDF so
a future model change fails loudly here instead of producing silently
asymmetric "symmetric" poses.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

APPS = Path(__file__).resolve().parents[1] / "apps"
sys.path.insert(0, str(APPS))

import kinematic_equilibrium as kin
import static_equilibrium as base
import static_equilibrium_sideways as sideways


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf", type=Path, default=Path(os.environ["VQR_URDF_PATH"])
    )
    arguments = parser.parse_args()

    urdf_path = arguments.urdf.resolve()
    library_path = Path(
        os.environ.get("VQR_TREAD_CONTACT_LIBRARY", "")
    ).resolve()
    if not library_path.exists():
        fallback = Path(__file__).resolve().parents[1] / "build"
        candidates = sorted(fallback.glob("libtread_contact_c_api.so"))
        if not candidates:
            raise RuntimeError("Could not locate libtread_contact_c_api.so")
        library_path = candidates[0].resolve()
    helper = kin.TreadContactLibrary(library_path, urdf_path)
    evaluator = sideways.PinocchioSidewaysTerms(urdf_path, helper)

    pose = np.zeros(base.N_POSE)
    pose[0] = 0.45
    values = {"HipX": 0.0, "HipY": -0.65, "Knee": 1.3}
    for offset, name in enumerate(kin.LEG_JOINT_NAMES):
        joint_type = name.split("_")[1]
        pose[3 + offset] = values[joint_type]

    contacts, *_ = evaluator.evaluate_numpy(pose)
    fl_body, fr_body, hl_body, hr_body = evaluator.contacts_body_numpy(contacts)
    stance_residual = fl_body[:2] + hr_body[:2]
    swing_residual = fr_body[:2] + hl_body[:2]
    height_residual = abs(fr_body[2] - hl_body[2])

    tolerance = 5.0e-3
    if float(np.max(np.abs(stance_residual))) > tolerance:
        raise RuntimeError(
            "Equal joint values no longer give stance point symmetry: "
            f"{stance_residual}"
        )
    if float(np.max(np.abs(swing_residual))) > tolerance:
        raise RuntimeError(
            "Equal joint values no longer give swing point symmetry: "
            f"{swing_residual}"
        )
    if height_residual > tolerance:
        raise RuntimeError(
            "Equal joint values no longer give equal swing contact heights: "
            f"{height_residual}"
        )

    pair_names = (
        ("FL_HipX_joint", "HR_HipX_joint"),
        ("FR_HipX_joint", "HL_HipX_joint"),
        ("FL_HipY_joint", "HR_HipY_joint"),
        ("FR_HipY_joint", "HL_HipY_joint"),
        ("FL_Knee_joint", "HR_Knee_joint"),
        ("FR_Knee_joint", "HL_Knee_joint"),
    )
    index = {name: offset for offset, name in enumerate(kin.LEG_JOINT_NAMES)}
    derived_pairs = {
        tuple(sorted((3 + index[first], 3 + index[second])))
        for first, second in pair_names
    }
    python_pairs = {tuple(sorted(pair)) for pair in sideways.JOINT_SYMMETRY_PAIRS}
    if python_pairs != derived_pairs:
        raise RuntimeError("JOINT_SYMMETRY_PAIRS no longer match the URDF pairs")

    print("joint symmetry convention = PASS")
    print(f"stance residual xy = {np.round(stance_residual, 9).tolist()}")
    print(f"swing residual xy = {np.round(swing_residual, 9).tolist()}")
    print(f"swing height difference = {height_residual:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
