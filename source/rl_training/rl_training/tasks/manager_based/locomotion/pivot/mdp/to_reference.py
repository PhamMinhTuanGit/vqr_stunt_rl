"""Pose-only TO bootstrap and batched transition math (no simulator dependency)."""

import math
from pathlib import Path

import torch
import yaml


REFERENCE_PATH = "pose_optimization/output/static_equilibrium_FL_HR_sideways.yaml"


def load_leg_reference(
    joint_names,
    reference_path=REFERENCE_PATH,
    expected_stance=("FL", "HR"),
    expected_swing=("FR", "HL"),
):
    path = Path(reference_path).expanduser()
    if not path.is_absolute():
        roots = [p for p in Path(__file__).resolve().parents if (p / "scripts").is_dir() and (p / "source").is_dir()]
        if not roots:
            raise FileNotFoundError("Cannot locate repo root; provide an absolute TO reference_path")
        path = roots[0] / path
    with path.open(encoding="utf-8") as stream:
        saved = yaml.safe_load(stream)
    if (
        saved.get("result") != "PASS"
        or saved.get("stance") != list(expected_stance)
        or saved.get("swing") != list(expected_swing)
    ):
        raise ValueError(
            f"Expected validated {expected_stance[0]}-{expected_stance[1]} TO reference"
        )
    positions = saved["variables"]["leg_joints"]
    if len(joint_names) != 12 or len(set(joint_names)) != 12 or set(positions) != set(joint_names):
        raise ValueError("TO reference must contain exactly the 12 named leg joints")
    values = [float(positions[name]) for name in joint_names]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite TO leg pose")
    # Intentionally expose only positions, never forces or actuator torques.
    return dict(zip(joint_names, values, strict=True))


def smoothstep(phase):
    phase = phase.clamp(0.0, 1.0)
    return phase.square() * (3.0 - 2.0 * phase)


def pose_reference(standing, target, phase):
    return standing + smoothstep(phase).unsqueeze(-1) * (target - standing)


def unload_fraction(phase):
    return smoothstep((phase - 0.4) / 0.4)


def contact_targets(phase):
    """FL, FR, HL, HR contact preference; continuous at .4 and .8."""
    lifted = 1.0 - unload_fraction(phase)
    return torch.stack((torch.ones_like(phase), lifted, lifted, torch.ones_like(phase)), dim=-1)
