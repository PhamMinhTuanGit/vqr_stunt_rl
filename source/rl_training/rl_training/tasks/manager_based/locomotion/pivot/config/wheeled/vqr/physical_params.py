# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Single source of truth for every VQRWheel pivot-task physical constant.

Nothing in this module imports Isaac Lab so that ``supervisor_gate`` and the
``state`` subpackage stay deployable (invariant I9).  Derived quantities are
computed once in ``__post_init__``; hand-written duplicates elsewhere are a
bug (section 2/11 of the pivot specification).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


@dataclass(frozen=True)
class VQRPhysicalParams:
    """Audited physical parameters of the VQRWheel articulation (SSOT)."""

    # -- geometry -----------------------------------------------------------
    track_width: float = 0.4730  # W [m], lateral wheel separation
    wheelbase: float = 0.4968  # L [m], front/rear axle separation
    wheel_radius: float = 0.091  # R [m]

    # -- mass ---------------------------------------------------------------
    mass: float = 32.86  # m [kg]
    # CoM longitudinal offset ahead of the rear axle at the balance pose.
    com_ahead_of_rear_axle: float = 0.2382  # d [m]
    # CoM height above the ground contact at the balance pose.
    com_height_balance: float = 0.4424  # h [m]
    gravity: float = 9.81  # [m/s^2]

    # -- balance dynamics ----------------------------------------------------
    # Rigid-body tip-over natural frequency; NOT the point-mass 4.709 value.
    omega0: float = 4.118  # [rad/s]
    tip_time_constant: float = 0.2429  # tau [s]
    # Nominal balance pitch (deg) and its spin-moment feed-forward coefficient
    # per (rad/s)^2: theta*(w) = theta_star_0 + c * w^2  (invariant I1).
    theta_star_0_deg: float = 32.58
    theta_star_spin_coeff_deg: float = 0.0931
    # |zeta| sustainable indefinitely at the 3 Nm continuous wheel torque [m].
    zeta_thermal: float = 0.0906
    # Peak normal-force budget for LAND (= 2 m g) [N].
    fn_peak_budget: float = 645.0

    # -- inertia (balance pose) ----------------------------------------------
    ixx: float = 1.28498
    iyy: float = 1.97865
    izz: float = 1.81878
    i_xz: float = -0.02332  # [kg m^2], sign per URDF convention
    # Yaw-axis inertia at the balance pose for spin feed-forward [kg m^2].
    i_spin: float = 1.685

    # -- actuation ------------------------------------------------------------
    wheel_peak_torque: float = 24.0  # [N m]
    wheel_continuous_torque: float = 3.0  # [N m]
    wheel_velocity_limit: float = 58.90  # [rad/s]
    leg_peak_torque: float = 60.0  # [N m]
    leg_velocity_limit: float = 14.66  # [rad/s]

    # -- joint limits [rad] ----------------------------------------------------
    # HAA (HipX), HFE (HipY), KFE (Knee) soft limits.
    haa_limit: float = _deg2rad(45.0)  # +/-45 deg
    hfe_lower_limit: float = _deg2rad(-194.8)
    hfe_upper_limit: float = _deg2rad(137.5)
    kfe_lower_limit: float = _deg2rad(46.5)
    kfe_upper_limit: float = _deg2rad(158.5)

    # -- supervisor capture-point safety fractions (section 3) -----------------
    xi_safe_fraction: float = 0.50  # xi_safe = 0.50 * mu_hat * h
    xi_max_fraction: float = 0.85  # xi_max  = 0.85 * mu_hat * h

    # -- runtime control --------------------------------------------------------
    control_dt: float = 0.02  # 50 Hz policy
    physics_dt: float = 0.005  # 200 Hz physics
    thermal_time_constant: float = 2.0  # EMA |tau_w| [s]
    hard_state_window: float = 0.30  # ring-buffer horizon for resets [s]
    wheel_thermal_limit: float = 3.0  # continuous torque [N m]
    command_history: int = 5  # observation history stack [frames]

    # -- spin limits per mode (invariant I10) -----------------------------------
    omega_z_limit_ground: float = 3.0
    omega_z_limit_balance: float = 6.0
    delta_theta_command_limit: float = _deg2rad(8.0)  # +/-8 deg around theta*
    # Pitch setpoint bias towards the front (invariant I5, 3-5 deg).
    pitch_setpoint_forward_bias_deg: float = 4.0
    # Backflip penalty adds its configured margin to theta*(omega_z)+delta.
    grace_window: float = 0.5  # termination grace window [s]
    termination_drift: float = 0.20  # 10 s BALANCE drift target conversion [m]

    def __post_init__(self) -> None:
        # Derived quantities: never hand-write these anywhere else.
        object.__setattr__(self, "theta_star_0", _deg2rad(self.theta_star_0_deg))
        object.__setattr__(self, "theta_star_spin_coeff", _deg2rad(self.theta_star_spin_coeff_deg))
        object.__setattr__(self, "pitch_setpoint_forward_bias", _deg2rad(self.pitch_setpoint_forward_bias_deg))
        object.__setattr__(self, "inertia_diag", (self.ixx, self.iyy, self.izz))
        object.__setattr__(self, "weight", self.mass * self.gravity)
        # Sustained torque equilibrium offset cross-check: m g zeta = tau.
        object.__setattr__(self, "zeta_at_continuous_torque", self.wheel_continuous_torque / self.weight)

    # -- feed-forward / safety formulas ---------------------------------------
    def theta_star(self, omega_z):
        """theta*(omega_z) = 32.58 deg + 0.0931 deg * omega_z^2 [rad]."""
        omega_z = omega_z if not hasattr(omega_z, "dtype") else omega_z
        return self.theta_star_0 + self.theta_star_spin_coeff * omega_z ** 2

    def xi_safe(self, mu_hat):
        """Capture-point safety radius xi_safe = 0.50 * mu_hat * h [m]."""
        return self.xi_safe_fraction * mu_hat * self.com_height_balance

    def xi_max(self, mu_hat):
        """Hard capture-point radius xi_max = 0.85 * mu_hat * h [m]."""
        return self.xi_max_fraction * mu_hat * self.com_height_balance

    @staticmethod
    def capture_point(zeta, zeta_dot, omega0: float = 4.118):
        """xi = zeta + zeta_dot / omega0 (section 3)."""
        return zeta + zeta_dot / omega0


# Module-level singleton used by every pivot-four-mode module.
VQR_PHYSICS = VQRPhysicalParams()


# Safe defaults for the deploy-time friction estimate until ``mu_estimator``
# provides operator input plus observer correction (section 12).
DEFAULT_MU_HAT = 1.0
