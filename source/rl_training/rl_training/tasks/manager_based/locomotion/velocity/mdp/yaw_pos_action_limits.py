"""Tensor-only target limiting shared by POS training and deployment references."""

import torch


def limit_position_target(goal, position, velocity, lower, upper, max_velocity: float, max_acceleration: float, dt: float):
    """Advance a bounded target with acceleration and braking-distance limits.

    A reversing goal may be overshot while braking; hard joint bounds may not.
    Reserving braking distance at each bound avoids an instantaneous velocity
    reset when an adversarial action asks for a joint-limit crossing.
    """
    if min(max_velocity, max_acceleration, dt) <= 0:
        raise ValueError("Target limits and dt must be positive.")
    goal = goal.clamp(min=lower, max=upper)
    error = goal - position
    dv = max_acceleration * dt

    def stopping_speed(distance):
        return (dv * dv + 2. * max_acceleration * distance.clamp_min(0.)).sqrt() - dv

    desired_velocity = error.sign() * stopping_speed(error.abs()).clamp_max(max_velocity)
    next_velocity = desired_velocity.clamp(min=velocity - dv, max=velocity + dv).clamp(-max_velocity, max_velocity)
    next_velocity = next_velocity.clamp(min=-stopping_speed(position - lower), max=stopping_speed(upper - position))
    next_position = (position + dt * next_velocity).clamp(min=lower, max=upper)
    return next_position, next_velocity


def limit_velocity_target(goal, current, max_velocity: float, max_acceleration: float, dt: float):
    if min(max_velocity, max_acceleration, dt) <= 0:
        raise ValueError("Target limits and dt must be positive.")
    goal = goal.clamp(-max_velocity, max_velocity)
    return current + (goal - current).clamp(-max_acceleration * dt, max_acceleration * dt)
