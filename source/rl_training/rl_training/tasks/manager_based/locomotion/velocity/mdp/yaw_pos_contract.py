"""CPU helpers for the versioned POS observation/action contract."""

import torch


def validate_pos_checkpoint(checkpoint_path, expected_contract):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actual = (payload.get("infos") or {}).get("yaw_pos_contract")
    if not isinstance(actual, dict):
        raise ValueError("Legacy POS checkpoint has no yaw_pos_contract. Train this USD/action contract from scratch.")
    for key in expected_contract:
        if key == "usd_path":  # A copied asset with identical layers has the same hash.
            continue
        if actual.get(key) != expected_contract[key]:
            raise ValueError(f"POS checkpoint contract differs at {key}; use its original configuration or train from scratch.")
    return payload


def initialize_pos_exploration(policy, leg_std=0.5, wheel_std=0.3):
    if leg_std <= 0 or wheel_std <= 0:
        raise ValueError("Initial POS exploration std must be positive.")
    if getattr(policy, "noise_std_type", None) != "log" or policy.log_std.numel() != 16:
        raise ValueError("POS exploration requires 16 independent log std parameters.")
    with torch.no_grad():
        policy.log_std[:12].fill_(float(torch.tensor(leg_std).log()))
        policy.log_std[12:].fill_(float(torch.tensor(wheel_std).log()))
