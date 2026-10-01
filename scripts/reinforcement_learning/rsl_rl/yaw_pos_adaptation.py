"""Strict POS weight initialization with a fresh PPO optimizer and run state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml

from yaw_pos_transfer import _ConfigLoader, _canonical, policy_contract

TASK_ID = "Flat-VQR-Wheel-Yaw-POS"
_AGENT_CONTRACT = (
    "policy", "algorithm", "obs_groups", "num_steps_per_env", "clip_actions", "empirical_normalization",
)


def checkpoint_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_checkpoint_contract(checkpoint_path, env_config, agent_config):
    """Reject observation/action reordering and any PPO/network configuration drift."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    configs = []
    for name in ("env", "agent"):
        config_path = path.parent / "params" / f"{name}.yaml"
        if not config_path.is_file():
            raise RuntimeError(f"Missing {config_path}; cannot verify adaptation compatibility.")
        with config_path.open() as stream:
            configs.append(yaml.load(stream, Loader=_ConfigLoader))
    source_env, source_agent = configs
    command_class = source_env["commands"]["yaw_rate_cmd"]["class_type"]
    if command_class != "rl_training.tasks.manager_based.locomotion.velocity.mdp.yaw_pos_commands:YawPosCommand":
        raise RuntimeError("Adaptation source must be a Flat-VQR-Wheel-Yaw-POS checkpoint.")
    current = env_config.to_dict() if hasattr(env_config, "to_dict") else env_config
    expected, actual = policy_contract(source_env), policy_contract(current)
    for key in expected:
        if expected[key] != actual[key]:
            raise RuntimeError(f"Checkpoint observation/action contract mismatch in {key}; refusing adaptation.")
    for key in _AGENT_CONTRACT:
        if _canonical(source_agent.get(key)) != _canonical(agent_config.get(key)):
            raise RuntimeError(f"Checkpoint PPO/network contract mismatch in {key}; refusing adaptation.")
    return path, actual


def optimizer_snapshot(optimizer):
    steps = [int(state["step"]) for state in optimizer.state.values() if "step" in state]
    return {"state_entries": len(optimizer.state), "steps": sorted(set(steps)),
            "max_step": max(steps, default=0), "learning_rates": [g["lr"] for g in optimizer.param_groups]}


def policy_matches(policy, source_state):
    current = policy.state_dict()
    return {
        "actor_exact": all(torch.equal(current[k].detach().cpu(), v) for k, v in source_state.items() if k.startswith("actor.")),
        "critic_exact": all(torch.equal(current[k].detach().cpu(), v) for k, v in source_state.items() if k.startswith("critic.")),
        "log_std_exact": torch.equal(current["log_std"].detach().cpu(), source_state["log_std"]),
        "all_state_exact": set(current) == set(source_state) and all(
            torch.equal(current[k].detach().cpu(), v) for k, v in source_state.items()
        ),
    }


def _write_report(runner, report):
    if runner.log_dir is not None:
        path = Path(runner.log_dir) / "adaptation_preflight.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")


def load_pos_policy(runner, task_env, checkpoint_path, agent_config):
    """Load model_state_dict directly; never invoke runner.load or optimizer.load_state_dict."""
    path, contract = audit_checkpoint_contract(checkpoint_path, task_env.cfg, agent_config)
    if task_env.common_step_counter != 0:
        raise RuntimeError("Adaptation initialization must happen before the first environment step.")
    if runner.log_dir is not None and Path(runner.log_dir).resolve() == path.parent:
        raise RuntimeError("Adaptation must use a separate run directory; source checkpoint is protected.")
    optimizer = runner.alg.optimizer
    if optimizer.state:
        raise RuntimeError("Adaptation requires a freshly constructed optimizer with empty state.")
    source_hash = checkpoint_sha256(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    source_state, target_state = saved["model_state_dict"], runner.alg.policy.state_dict()
    if set(source_state) != set(target_state) or "log_std" not in source_state:
        raise RuntimeError("Strict policy key mismatch; partial actor/critic/log_std loading is forbidden.")
    if not any(k.startswith("actor.") for k in source_state) or not any(k.startswith("critic.") for k in source_state):
        raise RuntimeError("Checkpoint must contain both actor and critic.")
    for key, value in source_state.items():
        if value.shape != target_state[key].shape or value.dtype != target_state[key].dtype:
            raise RuntimeError(f"Strict policy shape/dtype mismatch in {key}.")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Checkpoint contains non-finite policy state in {key}.")
    runner.alg.policy.load_state_dict(source_state, strict=True)
    matches = policy_matches(runner.alg.policy, source_state)
    if not all(matches.values()) or runner.alg.optimizer is not optimizer or optimizer.state:
        raise RuntimeError("Policy-only load failed to preserve exact weights and a fresh optimizer.")
    runner.current_learning_iteration = 0
    runner.tot_timesteps = 0
    runner.tot_time = 0.0
    # No checkpoint infos are read: initialize the configured curriculum from scratch.
    for name in list(vars(task_env)):
        if name.startswith("_yaw_task_curriculum_"):
            delattr(task_env, name)
    curriculum = task_env.cfg.curriculum.task_levels
    curriculum.func(task_env, [], **curriculum.params)
    task_env.reset()
    if task_env._yaw_task_curriculum_stage != 0 or task_env._yaw_task_curriculum_yaw_stage != 0:
        raise RuntimeError("Adaptation curriculum must start at stage zero.")
    collapse = task_env.reward_manager.get_term_cfg("support_y_collapse")
    if collapse.weight != -0.25 or collapse.params["minimum_separation"] != 0.45 or collapse.params["separation_scale"] != 0.05:
        raise RuntimeError("Adaptation requires support_y_collapse weight=-0.25, minimum=0.45, scale=0.05.")
    report = {
        "source_checkpoint": str(path), "source_sha256": source_hash, "source_iteration": int(saved["iter"]),
        "weights_after_load": matches, "strict_load": True, "model_state_tensors": len(source_state),
        "optimizer_restored": False, "optimizer_before_update": optimizer_snapshot(optimizer),
        "start_iteration": 0, "curriculum_restored": False, "clearance_stage": 0, "yaw_stage": 0,
        "support_y_collapse_weight": collapse.weight, "minimum_separation": 0.45, "separation_scale": 0.05,
        "policy_terms": [name for name, _ in contract["policy_terms"]],
        "critic_terms": [name for name, _ in contract["critic_terms"]],
        "action_terms": [name for name, _ in contract["actions"]],
        "ppo_network_contract_unchanged": True, "ppo_updates_completed": 0,
    }
    _write_report(runner, report)
    print("[POS adaptation] " + json.dumps(report, sort_keys=True))
    original_update = runner.alg.update

    def audited_update(*args, **kwargs):
        first = report["ppo_updates_completed"] == 0
        if first:
            report["weights_before_first_update"] = policy_matches(runner.alg.policy, source_state)
            report["optimizer_before_update"] = optimizer_snapshot(optimizer)
            if not all(report["weights_before_first_update"].values()) or optimizer.state:
                raise RuntimeError("Weights or optimizer changed before the first adaptation PPO update.")
            _write_report(runner, report)
        result = original_update(*args, **kwargs)
        report["ppo_updates_completed"] += 1
        if first:
            report["optimizer_after_first_update"] = optimizer_snapshot(optimizer)
            report["weights_after_first_update"] = policy_matches(runner.alg.policy, source_state)
            report["policy_finite_after_first_update"] = all(
                torch.isfinite(v).all().item() for v in runner.alg.policy.state_dict().values()
            )
            report["first_update_losses"] = {k: float(v) for k, v in result.items()}
            _write_report(runner, report)
        return result

    runner.alg.update = audited_update
    return report


def finish_adaptation_audit(runner, report):
    """Persist the completed update count and verify the original checkpoint stayed intact."""
    report["source_sha256_after"] = checkpoint_sha256(report["source_checkpoint"])
    report["source_checkpoint_unchanged"] = report["source_sha256_after"] == report["source_sha256"]
    _write_report(runner, report)
    if not report["source_checkpoint_unchanged"]:
        raise RuntimeError("Source checkpoint changed during adaptation.")
    print("[POS adaptation completed] " + json.dumps(report, sort_keys=True))
