"""Strict policy-only initialization and observation/action compatibility audit."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

TASK_ID = "Flat-VQR-Wheel-Yaw-POS-Transfer"


class _ConfigLoader(yaml.SafeLoader):
    """Read Isaac Lab YAML tuples/slices without constructing arbitrary objects."""


_ConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: loader.construct_sequence(node))
_ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: slice(*loader.construct_sequence(node)),
)


def _canonical(value):
    if isinstance(value, dict):
        # Managers resolve IDs after loading; ordered names define this mapping.
        return {key: _canonical(item) for key, item in value.items()
                if key not in ("joint_ids", "body_ids", "fixed_tendon_ids", "object_collection_ids")}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, slice):
        return [value.start, value.stop, value.step]
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    return value


def policy_contract(config):
    """Keep term ordering explicit; compare preprocessing and entity ordering too."""
    def terms(group):
        return [(name, _canonical(term)) for name, term in group.items() if isinstance(term, dict) and "func" in term]

    observations = config["observations"]
    return {
        "policy_terms": terms(observations["policy"]),
        "critic_terms": terms(observations["critic"]),
        "observation_groups": {
            group: {key: _canonical(value) for key, value in cfg.items()
                    if not isinstance(value, dict) and value is not None}
            for group, cfg in observations.items()
        },
        "actions": [(name, _canonical(term)) for name, term in config["actions"].items() if term is not None],
        "leg_joint_names": config["leg_joint_names"],
        "wheel_joint_names": config["wheel_joint_names"],
        "default_joint_positions": _canonical(config["scene"]["robot"]["init_state"]["joint_pos"]),
    }


def audit_checkpoint_contract(checkpoint_path, env_config):
    path = Path(checkpoint_path).expanduser().resolve()
    source_config = path.parent / "params/env.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    if not source_config.is_file():
        raise RuntimeError(f"Missing {source_config}; cannot verify checkpoint observation/action ordering.")
    with source_config.open() as stream:
        source = yaml.load(stream, Loader=_ConfigLoader)
    command_class = source["commands"]["yaw_rate_cmd"]["class_type"]
    if command_class != "rl_training.tasks.manager_based.locomotion.velocity.mdp.commands:YawRateCommand":
        raise RuntimeError("Transfer source must be a Flat-VQR-Wheel-Yaw checkpoint (scalar YawRateCommand).")
    current = env_config.to_dict() if hasattr(env_config, "to_dict") else env_config
    expected, actual = policy_contract(source), policy_contract(current)
    for key in expected:
        if expected[key] != actual[key]:
            raise RuntimeError(f"Checkpoint observation/action contract mismatch in {key}; refusing transfer.")
    return path, actual


def load_base_policy(runner, task_env, checkpoint_path):
    """Load all actor/critic/noise state, keeping a new optimizer and curriculum."""
    path, contract = audit_checkpoint_contract(checkpoint_path, task_env.cfg)
    if task_env.common_step_counter != 0:
        raise RuntimeError("Transfer initialization must happen before the first environment step.")
    optimizer = runner.alg.optimizer
    if optimizer.state:
        raise RuntimeError("Transfer requires a freshly created optimizer.")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    source_state = saved["model_state_dict"]
    target_state = runner.alg.policy.state_dict()
    if source_state.keys() != target_state.keys():
        raise RuntimeError("Checkpoint policy state keys differ; partial loading is prohibited.")
    for key, tensor in source_state.items():
        if tensor.shape != target_state[key].shape:
            raise RuntimeError(f"Checkpoint shape mismatch for {key}: {tuple(tensor.shape)} != {tuple(target_state[key].shape)}")
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Checkpoint contains non-finite policy state: {key}")
    # RSL-RL 3.1.2 also restores the iteration with load_optimizer=False.
    # Explicitly reset it; ignore infos containing the old curriculum.
    runner.load(str(path), load_optimizer=False, map_location=runner.device)
    runner.current_learning_iteration = 0
    runner.tot_timesteps = 0
    runner.tot_time = 0
    if runner.alg.optimizer is not optimizer or optimizer.state:
        raise RuntimeError("Transfer changed or populated the fresh optimizer.")
    for key, tensor in source_state.items():
        if not torch.equal(tensor, runner.alg.policy.state_dict()[key].cpu()):
            raise RuntimeError(f"Policy transfer was not exact: {key}")

    for name in tuple(vars(task_env)):
        if name.startswith(("_yaw_task_curriculum_", "_yaw_pos_transfer_curriculum_")):
            delattr(task_env, name)
    term = task_env.cfg.curriculum.task_levels
    term.func(task_env, [], **term.params)
    if task_env._yaw_task_curriculum_stage != 0 or task_env._yaw_task_curriculum_yaw_stage != 0:
        raise RuntimeError("Transfer curriculum did not start at stage zero.")
    # Apply initial-stage reset DR and sample fresh commands.
    task_env.reset()
    report = {
        "checkpoint": str(path), "source_iteration": int(saved["iter"]), "iteration": 0,
        "actor_dimension": source_state["actor.0.weight"].shape[1],
        "critic_dimension": source_state["critic.0.weight"].shape[1],
        "action_dimension": source_state["actor.6.weight"].shape[0],
        "policy_terms": [name for name, _ in contract["policy_terms"]],
        "critic_terms": [name for name, _ in contract["critic_terms"]],
        "action_order": contract["leg_joint_names"] + contract["wheel_joint_names"],
        "optimizer_fresh": not bool(optimizer.state),
        "learned_noise_loaded": "log_std" in source_state or "std" in source_state,
        "clearance_stage": 0, "yaw_stage": 0,
        "yaw_range": list(task_env.command_manager.get_term("yaw_rate_cmd").cfg.yaw_rate_range),
        "clearance_target": task_env.reward_manager.get_term_cfg("lift_clearance").params["target_clearance"],
        "dr_scale": float(term.params["dr_scale_levels"][0]),
    }
    print(f"[INFO] POS-Transfer initialization: {report}")
    return report
