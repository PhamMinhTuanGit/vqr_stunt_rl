"""Policy-only adaptation uses the complete local checkpoint and a real fresh PPO runner."""

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/reinforcement_learning/rsl_rl"
CHECKPOINT = ROOT / "logs/rsl_rl/vqr_wheel_yaw_flat_pos/2026-10-01_10-25-33/model_21000.pt"
pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(), reason="Local POS checkpoint required")


@pytest.fixture
def loader(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("yaw_pos_adaptation")


@pytest.fixture
def configs(loader):
    result = []
    for name in ("env", "agent"):
        with (CHECKPOINT.parent / "params" / f"{name}.yaml").open() as stream:
            result.append(yaml.load(stream, Loader=loader._ConfigLoader))
    return result


@pytest.fixture
def real_runner(configs, tmp_path):
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    env_config, agent_config = configs
    agent_config = copy.deepcopy(agent_config)
    agent_config["device"] = "cpu"
    obs = TensorDict({"policy": torch.randn(32, 55), "critic": torch.randn(32, 83)}, batch_size=[32])
    vec = NS(num_envs=32, num_actions=16, device="cpu", get_observations=lambda: obs)
    runner = OnPolicyRunner(vec, copy.deepcopy(agent_config), log_dir=str(tmp_path / "new_run"), device="cpu")
    task = NS(common_step_counter=0, _yaw_task_curriculum_stage=3, _yaw_task_curriculum_yaw_stage=3)
    def curriculum(env, ids, **kwargs):
        env._yaw_task_curriculum_stage = env._yaw_task_curriculum_yaw_stage = 0
    task.cfg = NS(to_dict=lambda: env_config, curriculum=NS(task_levels=NS(func=curriculum, params={})))
    task.reward_manager = NS(get_term_cfg=lambda _: NS(weight=-.25, params={"minimum_separation": .45, "separation_scale": .05}))
    task.reset = lambda: None
    yield runner, task, agent_config, obs
    torch.set_num_threads(old_threads)


def test_load_all_weights_without_optimizer_iteration_or_curriculum_restore(loader, real_runner, monkeypatch):
    runner, task, agent, obs = real_runner
    original_optimizer = runner.alg.optimizer
    monkeypatch.setattr(runner, "load", lambda *a, **k: pytest.fail("runner.load restores old run state"))
    monkeypatch.setattr(original_optimizer, "load_state_dict", lambda *a, **k: pytest.fail("old optimizer restored"))
    runner.current_learning_iteration = 99
    before = loader.checkpoint_sha256(CHECKPOINT)
    report = loader.load_pos_policy(runner, task, CHECKPOINT, agent)
    assert report["source_iteration"] == 21000
    assert all(report["weights_after_load"].values())
    assert report["model_state_tensors"] == 17
    assert runner.alg.optimizer is original_optimizer and not original_optimizer.state
    assert report["optimizer_before_update"]["max_step"] == 0
    assert report["optimizer_before_update"]["state_entries"] == 0
    assert runner.current_learning_iteration == runner.tot_timesteps == 0
    assert task._yaw_task_curriculum_stage == task._yaw_task_curriculum_yaw_stage == 0
    assert report["support_y_collapse_weight"] == -.25
    assert loader.checkpoint_sha256(CHECKPOINT) == before
    assert runner.alg.policy.act_inference(obs).shape == (32, 16)
    assert runner.alg.policy.evaluate(obs).shape == (32, 1)


@pytest.mark.parametrize("change,expected", [("action_order", "actions"), ("observation_order", "policy_terms"),
                                            ("critic_scale", "critic_terms"), ("ppo", "algorithm"), ("network", "policy")])
def test_reject_contract_drift(loader, configs, change, expected):
    env, agent = copy.deepcopy(configs)
    if change == "action_order":
        env["actions"]["joint_vel"]["joint_names"].reverse()
    elif change == "observation_order":
        env["observations"]["policy"]["base_ang_vel"] = env["observations"]["policy"].pop("base_ang_vel")
    elif change == "critic_scale":
        env["observations"]["critic"]["joint_vel"]["scale"] = .123
    elif change == "ppo":
        agent["algorithm"]["learning_rate"] *= 2
    else:
        agent["policy"]["actor_hidden_dims"] = [256, 256, 128]
    with pytest.raises(RuntimeError, match=expected):
        loader.audit_checkpoint_contract(CHECKPOINT, env, agent)


@pytest.mark.parametrize("corruption", ["missing_critic", "shape", "nonfinite"])
def test_strict_load_rejects_bad_weights_before_mutation(loader, real_runner, tmp_path, corruption):
    import shutil
    runner, task, agent, _ = real_runner
    source = tmp_path / "corrupted_source"
    (source / "params").mkdir(parents=True)
    for name in ("env", "agent"):
        shutil.copyfile(CHECKPOINT.parent / "params" / f"{name}.yaml", source / "params" / f"{name}.yaml")
    saved = torch.load(CHECKPOINT, weights_only=True, map_location="cpu")
    if corruption == "missing_critic":
        del saved["model_state_dict"]["critic.0.bias"]
    elif corruption == "shape":
        saved["model_state_dict"]["actor.0.weight"] = saved["model_state_dict"]["actor.0.weight"][:, :-1]
    else:
        saved["model_state_dict"]["log_std"][0] = float("nan")
    torch.save(saved, source / "bad.pt")
    before = copy.deepcopy(runner.alg.policy.state_dict())
    with pytest.raises(RuntimeError, match="mismatch|non-finite"):
        loader.load_pos_policy(runner, task, source / "bad.pt", agent)
    for key, value in runner.alg.policy.state_dict().items():
        assert torch.equal(value, before[key])
    assert not runner.alg.optimizer.state


def test_used_optimizer_and_source_directory_are_rejected(loader, real_runner):
    runner, task, agent, _ = real_runner
    parameter = next(iter(runner.alg.policy.parameters()))
    runner.alg.optimizer.state[parameter] = {"step": torch.tensor(20.)}
    with pytest.raises(RuntimeError, match="freshly constructed optimizer"):
        loader.load_pos_policy(runner, task, CHECKPOINT, agent)
    runner.alg.optimizer.state.clear()
    runner.log_dir = str(CHECKPOINT.parent)
    with pytest.raises(RuntimeError, match="source checkpoint is protected"):
        loader.load_pos_policy(runner, task, CHECKPOINT, agent)


def test_first_update_audit_counts_actual_ppo_and_preserves_source(loader, real_runner):
    runner, task, agent, obs = real_runner
    report = loader.load_pos_policy(runner, task, CHECKPOINT, agent)
    with torch.inference_mode():
        for _ in range(agent["num_steps_per_env"]):
            runner.alg.act(obs)
            runner.alg.process_env_step(obs, torch.randn(32), torch.zeros(32), {})
        runner.alg.compute_returns(obs)
    runner.alg.update()
    loader.finish_adaptation_audit(runner, report)
    assert report["ppo_updates_completed"] == 1
    assert all(report["weights_before_first_update"].values())
    assert report["optimizer_before_update"]["state_entries"] == 0
    assert report["optimizer_before_update"]["max_step"] == 0
    assert report["optimizer_after_first_update"]["state_entries"] == 17
    assert report["optimizer_after_first_update"]["steps"] == [20]
    assert report["policy_finite_after_first_update"] and report["source_checkpoint_unchanged"]
    saved_report = json.loads((Path(runner.log_dir) / "adaptation_preflight.json").read_text())
    assert saved_report["ppo_updates_completed"] == 1
    assert not report["weights_after_first_update"]["all_state_exact"]
