# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Initial RSL-RL PPO settings for M1 two-wheel balance."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class VQRTwoWheelBalancePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 250
    experiment_name = "vqr_two_wheel_balance"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
