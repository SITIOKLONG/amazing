# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from scripts.co_rl.core.wrapper.rl_cfg import CoRlPolicyRunnerCfg, CoRlPpoActorCriticCfg, CoRlPpoAlgorithmCfg


@configclass
class CoRlPPORunnerCfg(CoRlPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 1500
    save_interval = 100
    experiment_name = "amazing_velocity"
    experiment_description = "co-rl ppo"
    empirical_normalization = False

    policy = CoRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = CoRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# Backward-compatible aliases used by existing task registrations.
AmazingFlatPPORunnerCfg_Stand_Drive = CoRlPPORunnerCfg
AmazingFlatPPORunnerCfg_Track_Z = CoRlPPORunnerCfg
AmazingFlatPPORunnerCfg_Track_RP = CoRlPPORunnerCfg
AmazingFlatPPORunnerCfg_Track_YK = CoRlPPORunnerCfg
AmazingFlatPPORunnerCfg_Track_JUMP = CoRlPPORunnerCfg
AmazingRoughPPORunnerCfg_Stand_Drive = CoRlPPORunnerCfg
AmazingFlatSRMPPORunnerCfg_Stand_Drive = CoRlPPORunnerCfg
AmazingFlatSRMPPORunnerCfg_Track_Z = CoRlPPORunnerCfg
AmazingFlatSRMPPORunnerCfg_Track_YK = CoRlPPORunnerCfg
AmazingFlatSACRunnerCfg_Stand_Drive = CoRlPPORunnerCfg
AmazingFlatTQCRunnerCfg_Stand_Drive = CoRlPPORunnerCfg