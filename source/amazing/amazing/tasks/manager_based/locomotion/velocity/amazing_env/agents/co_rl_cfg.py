# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ported from:
#   https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/main/lab/flamingo/tasks/
#       manager_based/locomotion/velocity/flamingo_env/agents/co_rl_cfg.py

from isaaclab.utils import configclass

from scripts.co_rl.core.wrapper.rl_cfg import (
    CoRlPolicyRunnerCfg,
    CoRlPpoActorCriticCfg,
    CoRlPpoAlgorithmCfg,
    CoRlSrmPpoAlgorithmCfg,
)


######################################## [ PPO CONFIG ] ########################################


@configclass
class AmazingPPORunnerCfg(CoRlPolicyRunnerCfg):
    num_steps_per_env = 24
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


@configclass
class AmazingFlatPPORunnerCfg_Stand_Drive(AmazingPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Amazing_Flat_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class AmazingFlatPPORunnerCfg_Track_Z(AmazingPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Amazing_Flat_Track_Z"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class AmazingFlatPPORunnerCfg_Track_RP(AmazingPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Amazing_Flat_Track_Roll_Pitch"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class AmazingFlatPPORunnerCfg_Track_YK(AmazingPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Amazing_Flat_Yuna_Kim"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class AmazingFlatPPORunnerCfg_Track_JUMP(AmazingPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Amazing_Flat_Jump"
        self.policy.init_noise_std = 0.3
        self.algorithm.learning_rate = 3.0e-4
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class AmazingRoughPPORunnerCfg_Stand_Drive(AmazingPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "Amazing_Rough_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


###############################################################################################
######################################## [ SRMPPO CONFIG ] ######################################


@configclass
class AmazingSRMPPORunnerCfg(CoRlPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 250
    experiment_name = "amazing_velocity"
    experiment_description = "co-rl srm-ppo"
    empirical_normalization = False
    policy = CoRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = CoRlSrmPpoAlgorithmCfg(
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
        srm_net="gru",
        srm_input_dim=32,
        cmd_dim=4,
        srm_hidden_dim=256,
        srm_output_dim=5,
        srm_num_layers=1,
        srm_r_loss_coef=1.0,
        srm_rc_loss_coef=1.0e-1,
        use_acaps=False,
        acaps_lambda_t_coef=1.0e-1,
        acaps_lambda_s_coef=1.0e-2,
    )


@configclass
class AmazingFlatSRMPPORunnerCfg_Stand_Drive(AmazingSRMPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Amazing_Flat_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class AmazingFlatSRMPPORunnerCfg_Track_Z(AmazingSRMPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Amazing_Flat_Track_Z"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class AmazingFlatSRMPPORunnerCfg_Track_YK(AmazingSRMPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Amazing_Flat_Yuna_Kim"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.srm_input_dim = 34
        self.algorithm.cmd_dim = 6


###############################################################################################
######################################## [ SAC CONFIG ] ########################################


@configclass
class AmazingSACRunnerCfg(CoRlPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 200000
    save_interval = 200
    experiment_name = "amazing_velocity"
    experiment_description = "co-rl sac"
    empirical_normalization = False
    policy = CoRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = {"class_name": "SAC"}


@configclass
class AmazingFlatSACRunnerCfg_Stand_Drive(AmazingSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Amazing_Flat_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class AmazingFlatSACRunnerCfg_Track_Z(AmazingSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Amazing_Flat_Track_Z"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class AmazingRoughSACRunnerCfg_Stand_Drive(AmazingSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Amazing_Rough_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class AmazingRoughSACRunnerCfg_Track_Z(AmazingSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Amazing_Rough_Track_Z"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class AmazingRoughSACRunnerCfg_Stand_Walk(AmazingSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Amazing_Rough_Stand_Walk"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


###############################################################################################
######################################## [ TQC CONFIG ] ########################################


@configclass
class AmazingTQCRunnerCfg(CoRlPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 200000
    save_interval = 200
    experiment_name = "amazing_velocity"
    experiment_description = "co-rl tqc"
    empirical_normalization = False
    policy = CoRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = {"class_name": "TQC"}


@configclass
class AmazingFlatTQCRunnerCfg_Stand_Drive(AmazingTQCRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Amazing_Flat_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 512, 512]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class AmazingFlatTQCRunnerCfg_Track_Z(AmazingTQCRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Amazing_Flat_Track_Z"
        self.policy.actor_hidden_dims = [512, 512, 512]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class AmazingRoughTQCRunnerCfg_Stand_Drive(AmazingTQCRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Amazing_Rough_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 512, 512]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class AmazingRoughTQCRunnerCfg_Track_Z(AmazingTQCRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Amazing_Rough_Track_Z"
        self.policy.actor_hidden_dims = [512, 512, 512]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class AmazingRoughTQCRunnerCfg_Stand_Walk(AmazingTQCRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200000
        self.experiment_name = "Amazing_Rough_Stand_Walk"
        self.policy.actor_hidden_dims = [512, 512, 512]
        self.policy.critic_hidden_dims = [512, 256, 128]
