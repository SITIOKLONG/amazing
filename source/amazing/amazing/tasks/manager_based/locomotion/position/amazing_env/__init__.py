# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    agents,
    flat_env,
    rough_env,
)

##
# Register Gym environments.
##


#########################################CoRL###################################################
################################################################################################
gym.register(
    id="Isaac-Position-Flat-Amazing-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.AmazingFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatPPORunnerCfg_Position,
    },
)

gym.register(
    id="Isaac-Position-Flat-Amazing-v1-ppo-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.AmazingFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatPPORunnerCfg_Position,
    },
)


gym.register(
    id="Isaac-Position-Rough-Amazing-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.stair_env_cfg.AmazingRoughEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingRoughPPORunnerCfg_Position,
    },
)

gym.register(
    id="Isaac-Position-Rough-Amazing-v1-ppo-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.stair_env_cfg.AmazingRoughEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingRoughPPORunnerCfg_Position,
    },
)
