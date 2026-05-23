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
    id="Isaac-Velocity-Flat-Amazing-v0-ppo",
    entry_point="amazing.amazing.tasks.manager_based.locomotion.velocity.amazing_env.flat_env.stand_drive.flat_env_stand_drive_env:AmazingFlatStandDriveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.AmazingFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Amazing-Play-v0-ppo",
    entry_point="amazing.amazing.tasks.manager_based.locomotion.velocity.amazing_env.flat_env.stand_drive.flat_env_stand_drive_env:AmazingFlatStandDriveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.AmazingFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-TrackZ-Flat-Amazing-v0-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_track_z_cfg.AmazingFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatPPORunnerCfg_Track_Z,
    },
)

gym.register(
    id="Isaac-TrackZ-Flat-Amazing-Play-v0-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_track_z_cfg.AmazingFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatPPORunnerCfg_Track_Z,
    },
)

gym.register(
    id="Isaac-TrackRP-Flat-Amazing-v0-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_track_rp_cfg.AmazingFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatPPORunnerCfg_Track_RP,
    },
)

gym.register(
    id="Isaac-TrackRP-Flat-Amazing-Play-v0-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_track_rp_cfg.AmazingFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatPPORunnerCfg_Track_RP,
    },
)

gym.register(
    id="Isaac-TrackYK-Flat-Amazing-v0-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_track_yk_cfg.AmazingFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatPPORunnerCfg_Track_YK,
    },
)

gym.register(
    id="Isaac-TrackYK-Flat-Amazing-Play-v0-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_track_yk_cfg.AmazingFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatPPORunnerCfg_Track_YK,
    },
)

gym.register(
    id="Isaac-TrackJUMP-Flat-Amazing-v0-ppo",
    entry_point="amazing.amazing.tasks.manager_based.locomotion.velocity.amazing_env.flat_env.track_jump.flat_env_track_jump_env:AmazingFlatTrackJumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_track_jump_cfg.AmazingFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatPPORunnerCfg_Track_JUMP,
    },
)

gym.register(
    id="Isaac-TrackJUMP-Flat-Amazing-Play-v0-ppo",
    entry_point="amazing.amazing.tasks.manager_based.locomotion.velocity.amazing_env.flat_env.track_jump.flat_env_track_jump_env:AmazingFlatTrackJumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_track_jump_cfg.AmazingFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatPPORunnerCfg_Track_JUMP,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Amazing-v0-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_cfg.AmazingRoughEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingRoughPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Amazing-Play-v0-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_cfg.AmazingRoughEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingRoughPPORunnerCfg_Stand_Drive,
    },
)
##########################################SRM###################################################
gym.register(
    id="Isaac-Velocity-Flat-Amazing-v0-srmppo",
    entry_point="amazing.amazing.tasks.manager_based.locomotion.velocity.amazing_env.flat_env.stand_drive.flat_env_stand_drive_env:AmazingFlatStandDriveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.AmazingFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatSRMPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Amazing-v0-srmppo-Play",
    entry_point="amazing.amazing.tasks.manager_based.locomotion.velocity.amazing_env.flat_env.stand_drive.flat_env_stand_drive_env:AmazingFlatStandDriveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.AmazingFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatSRMPPORunnerCfg_Stand_Drive,
    },
)
gym.register(
    id="Isaac-TrackZ-Flat-Amazing-v0-srmppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_track_z_cfg.AmazingFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatSRMPPORunnerCfg_Track_Z,
    },
)

gym.register(
    id="Isaac-TrackZ-Flat-Amazing-v0-srmppo-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_track_z_cfg.AmazingFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatSRMPPORunnerCfg_Track_Z,
    },
)
gym.register(
    id="Isaac-TrackYK-Flat-Amazing-v0-srmppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_track_yk_cfg.AmazingFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatSRMPPORunnerCfg_Track_YK,
    },
)

gym.register(
    id="Isaac-TrackYK-Flat-Amazing-Play-v0-srmppo-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_track_yk_cfg.AmazingFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatSRMPPORunnerCfg_Track_YK,
    },
)


#########################################CoRL###################################################
gym.register(
    id="Isaac-Velocity-Flat-Amazing-v3-sac",
    entry_point="amazing.amazing.tasks.manager_based.locomotion.velocity.amazing_env.flat_env.stand_drive.flat_env_stand_drive_env:AmazingFlatStandDriveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.AmazingFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatSACRunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Amazing-v3-tqc",
    entry_point="amazing.amazing.tasks.manager_based.locomotion.velocity.amazing_env.flat_env.stand_drive.flat_env_stand_drive_env:AmazingFlatStandDriveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.AmazingFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.AmazingFlatTQCRunnerCfg_Stand_Drive,
    },
)
