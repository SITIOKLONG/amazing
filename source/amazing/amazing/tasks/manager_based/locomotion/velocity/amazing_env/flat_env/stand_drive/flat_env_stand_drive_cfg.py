# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

import amazing.amazing.tasks.manager_based.locomotion.velocity.mdp as mdp
from amazing.amazing.tasks.manager_based.locomotion.velocity.amazing_env.velocity_env_cfg import (
    LocomotionVelocityFlatEnvCfg,
    CurriculumCfg,
    EventCfg,
)

from amazing.assets.amazing import AmazingCfg  # isort: skip


# -----------------------------------------------------------------------------
# HoST-style stand-up curriculum
#   Reference: https://github.com/InternRobotics/HoST
#   Idea: apply an upward external force ("helper hand") on the torso/base while
#         the policy is learning to stand from a fallen pose, then decay that
#         assistance to zero in 3 stages so the policy ends up standing on its own.
#
# Time accounting (matches AmazingFlatPPORunnerCfg_Stand_Drive in agents/co_rl_cfg.py):
#   max_iterations          = 5000
#   num_steps_per_env       = 24
#   env.common_step_counter increments once per env.step(), so:
#       1 PPO iter   == 24 env steps
#       5000 iters   == 120_000 env steps
# -----------------------------------------------------------------------------

# Pull-force magnitude schedule. The base 0-th stage value is whatever we set as
# the initial param on the pull_force_up event (strong). The curriculum then
# overrides it at the following phase boundaries.
_PULL_STAGE2_AT = 24_000   # ~iter 1_000  : reduce to medium pull
_PULL_STAGE3_AT = 60_000   # ~iter 2_500  : reduce to light pull
_PULL_STAGE4_AT = 96_000   # ~iter 4_000  : turn assistance fully off

_PULL_RANGE_STAGE2 = (15.0, 30.0)
_PULL_RANGE_STAGE3 = (5.0, 15.0)
_PULL_RANGE_STAGE4 = (0.0, 0.0)


def _set_force_after_steps(env, env_ids, data, value, num_steps):
    """``modify_fn`` for ``mdp.modify_term_cfg`` — overwrite force_range past ``num_steps``."""
    if env.common_step_counter > num_steps:
        return value
    return mdp.modify_term_cfg.NO_CHANGE


# @configclass
# class AmazingStandUpEventCfg(EventCfg):
#     """Inherits all events from velocity_env_cfg.EventCfg and adds the HoST pull-up force."""

#     pull_force_up = EventTerm(
#         func=mdp.apply_pull_force_z,
#         mode="reset",
#         params={
#             # initial (stage-1) range; decayed by AmazingCurriculumCfg below.
#             # Robot mass is ~6.25 kg, so this starts near HoST's 60% gravity target.
#             "force_range": (0.0, 0.0),
#             "orientation_gate_threshold": 0.55,
#             "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
#         },
#     )


@configclass
class AmazingCurriculumCfg(CurriculumCfg):

    modify_base_velocity_range = CurrTerm(
        func=mdp.modify_base_velocity_range,
        params={
            "term_name": "base_velocity",
            "mod_range": {"lin_vel_x": (-2.0, 2.0), "ang_vel_z": (-3.14, 3.14)},
            "num_steps": 25000,
        },
    )

    # ---- HoST pull-force decay (3 stage cuts down to zero) ----
    # decay_pull_force_stage2 = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.pull_force_up.params.force_range",
    #         "modify_fn": _set_force_after_steps,
    #         "modify_params": {"value": _PULL_RANGE_STAGE2, "num_steps": _PULL_STAGE2_AT},
    #     },
    # )
    # decay_pull_force_stage3 = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.pull_force_up.params.force_range",
    #         "modify_fn": _set_force_after_steps,
    #         "modify_params": {"value": _PULL_RANGE_STAGE3, "num_steps": _PULL_STAGE3_AT},
    #     },
    # )
    # decay_pull_force_stage4 = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.pull_force_up.params.force_range",
    #         "modify_fn": _set_force_after_steps,
    #         "modify_params": {"value": _PULL_RANGE_STAGE4, "num_steps": _PULL_STAGE4_AT},
    #     },
    # )


@configclass
class AmazingRewardsCfg():
    # HoST reward groups flattened into IsaacLab term weights.
    # Reference groups: task/regu/style/target with weights [2.5, 0.1, 1, 1].
    # The names keep the group prefix so TensorBoard still shows the HoST structure.

    # -- task group (2.5x): solve the stand-up objective first
    task_orientation = RewTerm(
        func=mdp.host_upright_exp,
        weight=2.5,
        params={"sigma": 1.0, "asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )
    task_base_height = RewTerm(
        func=mdp.host_base_height_exp,
        weight=2.5,
        params={
            "target_height": 0.70,
            "sigma": 0.25,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    # Keep a small drive objective so the final stand-up policy can still move.
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_link_exp,
        weight=0.25,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-800.0)

    # -- regularization group (0.1x): keep this light, as in HoST
    regu_lin_vel_z = RewTerm(func=mdp.lin_vel_z_link_l2, weight=-0.1)
    regu_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.002)
    regu_dof_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-7)
    regu_dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-8)
    regu_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    regu_smoothness = RewTerm(func=mdp.action_smoothness_hard, weight=-0.001)
    regu_dof_vel = RewTerm(
        func=mdp.joint_velocity_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    regu_dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf", ".*_thigh", ".*_abd", "arm_.*"])},
    )
    regu_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # -- style group (1x): discourage HoST-like ugly contacts/postures
    style_joint_deviation = RewTerm(
        func=mdp.joint_deviation_zero_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf", ".*_thigh", ".*_abd"])},
    )
    style_arm_deviation = RewTerm(
        func=mdp.joint_deviation_zero_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*")},
    )
    style_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FR1_1", "FL1_1", "FR2_1", "FL2_1", "SR_1", "SL_1",
                            "base_link", "T1_1", "T2_1", "T3_1", "T4_1", "T5_1", "T6_1"],
            ),
            "threshold": 1.0,
        },
    )
    style_joint_align = RewTerm(
        func=mdp.joint_align_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh", ".*_calf"])},
    )

    # -- target group (1x): HoST post-task standing targets
    # The "stillness" + arm-pose rewards are gated by orientation so the policy
    # cannot farm them by lying still. They only fire once the base z-axis is
    # roughly upright (projected_gravity_b[:, 2] < -0.7). Mirrors HoST's
    # post_task=True flag.
    target_ang_vel_xy = RewTerm(
        func=mdp.host_low_base_ang_vel_xy_exp,
        weight=10.0,
        params={"sigma": 0.25, "orientation_gate_threshold": 0.7},
    )
    target_lin_vel_xy = RewTerm(
        func=mdp.host_low_base_lin_vel_xy_exp,
        weight=10.0,
        params={"sigma": 0.25, "orientation_gate_threshold": 0.7},
    )
    target_orientation = RewTerm(
        func=mdp.host_upright_exp,
        weight=10.0,
        params={"sigma": 1.0, "asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )
    target_base_height = RewTerm(
        func=mdp.host_base_height_exp,
        weight=10.0,
        params={
            "target_height": 0.70,
            "sigma": 0.25,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )
    target_upper_dof_pos = RewTerm(
        func=mdp.host_target_joint_pos_exp,
        weight=10.0,
        params={
            "sigma": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*"),
            "orientation_gate_threshold": 0.7,
        },
    )


@configclass
class AmazingFlatEnvCfg(LocomotionVelocityFlatEnvCfg):

    # No-control window after every reset (lets the robot fall before the policy acts).
    free_fall_seconds: float = 2.0
    rewards: AmazingRewardsCfg = AmazingRewardsCfg()
    # events: AmazingStandUpEventCfg = AmazingStandUpEventCfg()
    curriculum: AmazingCurriculumCfg = AmazingCurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 5.0
        # scene
        self.scene.robot = AmazingCfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Terrain curriculum
        self.curriculum.terrain_levels = None

        #! ****************** Observations setup ****************** !#
        self.observations.none_stack_policy.roll_pitch_commands = None
        self.observations.none_stack_policy.event_commands = None
        self.observations.none_stack_critic.roll_pitch_commands = None
        self.observations.none_stack_critic.event_commands = None
        #! ********************************************************* !#

        # joint reset jitter
        # self.events.reset_robot_joints.params["position_range"] = (-1.0, 1.0)     # no need for HOST

        # Start each episode from a randomized attitude and let gravity create
        # the fall. HoST does not need initial root velocity for stand-up.
        # Spawn well above the ground so the random pitch/roll/yaw can place the
        # robot in any attitude without geometry clipping into the floor (which
        # PhysX otherwise resolves by ejecting the body upward at sim start).
        # Default AmazingCfg init z = 0.5 m, so this lands the spawn at 1.0-1.5 m.
        self.events.reset_base.params = {
            "pose_range": {
                # "z": (0.5, 1.0),
                # "roll": (-0.5, 0.5),
                "pitch": (-math.pi/2, -math.pi/2),
                # "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
        }

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

        # Standing-up training: torso/base contact must not terminate the episode.
        self.terminations.base_contact.params["sensor_cfg"].body_names = []


@configclass
class AmazingFlatEnvCfg_PLAY(AmazingFlatEnvCfg):

    free_fall_seconds: float = 2.0

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 5.0
        self.sim.render_interval = self.decimation
        self.debug_vis = True
        self.scene.robot = AmazingCfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # observations
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False

        # PLAY: disable the helper pull so we see the true policy
        # self.events.pull_force_up.params["force_range"] = (0.0, 0.0)

        # self.events.reset_robot_joints.params["position_range"] = (-1.0, 1.0)    # no need for HOST
        # Spawn well above the ground so the random pitch/roll/yaw can place the
        # robot in any attitude without geometry clipping into the floor (which
        # PhysX otherwise resolves by ejecting the body upward at sim start).
        # Default AmazingCfg init z = 0.5 m, so this lands the spawn at 1.0-1.5 m.
        self.events.reset_base.params = {
            "pose_range": {
                "z": (0.5, 1.0),
                "roll": (-0.5, 0.5),
                "pitch": (-math.pi, math.pi),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
        }

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)
        self.commands.base_velocity.ranges.heading = (-0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = []
