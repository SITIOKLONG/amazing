
from __future__ import annotations

import torch
import numpy as np
import math
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
from amazing.amazing.tasks.manager_based.locomotion.velocity.sensors import LiftMask
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_z_event(
    env: ManagerBasedRLEnv,
    event_command_name: str = "event",
    event_time_range: tuple = (0.3, 0.8),
    max_up_vel: float = 4.0,
    up_vel_coef: float = 20.0,
    down_vel_coef: float = 1.0,
    temperature: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    event_command = env.command_manager.get_command(event_command_name)
    event_time    = event_command[:, 1]
    asset: RigidObject = env.scene[asset_cfg.name]

    lin_vel = asset.data.root_lin_vel_w
    lin_vel_z = lin_vel[:, 2]
    lin_vel_mag = torch.norm(lin_vel, dim=1) + 1e-6

    z_axis = torch.tensor([0, 0, 1.0], device=lin_vel.device)
    alignment = torch.sum(lin_vel * z_axis, dim=1) / lin_vel_mag  # = cos(theta)

    alignment_reward = torch.abs(alignment)

    target_up_vel = max_up_vel
    # up_vel = torch.clamp(lin_vel_z, min=0, max=max_up_vel)
    # down_vel = torch.clamp(-lin_vel_z, min=-max_up_vel, max=max_up_vel)

    pre_jump = (event_time < event_time_range[0]).float()

    descent_vel = torch.clamp(-lin_vel_z, min=0.0)
    max_descent_vel = 1.0

    penalty_coef = 1.0
    descent_penalty = torch.clamp(descent_vel - max_descent_vel, min=0.0) * penalty_coef

    jump_phase = torch.logical_and(event_time >= event_time_range[0], event_time <= event_time_range[1]).float()

    after_jump = torch.logical_and(event_time > event_time_range[1], event_time <= event_time_range[1] + 0.4).float()

    up_vel_reward   = torch.exp(-torch.abs((target_up_vel - lin_vel_z)/max_up_vel) * temperature)
    down_vel_reward = torch.exp(-torch.abs((-target_up_vel - lin_vel_z)/max_up_vel) * temperature)

    reward = up_vel_reward * up_vel_coef  * event_command[:, 0] * jump_phase  * alignment_reward
    reward += down_vel_reward * down_vel_coef * event_command[:, 0] * after_jump * alignment_reward
    reward -= descent_penalty * event_command[:, 0] * pre_jump

    return reward

def wheel_action_zero_event(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    event_command_name: str = "event",
    wheel_action_name: str = "wheel_vel",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    
    cmd = env.command_manager.get_command(command_name)         # [B, D]
    event_command = env.command_manager.get_command(event_command_name)  # [B, 2]
    
    wheel_action = env.action_manager.get_term(wheel_action_name).processed_actions
    wheel_action_l2 = torch.sum(torch.square(wheel_action), dim=1)  # [B]

    no_cmd_mask = torch.norm(cmd, dim=-1) < 1e-3

    return wheel_action_l2 * no_cmd_mask * event_command[:, 0]

def reward_push_ground_event(
    env: ManagerBasedRLEnv,
    event_command_name: str = "event",
    event_time_range: tuple = (0.3, 0.8),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:

    event_command = env.command_manager.get_command(event_command_name)
    event_time = event_command[:, 1]

    contact_sensor = env.scene.sensors[sensor_cfg.name]
    foot_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]

    z_axis = torch.tensor([0, 0, 1.0], device=foot_force.device).view(1, 1, 3)  # shape [1, 1, 3]

    force_mag = torch.norm(foot_force, dim=2) + 1e-6  # [B, N_foot]
    alignment = torch.sum(foot_force * z_axis, dim=2) / force_mag  # [B, N_foot]
    alignment = torch.abs(alignment)

    aligned_force = force_mag * alignment  # [B, N_foot]

    force_diff = torch.abs(aligned_force[:, 0] - aligned_force[:, 1])  # [B]

    total_force = aligned_force.sum(dim=1).clamp(max=300)  # [B]
    reward = total_force * torch.exp(-force_diff / 20)

    return reward * event_command[:, 0] * torch.logical_and(event_time >= event_time_range[0], event_time <= event_time_range[1])

def base_height_clearance_event(
    env: ManagerBasedRLEnv,
    event_command_name: str = "event",
    active_time_range: tuple = (0.35, 1.0),
    stand_height: float = 0.6129,
    target_height: float = 0.72,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base_link"),
) -> torch.Tensor:
    event_command = env.command_manager.get_command(event_command_name)
    event_time = event_command[:, 1]
    asset: RigidObject = env.scene[asset_cfg.name]

    base_z = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    height_progress = torch.clamp((base_z - stand_height) / max(target_height - stand_height, 1e-6), 0.0, 1.0)
    active = torch.logical_and(event_time >= active_time_range[0], event_time <= active_time_range[1])
    return height_progress * event_command[:, 0] * active


def base_height_target_event(
    env: ManagerBasedRLEnv,
    event_command_name: str = "event",
    active_time_range: tuple = (0.0, 0.3),
    target_height: float = 0.45,
    sigma: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base_link"),
) -> torch.Tensor:
    event_command = env.command_manager.get_command(event_command_name)
    event_time = event_command[:, 1]
    asset: RigidObject = env.scene[asset_cfg.name]

    base_z = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    active = torch.logical_and(event_time >= active_time_range[0], event_time <= active_time_range[1])
    return torch.exp(-torch.square(base_z - target_height) / sigma) * event_command[:, 0] * active


def leg_crouch_pose(
    env: ManagerBasedRLEnv,
    thigh_abs_target: float = 0.45,
    calf_abs_target: float = 0.65,
    abd_abs_target: float = 0.0,
    sigma: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]

    thigh = q[:, 0:2].abs()
    calf = q[:, 2:4].abs()
    abd = q[:, 4:6].abs()

    error = torch.sum(torch.square(thigh - thigh_abs_target), dim=1)
    error += torch.sum(torch.square(calf - calf_abs_target), dim=1)
    error += torch.sum(torch.square(abd - abd_abs_target), dim=1)
    return torch.exp(-error / sigma)


def leg_bend_progress(
    env: ManagerBasedRLEnv,
    thigh_target: float = 0.55,
    calf_target: float = 0.85,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids].abs()

    thigh_progress = torch.clamp(q[:, 0:2] / max(thigh_target, 1e-6), 0.0, 1.0).mean(dim=1)
    calf_progress = torch.clamp(q[:, 2:4] / max(calf_target, 1e-6), 0.0, 1.0).mean(dim=1)
    return 0.5 * (thigh_progress + calf_progress)


def leg_bend_progress_event(
    env: ManagerBasedRLEnv,
    event_command_name: str = "event",
    active_time_range: tuple = (0.0, 0.25),
    thigh_target: float = 0.55,
    calf_target: float = 0.85,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    event_command = env.command_manager.get_command(event_command_name)
    event_time = event_command[:, 1]
    reward = leg_bend_progress(env, thigh_target=thigh_target, calf_target=calf_target, asset_cfg=asset_cfg)
    active = torch.logical_and(event_time >= active_time_range[0], event_time <= active_time_range[1])
    return reward * event_command[:, 0] * active


def leg_extension_event(
    env: ManagerBasedRLEnv,
    event_command_name: str = "event",
    active_time_range: tuple = (0.25, 0.55),
    thigh_abs_target: float = 0.08,
    calf_abs_target: float = 0.08,
    abd_abs_target: float = 0.0,
    sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    event_command = env.command_manager.get_command(event_command_name)
    event_time = event_command[:, 1]
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids].abs()

    error = torch.sum(torch.square(q[:, 0:2] - thigh_abs_target), dim=1)
    error += torch.sum(torch.square(q[:, 2:4] - calf_abs_target), dim=1)
    error += torch.sum(torch.square(q[:, 4:6] - abd_abs_target), dim=1)
    active = torch.logical_and(event_time >= active_time_range[0], event_time <= active_time_range[1])
    return torch.exp(-error / sigma) * event_command[:, 0] * active


def abd_spread_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids]), dim=1)


def wheel_air_event(
    env: ManagerBasedRLEnv,
    event_command_name: str = "event",
    active_time_range: tuple = (0.4, 1.0),
    min_air_height: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    event_command = env.command_manager.get_command(event_command_name)
    event_time = event_command[:, 1]
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    wheel_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    wheel_height_progress = torch.clamp(wheel_z / max(min_air_height, 1e-6), 0.0, 1.0).mean(dim=1)
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    no_contact = (~in_contact).float().mean(dim=1)

    active = torch.logical_and(event_time >= active_time_range[0], event_time <= active_time_range[1])
    return (0.5 * wheel_height_progress + 0.5 * no_contact) * event_command[:, 0] * active


def feet_air_time_event(
    env: ManagerBasedRLEnv,
    event_command_name: str,
    threshold: float = 0.1,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    event_command = env.command_manager.get_command(event_command_name) 
    
    return (reward * event_command[:,0])
    
def feet_air_time_target_event(
    env: ManagerBasedRLEnv,
    event_command_name: str,
    target_air_time: float = 0.5,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Reward the robot for matching foot air time to a target value.

    This reward encourages the robot to lift its feet off the ground for a specified duration (`target_air_time`),
    penalizing deviations using squared error (L2 loss).
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]  # shape: (N, num_feet)

    event_command = env.command_manager.get_command(event_command_name)
    event_time = event_command[:, 1]

    air_time_error = last_air_time - target_air_time
    reward = torch.sum(torch.square(air_time_error), dim=1)  # shape: (N,)

    return reward * event_command[:, 0] * torch.logical_and(event_time >= 0.3, event_time <= 0.8)

class RewardCompleteEvent(ManagerTermBase):

    def __init__(self,
                 cfg: RewardTermCfg,
                 env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor = env.scene.sensors[cfg.params["contact_sensor_cfg"].name]

        self.base_id = cfg.params["asset_cfg"].body_ids[0]
        self.foot_ids = cfg.params["foot_asset_cfg"].body_ids
        self.contact_foot_ids = cfg.params["contact_sensor_cfg"].body_ids

        self.jump_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.jump_land_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        self.base_height_threshold = cfg.params["base_height_threshold"]
        self.foot_height_threshold = cfg.params["foot_height_threshold"]
        self.landing_time = cfg.params["landing_time"]

    def __call__(self,
                 env: ManagerBasedRLEnv,
                 event_command_name: str,
                 asset_cfg: SceneEntityCfg,
                 foot_asset_cfg: SceneEntityCfg,
                 contact_sensor_cfg: SceneEntityCfg,
                 jump_time_range: tuple,
                 base_height_threshold: float,
                 foot_height_threshold: float,
                 landing_time: float):

        # === 1. Command and time ===
        command = env.command_manager.get_command(event_command_name)
        command_flag = command[:, 0]      # 0 or 1.0
        event_time = command[:, 1]

        # === 2. Observation ===
        pos = self.asset.data.body_pos_w
        base_z = pos[:, self.base_id, 2]
        foot_z = pos[:, self.foot_ids, 2]     # shape: (N, 2)

        # === 3. Sensor contact ===
        contact_time = self.contact_sensor.data.current_contact_time[:, self.contact_foot_ids]
        in_contact = contact_time > 0
        wheels_contact = in_contact.all(dim=1)

        # === 4. Boolean masks ===
        jump_window = (event_time > jump_time_range[0]) & (event_time < jump_time_range[1])
        jumped_high_enough = (base_z > base_height_threshold) & torch.all(foot_z > foot_height_threshold, dim=1)
        jump_success = jump_window & jumped_high_enough

        landing_success = (event_time >= landing_time) & wheels_contact
        command_active = command_flag > 0.5
        is_dead = env.reset_buf > 0

        # === 5. Flag updates ===
        self.jump_flag |= jump_success
        self.jump_land_flag |= self.jump_flag & landing_success

        # === 6. Reward computation ===
        reward = torch.zeros_like(event_time)

        jump_success_mask = self.jump_flag & ~is_dead & command_active
        jump_land_success_mask = self.jump_land_flag & ~is_dead & command_active
        # death_mask       = self.jump_land_flag &  is_dead & command_active
        # total_fail_mask  = ~self.jump_land_flag & command_active

        reward[jump_success_mask] += 1.0
        reward[jump_land_success_mask] += 1.0

        # === 7. Reset ===
        reset_mask = is_dead | (event_time < 0.02)
        self.jump_flag[reset_mask] = False
        self.jump_land_flag[reset_mask] = False

        return reward

def base_height_adaptive_l2_event(
    env: ManagerBasedRLEnv,
    target_height: float,
    event_command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    event_cmd = env.command_manager.get_command(event_command_name)  # shape: (num_envs, 2)

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_link_pos_w[:, 2] - adjusted_target_height) * (1 - event_cmd[:,0])

def base_target_height_adaptive_l2_event(
    env: ManagerBasedRLEnv,
    target_height: float,
    event_command_name: str,
    event_target_height: float = 0.56288,
    active_time_range: tuple = (0.8, 1.2),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """
    Penalize deviation from adaptive base height.
    
    - Default: penalize from `target_height`
    - If event is active and time_elapsed is in [0.8, 1.2], penalize from `event_target_height`
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    event_cmd = env.command_manager.get_command(event_command_name)  # shape: (num_envs, 2)
    event_active = event_cmd[:, 0] == 1.0
    time_elapsed = event_cmd[:, 1]
    in_event_window = (time_elapsed >= active_time_range[0]) & (time_elapsed <= active_time_range[1])
    use_event_target = event_active & in_event_window

    # Base height target selection
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        terrain_adjustment = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
        default_target = target_height + terrain_adjustment
        event_target = event_target_height + terrain_adjustment
    else:
        default_target = target_height
        event_target = event_target_height

    # Select final height target
    final_target = torch.where(use_event_target, event_target, default_target)

    # Compute L2 penalty on height deviation
    current_height = asset.data.root_link_pos_w[:, 2]
    return torch.square(current_height - final_target)


def over_height(
    env: ManagerBasedRLEnv,
    event_command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_height :float = 0.65):

    asset: RigidObject = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:,2]
    penalty = torch.where(current_height > max_height, torch.ones_like(current_height), torch.zeros_like(current_height))
    event_command = env.command_manager.get_command(event_command_name)

    return penalty * event_command[:,0]
