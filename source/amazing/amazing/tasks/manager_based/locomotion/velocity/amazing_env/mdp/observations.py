# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import torch
import math
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.utils.math import wrap_to_pi
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
from typing import Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


_OBS_DEBUG_EVERY_STEP = os.getenv("AMAZING_DEBUG_OBS_EVERY_STEP", "0") == "1"
_OBS_SANITIZE_WARNED_TERMS = set()


def _flatten_by_env(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    return tensor.reshape(tensor.shape[0], -1)


def _debug_observation_tensor(term_name: str, tensor: torch.Tensor) -> torch.Tensor:
    finite_mask = torch.isfinite(tensor)
    has_non_finite = not bool(finite_mask.all())
    should_print = has_non_finite or _OBS_DEBUG_EVERY_STEP
    if not should_print:
        return tensor

    finite_values = tensor[finite_mask]
    min_value = finite_values.min().item() if finite_values.numel() > 0 else float("nan")
    max_value = finite_values.max().item() if finite_values.numel() > 0 else float("nan")
    mean_value = finite_values.mean().item() if finite_values.numel() > 0 else float("nan")

    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    print(
        f"[OBS-DEBUG] term={term_name} shape={tuple(tensor.shape)} "
        f"nan_count={nan_count} inf_count={inf_count} min={min_value:.6f} "
        f"max={max_value:.6f} mean={mean_value:.6f}"
    )

    if has_non_finite:
        per_env = _flatten_by_env(tensor)
        bad_env_ids = torch.nonzero(~torch.isfinite(per_env).all(dim=1), as_tuple=False).squeeze(-1)
        if bad_env_ids.numel() > 0:
            first_bad_env_id = int(bad_env_ids[0].item())
            preview = per_env[first_bad_env_id, :16].detach().cpu().tolist()
            print(
                f"[OBS-DEBUG] term={term_name} first_bad_env={first_bad_env_id} "
                f"preview_first_16={preview}"
            )
        raise RuntimeError(f"[OBS-DEBUG] Non-finite value detected in observation term '{term_name}'.")

    return tensor


def _sanitize_non_finite(term_name: str, tensor: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all()):
        return tensor

    if term_name not in _OBS_SANITIZE_WARNED_TERMS:
        per_env = _flatten_by_env(tensor)
        bad_env_ids = torch.nonzero(~torch.isfinite(per_env).all(dim=1), as_tuple=False).squeeze(-1)
        bad_env_list = bad_env_ids.detach().cpu().tolist()
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
        print(
            f"[OBS-WARN] term={term_name} has non-finite values "
            f"(nan={nan_count}, inf={inf_count}), bad_envs={bad_env_list}. "
            f"Replacing with {fill_value}."
        )
        _OBS_SANITIZE_WARNED_TERMS.add(term_name)

    return torch.nan_to_num(tensor, nan=fill_value, posinf=fill_value, neginf=fill_value)


def base_lin_vel_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b

def base_lin_vel_x_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 0].unsqueeze(-1)

def base_lin_vel_y_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 1].unsqueeze(-1)

def base_lin_vel_z_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 1].unsqueeze(-1)

def base_ang_vel_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    tensor = asset.data.root_link_ang_vel_b
    if not bool(torch.isfinite(tensor).all()):
        per_env = _flatten_by_env(tensor)
        bad_env_ids = torch.nonzero(~torch.isfinite(per_env).all(dim=1), as_tuple=False).squeeze(-1)
        if bad_env_ids.numel() > 0:
            first_bad = int(bad_env_ids[0].item())
            try:
                print(f"[OBS-DEBUG-CTX] base_ang_vel_link first_bad_env={first_bad} asset_name={asset_cfg.name}")
                try:
                    print(
                        "[OBS-DEBUG-CTX] asset.data.root_link_ang_vel_b[bad]",
                        asset.data.root_link_ang_vel_b[first_bad].detach().cpu().tolist(),
                    )
                except Exception as _e:
                    print("[OBS-DEBUG-CTX] failed to read asset.data.root_link_ang_vel_b:", _e)
                try:
                    print(
                        "[OBS-DEBUG-CTX] asset.data.root_link_lin_vel_b[bad]",
                        asset.data.root_link_lin_vel_b[first_bad].detach().cpu().tolist(),
                    )
                except Exception as _e:
                    print("[OBS-DEBUG-CTX] failed to read asset.data.root_link_lin_vel_b:", _e)
                try:
                    print(
                        "[OBS-DEBUG-CTX] asset.data.root_link_pos_w[bad]",
                        asset.data.root_link_pos_w[first_bad].detach().cpu().tolist(),
                    )
                except Exception as _e:
                    print("[OBS-DEBUG-CTX] failed to read asset.data.root_link_pos_w:", _e)
                try:
                    print(
                        "[OBS-DEBUG-CTX] asset.data.root_link_quat_w[bad]",
                        asset.data.root_link_quat_w[first_bad].detach().cpu().tolist(),
                    )
                except Exception as _e:
                    print("[OBS-DEBUG-CTX] failed to read asset.data.root_link_quat_w:", _e)
                try:
                    print(
                        "[OBS-DEBUG-CTX] asset.data.applied_torque[bad]",
                        asset.data.applied_torque[first_bad].detach().cpu().tolist(),
                    )
                except Exception as _e:
                    print("[OBS-DEBUG-CTX] failed to read asset.data.applied_torque:", _e)
            except Exception as e:
                print("[OBS-DEBUG-CTX] failed to extract base_ang_vel context:", e)

            # attempt recovery (write defaults) for bad envs and recompute
            try:
                recovered = _recover_envs_from_nan(env, asset_cfg, bad_env_ids)
                if len(recovered) > 0:
                    print(f"[OBS-DEBUG-RECOVER] attempted recovery for envs={recovered}; recomputing base_ang_vel")
                    tensor = asset.data.root_link_ang_vel_b
                else:
                    print("[OBS-DEBUG-RECOVER] no envs recovered for base_ang_vel_link")
            except Exception as e:
                print("[OBS-DEBUG-RECOVER] recovery attempt raised for base_ang_vel_link:", e)

    return _debug_observation_tensor("base_ang_vel_link", tensor)
        
def base_pos_z_rel_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        value = asset.data.root_link_pos_w[:, 2].unsqueeze(-1) - sensor.data.ray_hits_w[..., 2]
        return _sanitize_non_finite("base_pos_z_rel_link", value)
    else:
        value = asset.data.root_link_pos_w[:, 2].unsqueeze(-1)
        return _sanitize_non_finite("base_pos_z_rel_link", value)
    
def base_pos_z_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        value = asset.data.root_link_pos_w[:, 2].unsqueeze(-1) - sensor.data.ray_hits_w[..., 2]
        return _sanitize_non_finite("base_pos_z_rel", value)
    else:
        value = asset.data.root_link_pos_w[:, 2].unsqueeze(-1)
        return _sanitize_non_finite("base_pos_z_rel", value)


def _recover_envs_from_nan(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, bad_env_ids: torch.Tensor) -> list:
    """Attempt best-effort recovery for env ids by writing default joint/root states back into sim.

    Returns list of env ids that were successfully written.
    """
    asset = env.scene[asset_cfg.name]
    recovered = []
    # accept both tensors and python lists
    if isinstance(bad_env_ids, torch.Tensor):
        bad_list = bad_env_ids.cpu().tolist()
    else:
        bad_list = list(bad_env_ids)

    for eid in bad_list:
        try:
            eid = int(eid)
        except Exception:
            continue
        # determine a device for the asset data
        try:
            device = asset.device
        except Exception:
            device = env.device

        # try to write default joint state
        try:
            default_jpos = asset.data.default_joint_pos[eid].unsqueeze(0).to(device)
            default_jvel = asset.data.default_joint_vel[eid].unsqueeze(0).to(device)
            # try writing with device-backed env_ids first
            try:
                asset.write_joint_state_to_sim(default_jpos, default_jvel, env_ids=torch.tensor([eid], device=device))
            except Exception:
                try:
                    asset.write_joint_state_to_sim(default_jpos.cpu(), default_jvel.cpu(), env_ids=torch.tensor([eid], device="cpu"))
                except Exception as e:
                    print("[OBS-DEBUG-RECOVER] write_joint_state_to_sim failed for env", eid, e)
                    continue
        except Exception as e:
            print("[OBS-DEBUG-RECOVER] failed to read/write default joint state for env", eid, e)
            continue

        # try to write default root pose + velocity
        try:
            default_root = asset.data.default_root_state[eid].unsqueeze(0).to(device)
            pose = torch.cat([default_root[:, :3], default_root[:, 3:7]], dim=-1)
            try:
                asset.write_root_link_pose_to_sim(pose, env_ids=torch.tensor([eid], device=device))
                asset.write_root_com_velocity_to_sim(default_root[:, 7:13], env_ids=torch.tensor([eid], device=device))
            except Exception:
                try:
                    asset.write_root_link_pose_to_sim(pose.cpu(), env_ids=torch.tensor([eid], device="cpu"))
                    asset.write_root_com_velocity_to_sim(default_root[:, 7:13].cpu(), env_ids=torch.tensor([eid], device="cpu"))
                except Exception as e:
                    print("[OBS-DEBUG-RECOVER] root write failed for env", eid, e)
                    # don't treat this as fatal if joints were recovered
        except Exception as e:
            print("[OBS-DEBUG-RECOVER] failed to read/write default root state for env", eid, e)

        recovered.append(eid)

    return recovered


def joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    tensor = isaaclab_mdp.joint_pos(env, asset_cfg)
    # extra context prints when NaNs are present to help root-cause diagnosis
    if not bool(torch.isfinite(tensor).all()):
        per_env = _flatten_by_env(tensor)
        bad_env_ids = torch.nonzero(~torch.isfinite(per_env).all(dim=1), as_tuple=False).squeeze(-1)
        if bad_env_ids.numel() > 0:
            first_bad = int(bad_env_ids[0].item())
            try:
                asset = env.scene[asset_cfg.name]
                print(f"[OBS-DEBUG-CTX] joint_pos first_bad_env={first_bad} asset_name={asset_cfg.name}")
                # raw joint arrays for the problematic env
                try:
                    print(
                        "[OBS-DEBUG-CTX] asset.data.joint_pos[bad]",
                        asset.data.joint_pos[first_bad].detach().cpu().tolist(),
                    )
                except Exception as _e:
                    print("[OBS-DEBUG-CTX] failed to read asset.data.joint_pos:", _e)
                try:
                    print(
                        "[OBS-DEBUG-CTX] asset.data.default_joint_pos[bad]",
                        asset.data.default_joint_pos[first_bad].detach().cpu().tolist(),
                    )
                except Exception:
                    print("[OBS-DEBUG-CTX] no default_joint_pos available or failed to read")
                try:
                    print("[OBS-DEBUG-CTX] asset_cfg.joint_ids=", getattr(asset_cfg, "joint_ids", None))
                except Exception:
                    pass
                # additional context: velocities, accelerations, torques, root pose, contact
                try:
                    print(
                        "[OBS-DEBUG-CTX] asset.data.joint_vel[bad]",
                        asset.data.joint_vel[first_bad].detach().cpu().tolist(),
                    )
                except Exception as _e:
                    print("[OBS-DEBUG-CTX] failed to read asset.data.joint_vel:", _e)
                try:
                    print(
                        "[OBS-DEBUG-CTX] asset.data.joint_acc[bad]",
                        asset.data.joint_acc[first_bad].detach().cpu().tolist(),
                    )
                except Exception as _e:
                    print("[OBS-DEBUG-CTX] failed to read asset.data.joint_acc:", _e)
                try:
                    print(
                        "[OBS-DEBUG-CTX] asset.data.applied_torque[bad]",
                        asset.data.applied_torque[first_bad].detach().cpu().tolist(),
                    )
                except Exception as _e:
                    print("[OBS-DEBUG-CTX] failed to read asset.data.applied_torque:", _e)
                try:
                    print(
                        "[OBS-DEBUG-CTX] asset.data.root_link_pos_w[bad]",
                        asset.data.root_link_pos_w[first_bad].detach().cpu().tolist(),
                    )
                except Exception as _e:
                    print("[OBS-DEBUG-CTX] failed to read asset.data.root_link_pos_w:", _e)
                try:
                    print(
                        "[OBS-DEBUG-CTX] asset.data.root_link_quat_w[bad]",
                        asset.data.root_link_quat_w[first_bad].detach().cpu().tolist(),
                    )
                except Exception as _e:
                    print("[OBS-DEBUG-CTX] failed to read asset.data.root_link_quat_w:", _e)
                try:
                    # contact sensor may not exist; guard access
                    try:
                        cs = env.scene.sensors["contact_forces"]
                    except Exception:
                        cs = None
                    if cs is not None:
                        print(
                            "[OBS-DEBUG-CTX] contact_forces.net_forces_w[bad]",
                            cs.data.net_forces_w[first_bad].detach().cpu().tolist(),
                        )
                except Exception as _e:
                    print("[OBS-DEBUG-CTX] failed to read contact sensor data:", _e)
            except Exception as e:
                print("[OBS-DEBUG-CTX] failed to extract asset context:", e)

            # Attempt a best-effort recovery: write default states back into sim for bad envs
            try:
                recovered = _recover_envs_from_nan(env, asset_cfg, bad_env_ids)
                if len(recovered) > 0:
                    print(f"[OBS-DEBUG-RECOVER] attempted recovery for envs={recovered}; recomputing observation")
                    tensor = isaaclab_mdp.joint_pos(env, asset_cfg)
                else:
                    print("[OBS-DEBUG-RECOVER] no envs recovered")
            except Exception as e:
                print("[OBS-DEBUG-RECOVER] recovery attempt raised:", e)

    return _debug_observation_tensor("joint_pos", tensor)


def joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _debug_observation_tensor("joint_vel", isaaclab_mdp.joint_vel(env, asset_cfg))


def projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _debug_observation_tensor("projected_gravity", isaaclab_mdp.projected_gravity(env, asset_cfg))


def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    return _debug_observation_tensor("last_action", isaaclab_mdp.last_action(env, action_name=action_name))


def base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _debug_observation_tensor("base_ang_vel", isaaclab_mdp.base_ang_vel(env, asset_cfg))


def current_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The current reward value. Returns zeros if the reward manager is not initialized."""
    if not hasattr(env, "reward_manager") or env.reward_manager is None:
        # Assuming the shape should be (num_envs,) based on the environment
        return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)

    try:
        return env.reward_buf.unsqueeze(-1)
    except AttributeError:
        # Fallback to zeros if the reward_manager is initialized but compute isn't ready
        return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)


def joint_torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def is_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return is_contact.float()

def is_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    return contact_time

def lift_mask_by_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg_left: SceneEntityCfg,
    sensor_cfg_right: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    
    """
    Generate a lift mask for the robot's legs based on row-wise height scan gradients from separate left and right sensors.

    Args:
        env (ManagerBasedRLEnv): Simulation environment.
        sensor_cfg_left (SceneEntityCfg): Configuration for the left raycast sensor.
        sensor_cfg_right (SceneEntityCfg): Configuration for the right raycast sensor.
        command_name (str): Command name to check movement intention.
        gradient_threshold (float): Threshold for row-wise height gradient to detect steps.

    Returns:
        torch.Tensor: Lift mask for left and right legs. Shape: [num_envs, 2].
    """
    #* Step 1: Extract ray hit positions (Z coordinates) from left and right sensors
    left_lift_mask_sensor = env.scene.sensors[sensor_cfg_left.name]
    right_lift_mask_sensor = env.scene.sensors[sensor_cfg_right.name]

    left_mask= left_lift_mask_sensor.data.mask 
    right_mask = right_lift_mask_sensor.data.mask  
    
    lift_mask = torch.stack([left_mask, right_mask], dim=1) 

    command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)  # Shape: [num_envs]
    lift_mask *= (command_norm > 0.1).unsqueeze(-1).float()  # Apply movement condition

    return lift_mask

def joint_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def base_euler_angle(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation in the environment frame as Euler angles (roll, pitch, yaw)."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_com_quat_w)

    # Map angles from [0, 2*pi] to [-pi, pi]
    roll = (roll + math.pi) % (2 * math.pi) - math.pi
    pitch = (pitch + math.pi) % (2 * math.pi) - math.pi
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    rpy = torch.stack((roll, pitch, yaw), dim=-1)
    return rpy

def base_euler_angle_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation in the environment frame as Euler angles (roll, pitch, yaw)."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)

    # Map angles from [0, 2*pi] to [-pi, pi]
    roll = (roll + math.pi) % (2 * math.pi) - math.pi
    pitch = (pitch + math.pi) % (2 * math.pi) - math.pi
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    rpy = torch.stack((roll, pitch, yaw), dim=-1)
    if not bool(torch.isfinite(rpy).all()):
        _debug_observation_tensor("root_link_quat_w", asset.data.root_link_quat_w)
    return _debug_observation_tensor("base_euler_angle_link", rpy)


def joint_pos_rel_sin(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions as sine values.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    current_value = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    current_value_sin = torch.sin(current_value)
    return current_value_sin


def joint_pos_rel_cos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions as cosine values.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    current_value = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    current_value_cos = torch.cos(current_value)
    return current_value_cos


def height_scan_raw(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return _sanitize_non_finite("height_scan_raw", sensor.data.ray_hits_w[..., 2])


def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    value = isaaclab_mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
    return _sanitize_non_finite("height_scan", value)


def generated_partial_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)[:, 0]


def generated_scaled_commands(env: ManagerBasedRLEnv, command_name: str, scale: tuple) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    scaled_command = env.command_manager.get_command(command_name).clone()
    scaled_command[:, :3] *= torch.tensor(scale, device=env.device)
    return scaled_command

def generated_scaled_event_commands(env: ManagerBasedRLEnv, command_name: str, scale: tuple) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    scaled_command = env.command_manager.get_command(command_name).clone()
    scaled_command[:, :2] *= torch.tensor(scale, device=env.device)
    return scaled_command

def joint_pos_leg_gear(
    env: ManagerBasedEnv,
    gear_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return joint positions for the configured (leg) joints, scaled by `gear_ratio`."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return pos * gear_ratio

def joint_vel_leg_gear(
    env: ManagerBasedEnv,
    gear_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return joint velocities for the configured (leg) joints, scaled by `gear_ratio`."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return vel * gear_ratio


def robot_joint_torque(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint torque of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.applied_torque.to(device)


def robot_joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint acc of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.joint_acc.to(device)



def measure_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    FL_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids[0]]
    FR_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids[1]]
    RL_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids[2]]
    RR_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids[3]]
    return torch.concat([FL_contact_forces,FR_contact_forces,RL_contact_forces,RR_contact_forces],dim=1)