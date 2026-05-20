"""Stand-drive env with a no-control window after every reset (fall-recovery training).

Zeros the raw action while episode_length_buf < free_fall_steps. Both action
terms (joint_pos, wheel_vel) use offset=0, so processed action = 0 (joint
targets 0 rad, wheel targets 0 rad/s). This is "no policy control", NOT
zero torque -- actuators still drive joints to those targets. The reset_base
event imparts random roll/pitch/yaw velocity so the robot tumbles during
the window. If true zero-torque is needed later, gate
robot.write_joint_stiffness_to_sim(0, env_ids=falling) on the same mask.
"""
from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

import amazing.amazing.tasks.manager_based.locomotion.velocity.mdp as mdp


class AmazingFlatStandDriveEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        env_dt = self.cfg.sim.dt * self.cfg.decimation
        self._free_fall_steps = int(round(self.cfg.free_fall_seconds / env_dt))
        print(
            f"[AmazingFlatStandDriveEnv] free-fall window = "
            f"{self.cfg.free_fall_seconds}s = {self._free_fall_steps} env steps "
            f"(env dt={env_dt}s)"
        )

    def step(self, action: torch.Tensor):
        if self._free_fall_steps > 0:
            mask = (self.episode_length_buf < self._free_fall_steps).unsqueeze(1)
            action = torch.where(mask, torch.zeros_like(action), action)
        # event_params = self.cfg.events.pull_force_up.params
        # mdp.apply_pull_force_z(
        #     self,
        #     torch.arange(self.num_envs, device=self.device),
        #     **event_params,
        # )
        return super().step(action)
