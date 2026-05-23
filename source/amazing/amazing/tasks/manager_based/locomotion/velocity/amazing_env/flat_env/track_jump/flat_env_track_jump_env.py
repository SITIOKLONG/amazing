"""Track-jump env subclass.

Placeholder for jump-specific behavior. Currently a pass-through of
:class:`ManagerBasedRLEnv` so the gym entry_point matches the stand_drive
structure and we have a hook to add jump-side logic later (e.g. a settle
window before commands fire, or per-step diagnostics).
"""
from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv


class AmazingFlatTrackJumpEnv(ManagerBasedRLEnv):
    pass
