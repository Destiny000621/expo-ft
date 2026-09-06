"""Franka FR3 double-cable insertion — the avantbot station's EXPO-FT task.

Upstream's task configs describe a DROID environment that the LEARNER drives one
env step at a time (`client/envs/droid_env.py` + `client/run_client.py`). This
station does not work that way: the robot loop lives in avantbot
(`python -m avantbot.collect`, agent `franka_pi05_expo`) and owns the 30 Hz
control tick, the cameras, the recorder and the operator keys. So this config
carries only what the LEARNER still needs to know about the task — shapes,
cadence, prompt and episode limits — and nothing about how to move an arm.

The numbers here are the same ones the DSRL baseline runs with, on purpose:
DSRL and EXPO-FT are meant to be compared, so they must see the same task.
"""

import numpy as np

from configs.task import real_base


def get_config():
    config = real_base.get_config()

    # Not "droid": the learner must not try to import client.envs or build an env.
    # train_franka_service.py serves decisions to the robot instead of stepping one.
    config.env_type = "franka"
    config.env_name = "franka_cable"
    config.env = None

    config.language_instruction = (
        "Unplug the two cables from the right router, then insert them into the left router"
    )

    # --- shapes -------------------------------------------------------------
    # rot6d10: absolute [x, y, z, r6_0..r6_5, gripper]; the same contract the
    # avantbot client speaks (state_schema "r6") and the SFT config was trained on.
    # 10-D actions, NOT DROID's 7-D cartesian velocity: there is no clip(-1, 1)
    # and no gripper binarisation anywhere in this port.
    config.action_dim = 10
    config.state_dim = 10
    config.action_horizon = 50  # pi0.5 chunk (Pi0Config(pi05=True) default)
    config.example_action = np.zeros((1, 10), dtype=np.float32)

    # --- cadence ------------------------------------------------------------
    # One decision = one replan = 25 of the chunk's 50 rows = 0.83 s at 30 Hz.
    # The robot side MUST agree: session YAML vla.chunk_size_threshold = 1 -
    # replan_steps / action_horizon = 0.5, blend_mode latest_only. The agent
    # refuses to start if they disagree, because a mismatch silently trains the
    # critic on action chunks the arm never executed.
    config.control_hz = 30
    config.replan_steps = 25
    # 2700 ticks = 90 s, ~108 decisions. The 100 SFT demos average 38.4 s, so this
    # is ~2.3x headroom; same cap as the DSRL runs.
    config.max_episode_steps = 2700

    # --- residual edit space ------------------------------------------------
    # False: the edit policy may move rotation too. Upstream's pick task sets this
    # True because a top-down grasp needs no reorientation; an RJ45 insertion is
    # exactly the case where a few degrees of wrist alignment decide the episode.
    config.residual_action_xyzg = False

    # --- reward -------------------------------------------------------------
    # Sparse and operator-labelled: 1 on the last decision of a success episode,
    # 0 everywhere else; mask 0 on any terminal decision (upstream's convention,
    # `client/envs/droid_env.py:get_info_for_step`). Failures are the majority of
    # the early signal — they must be labelled 0, never dropped.
    config.success_reward = 1.0

    # Camera roles, for the wire contract only (avantbot owns the devices):
    #   side  -> observation/image       -> base_0_rgb
    #   wrist -> observation/wrist_image -> left_wrist_0_rgb (server-side wcrop)
    config.side_camera_id = "side"
    config.wrist_camera_id = "wrist"

    # Unused on this station; kept non-None so nothing downstream trips over them.
    config.bounds = None
    config.reset_joints = None
    config.auto_reset_steps = 0

    return config
