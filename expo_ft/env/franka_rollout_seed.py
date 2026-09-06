"""Seed the replay buffer from recorded pi0.5 ROLLOUTS (avantbot recorder format).

Upstream seeds from teleop demonstrations. This station seeds from its own
frozen-policy eval rollouts (`data_log_eval_wcrop`) instead, and that is the
better data for EXPO-FT for three reasons:

* they are **on-policy** for the checkpoint RL starts from — the critic's first
  job is to rank chunks *this* policy produces, not chunks a human teleoperated;
* they already look like the online data: same 30 Hz Cartesian plant, same
  cameras, same chunked execution, same episode lengths;
* the failures are usable too. A critic trained only on successes has no idea
  what a failure looks like and rates everything highly; the recorded failures
  are exactly the negative signal the online run would otherwise have to buy with
  robot time (they enter the buffer for the critic only — the actor's BC pool is
  success-only by config, so nothing distils toward a failed rollout).

Conventions are taken from the recorder and from openpi's
`scripts/convert_franka_raw_to_lerobot.py`, so a seeded transition is the same
quantity the SFT saw and the same quantity the live wire carries:

    state  (10) = [x, y, z, rot6d(6), gripper_rad]   from arm0_states.npz  ee_pose + gripper_pos
    action (10) = [x, y, z, rot6d(6), gripper_rad]   from arm0_actions.npz target_pose + gripper_target

**Gripper units are radians, not the recorder's 0-1 position.** The checkpoint's
own norm stats settle it: the state gripper column has q99 = 0.7263 and the action
column q99 = 0.7927 — i.e. knuckle radians with a 0.7929 maximum. The training
recordings carried `joint_pos[:, 7]` (radians) and these newer eval recordings
carry `gripper_pos` (0-1, 1 = open), so the conversion below is the deploy
client's own mapping, `rad = (1 - pos) * GRIPPER_MAX_RAD`. Feeding the raw 0-1
column would put a plausible-looking, systematically wrong gripper dimension into
every seeded state.

Images are prepared exactly as the live wire prepares them: the side frame is
`resize_with_pad`-ed to 224 on the client, the wrist frame is sent RAW because the
wrist crop is a learner-side transform. Doing anything else here would seed the
buffer with a different resampling of the same pixels than the online data has.
"""

import json
import logging
import pathlib
from typing import Iterator

import numpy as np
from scipy.spatial.transform import Rotation

# avantbot's FrankaEEPi05Agent.GRIPPER_MAX_RAD — the robotiq knuckle's open angle.
GRIPPER_MAX_RAD = 0.7929


def _pose7_to_xyz_rot6d(pose7: np.ndarray) -> np.ndarray:
    """[qw,qx,qy,qz,x,y,z] (T,7) -> [x,y,z, rot6d(6)] (T,9), sign-invariant.

    Verbatim from openpi's converter: rot6d is the first two COLUMNS of the
    rotation matrix, and R(-q) = R(q), so the recorded quaternion sign flips need
    no canonicalization.
    """
    quat_xyzw = pose7[:, [1, 2, 3, 0]].astype(np.float64)  # scipy is scalar-last
    rot = Rotation.from_quat(quat_xyzw).as_matrix()
    rot6d = np.concatenate([rot[:, :, 0], rot[:, :, 1]], axis=1)
    return np.concatenate([pose7[:, 4:7], rot6d.astype(np.float32)], axis=1).astype(np.float32)


def _gripper_rad(position_0_1: np.ndarray) -> np.ndarray:
    """Recorder gripper position (1 = open) -> knuckle radians (0 = open)."""
    return ((1.0 - np.asarray(position_0_1, np.float32).reshape(-1)) * GRIPPER_MAX_RAD).astype(np.float32)


def _decode_frames(path: pathlib.Path, wanted: np.ndarray) -> dict[int, np.ndarray]:
    """Decode only the frame indices in `wanted` from an mp4, as uint8 RGB HWC.

    PyAV, not OpenCV: these recordings are HEVC (and the LeRobot demo videos are
    AV1, which OpenCV fails on SILENTLY — it returns False rather than raising).
    Sequential decode with a skip set is deliberate; seeking per frame on a
    fragmented recording is slower and can land off-keyframe.
    """
    import av  # noqa: PLC0415

    want = set(int(i) for i in wanted)
    out: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for idx, frame in enumerate(container.decode(stream)):
            if idx in want:
                out[idx] = frame.to_ndarray(format="rgb24")
                if len(out) == len(want):
                    break
    return out


def _frame_for_ticks(cam_ts: np.ndarray, ticks: np.ndarray) -> np.ndarray:
    """Newest camera frame captured at or before each control tick.

    Same rule the converter and the deploy client use — never a future frame.
    Camera timestamps are recorded in MILLISECONDS while the control ticks are in
    seconds, so they are rescaled here rather than silently mis-paired.
    """
    cam = np.asarray(cam_ts, np.float64)
    if cam.size and cam.max() > 1e11:  # ms since epoch
        cam = cam / 1000.0
    idx = np.searchsorted(cam, np.asarray(ticks, np.float64), side="right") - 1
    return np.clip(idx, 0, max(len(cam) - 1, 0))


def find_rollout_episodes(root: pathlib.Path, include_failures: bool) -> list[tuple[pathlib.Path, bool]]:
    """Return [(episode_dir, is_success)] for every recorded episode under `root`."""
    eps = sorted(p.parent for p in pathlib.Path(root).rglob("arm0_states.npz"))
    out = []
    for ep in eps:
        meta_path = ep / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        # Two independent markers, written by different parts of the recorder;
        # either one counts, and they have agreed on every episode checked.
        success = bool(meta.get("success", False)) or (ep / "SUCCESS").exists()
        if success or include_failures:
            out.append((ep, success))
    return out


def process_franka_rollouts(
    root,
    task_config,
    *,
    action_horizon: int = 50,
    stride: int | None = None,
    num_episodes: int = 0,
    include_failures: bool = True,
    prompt: str | None = None,
    side_camera: str = "external_right",
    wrist_camera: str = "wrist",
) -> Iterator[dict]:
    """Yield decision-level transitions ready for `FrankaChunkReplayBuffer.insert`.

    A generator, not a list: a decision carries a raw 720p wrist frame (2.7 MB), so
    materialising a whole seed set costs tens of gigabytes. The caller inserts each
    transition (which resizes it to 224) and drops it.
    """
    from openpi_client import image_tools  # noqa: PLC0415

    root = pathlib.Path(root)
    stride = int(stride or task_config.replan_steps)
    prompt = prompt or task_config.language_instruction
    episodes = find_rollout_episodes(root, include_failures)
    if num_episodes and num_episodes > 0:
        episodes = episodes[:num_episodes]
    n_succ = sum(1 for _, s in episodes if s)
    logging.info(
        "Seeding from rollouts under %s: %d episodes (%d success / %d failure), "
        "stride %d ticks = %.2f s per decision",
        root, len(episodes), n_succ, len(episodes) - n_succ, stride,
        stride / float(task_config.control_hz),
    )
    if not episodes:
        raise FileNotFoundError(f"no recorded episodes under {root}")
    if include_failures:
        logging.info(
            "  failures ARE included: they are critic data only (rewards 0, terminal). "
            "The actor's BC pool is success-only, so nothing distils toward them."
        )

    import time  # noqa: PLC0415

    t_start = time.time()
    for k, (ep, success) in enumerate(episodes, 1):
        t_ep = time.time()
        try:
            n_before = [0]
            # Progress matters here: decoding two 720p HEVC videos per episode is
            # minutes of wall clock with no other output, which reads as a hang.
            logging.info("[seed %d/%d] %s (%s)", k, len(episodes), ep.name,
                         "success" if success else "failure")
            yield from _episode_transitions(
                ep, success, task_config, action_horizon, stride, prompt,
                side_camera, wrist_camera, image_tools,
            )
            logging.info("[seed %d/%d] done in %.1fs (%.1f min elapsed)",
                         k, len(episodes), time.time() - t_ep, (time.time() - t_start) / 60.0)
        except Exception:  # noqa: BLE001
            logging.exception("skipping rollout %s", ep.name)


def _episode_transitions(
    ep, success, task_config, action_horizon, stride, prompt, side_camera, wrist_camera, image_tools
):
    st = np.load(ep / "arm0_states.npz")
    ac = np.load(ep / "arm0_actions.npz")
    ticks = np.load(ep / "timestamps.npy").astype(np.float64)

    state = np.concatenate(
        [_pose7_to_xyz_rot6d(st["ee_pose"].astype(np.float32)),
         _gripper_rad(st["gripper_pos"])[:, None]], axis=1
    )
    action = np.concatenate(
        [_pose7_to_xyz_rot6d(ac["target_pose"].astype(np.float32)),
         _gripper_rad(ac["gripper_target"])[:, None]], axis=1
    )
    t = min(len(state), len(action), len(ticks))
    if t < 2 * stride:
        logging.warning("rollout %s has %d ticks (< 2 decisions) — skipping", ep.name, t)
        return
    state, action, ticks = state[:t], action[:t], ticks[:t]

    decisions = list(range(0, t - stride, stride))
    side_idx = _frame_for_ticks(np.load(ep / f"{side_camera}_timestamps.npy"), ticks)
    wrist_idx = _frame_for_ticks(np.load(ep / f"{wrist_camera}_timestamps.npy"), ticks)
    side_frames = _decode_frames(ep / f"{side_camera}.mp4", side_idx[decisions])
    wrist_frames = _decode_frames(ep / f"{wrist_camera}.mp4", wrist_idx[decisions])

    # Chunk = the rows actually executed from this decision onward, tiled with the
    # last row past the end of the episode — the same quantity build_action_chunks
    # assembles from live decisions.
    padded = np.concatenate([action, np.repeat(action[-1:], action_horizon, axis=0)], axis=0)

    n = len(decisions)
    for j, i in enumerate(decisions):
        side = side_frames.get(int(side_idx[i]))
        wrist = wrist_frames.get(int(wrist_idx[i]))
        if side is None or wrist is None:
            logging.warning("rollout %s decision %d: missing frame — skipping", ep.name, j)
            continue
        last = j == n - 1
        yield {
            "observations": {
                # Exactly what the live client puts on the wire: the side view
                # pad-resized to 224 here, the wrist frame RAW (the crop is a
                # learner-side transform and rejects a pre-resized wrist).
                "observation/image": np.ascontiguousarray(
                    image_tools.resize_with_pad(side, 224, 224), dtype=np.uint8
                ),
                "observation/wrist_image": np.ascontiguousarray(wrist, dtype=np.uint8),
                "observation/state": state[i].astype(np.float32),
                "prompt": prompt,
            },
            "actions": padded[i : i + action_horizon].astype(np.float32),
            "rewards": np.float32(task_config.success_reward if (last and success) else 0.0),
            "masks": np.float32(0.0 if last else 1.0),
            "dones": bool(last),
            # Not human data: these are the policy's own rollouts.
            "is_hil": False,
            # Only successful rollouts enter the actor's success-only BC pool.
            "is_success": bool(success),
        }
