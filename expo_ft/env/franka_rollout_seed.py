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


def _hold_fill(rows: np.ndarray, name: str, ep_name: str) -> np.ndarray:
    """Replace non-finite rows with the last valid one (the arm held that command).

    The recorder writes NaN for a tick the agent did not command — `active=False`,
    which is what a pi0.5 rollout does while it waits for the next chunk or while the
    pose stream is stale. Physically the arm held its previous Cartesian target
    (the impedance controller keeps the last one), so a hold-fill is not a repair of
    corrupt data: it is the command that was actually in force. 15-150 such ticks per
    episode appear in the 2026-09-05 rollouts, and scipy rejects them outright
    ("Found zero norm quaternions"), which silently costs a whole episode.

    A leading run of invalid rows is back-filled from the first valid one, since
    there is no earlier command to hold.
    """
    bad = ~np.isfinite(rows).all(axis=1)
    if not bad.any():
        return rows
    if bad.all():
        raise ValueError(f"{ep_name}: every {name} row is non-finite")
    idx = np.where(~bad, np.arange(len(rows)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = rows[idx].copy()
    first_valid = int(np.argmax(~bad))
    if first_valid > 0:
        filled[:first_valid] = rows[first_valid]
    logging.info("[%s] %s: hold-filled %d/%d ticks the agent did not command",
                 ep_name, name, int(bad.sum()), len(rows))
    return filled


def _decode_frames(path: pathlib.Path, wanted: np.ndarray, resize: int = 0) -> dict[int, np.ndarray]:
    """Decode the frame indices in `wanted` as uint8 RGB HWC.

    Runs in a SUBPROCESS by default (`expo_ft.env.rollout_decode_worker`). In-process
    decoding is fast in isolation and deadlocks once the learner's stack is loaded —
    lerobot brings a second libav alongside PyAV's, and ffmpeg's frame threads wedge.
    Set EXPO_INPROCESS_DECODE=1 to decode here anyway (fine on a machine where only
    the seeder is running, and ~1 s faster per episode).
    """
    import os  # noqa: PLC0415

    if os.environ.get("EXPO_INPROCESS_DECODE") == "1":
        from expo_ft.env.rollout_decode_worker import decode_frames  # noqa: PLC0415

        return {int(k): v for k, v in decode_frames(str(path), wanted, resize).items()}

    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="expo_decode_") as tmp:
        idx_path = os.path.join(tmp, "idx.npy")
        out_path = os.path.join(tmp, "frames.npz")
        np.save(idx_path, np.asarray(wanted, np.int64))
        cmd = [sys.executable, "-m", "expo_ft.env.rollout_decode_worker",
               str(path), idx_path, out_path, "--resize", str(int(resize))]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=900)
        if proc.returncode != 0:
            raise RuntimeError(f"decode of {path} failed: {proc.stderr[-2000:]}")
        with np.load(out_path) as z:
            return {int(k): np.asarray(z[k]) for k in z.files}


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
    logging.debug("[%s] loading npz", ep.name)
    st = np.load(ep / "arm0_states.npz")
    ac = np.load(ep / "arm0_actions.npz")
    ticks = np.load(ep / "timestamps.npy").astype(np.float64)
    logging.debug("[%s] npz loaded (%d ticks)", ep.name, len(ticks))

    # Hold-fill BEFORE the rot6d conversion: scipy refuses a NaN quaternion, and an
    # uncaught one costs the whole episode.
    ee_pose = _hold_fill(np.asarray(st["ee_pose"], np.float32), "ee_pose", ep.name)
    target_pose = _hold_fill(np.asarray(ac["target_pose"], np.float32), "target_pose", ep.name)
    gripper_state = _hold_fill(
        np.asarray(st["gripper_pos"], np.float32).reshape(-1, 1), "gripper_pos", ep.name
    )
    gripper_action = _hold_fill(
        np.asarray(ac["gripper_target"], np.float32).reshape(-1, 1), "gripper_target", ep.name
    )
    state = np.concatenate(
        [_pose7_to_xyz_rot6d(ee_pose), _gripper_rad(gripper_state)[:, None]], axis=1
    )
    action = np.concatenate(
        [_pose7_to_xyz_rot6d(target_pose), _gripper_rad(gripper_action)[:, None]], axis=1
    )
    logging.debug("[%s] state/action built", ep.name)
    t = min(len(state), len(action), len(ticks))
    if t < 2 * stride:
        logging.warning("rollout %s has %d ticks (< 2 decisions) — skipping", ep.name, t)
        return
    state, action, ticks = state[:t], action[:t], ticks[:t]

    decisions = list(range(0, t - stride, stride))
    side_idx = _frame_for_ticks(np.load(ep / f"{side_camera}_timestamps.npy"), ticks)
    wrist_idx = _frame_for_ticks(np.load(ep / f"{wrist_camera}_timestamps.npy"), ticks)
    logging.debug("[%s] decoding %d side frames", ep.name, len(decisions))
    side_frames = _decode_frames(ep / f"{side_camera}.mp4", side_idx[decisions], resize=224)
    logging.debug("[%s] decoding %d wrist frames", ep.name, len(decisions))
    wrist_frames = _decode_frames(ep / f"{wrist_camera}.mp4", wrist_idx[decisions])
    logging.debug("[%s] frames decoded", ep.name)

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
                # Already pad-resized to 224 by the decode worker — exactly what
                # the live client puts on the wire.
                "observation/image": np.ascontiguousarray(side, dtype=np.uint8),
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


def rows_for_episode(
    ep,
    success: bool,
    task_config,
    pi05_train_config,
    *,
    resize_size: int = 224,
    stride: int | None = None,
    prompt: str | None = None,
) -> list[dict]:
    """Preprocess ONE rollout into buffer rows (224 px, normalized, tokenized).

    Split out so seeding can run in worker processes: a row is ~450 KB while the
    raw decision it came from carries a 720p wrist frame, so returning rows across
    a process boundary is ~6x cheaper than returning transitions — and doing the
    decode outside the learner keeps ffmpeg's thread pools away from JAX's.
    """
    from expo_ft.data.franka_replay_buffer import FrankaChunkReplayBuffer  # noqa: PLC0415

    transitions = list(
        _episode_transitions(
            pathlib.Path(ep),
            bool(success),
            task_config,
            pi05_train_config.model.action_horizon,
            int(stride or task_config.replan_steps),
            prompt or task_config.language_instruction,
            "external_right",
            "wrist",
            __import__("openpi_client", fromlist=["image_tools"]).image_tools,
        )
    )
    if not transitions:
        return []
    buf = FrankaChunkReplayBuffer(
        example_action=np.zeros((task_config.action_dim,), np.float32),
        capacity=len(transitions),
        pi_train_config=pi05_train_config,
        resize_size=resize_size,
        task_description=prompt or task_config.language_instruction,
    )
    return [buf.insert(tr) for tr in transitions]
