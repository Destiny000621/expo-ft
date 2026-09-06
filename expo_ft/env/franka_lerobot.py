"""Seed the replay buffer from the Franka SFT demonstrations (LeRobot format).

Upstream seeds from raw DROID episode directories (`env/droid_utils.py` reads
`traj.hdf5` per episode). Our demonstrations are the very dataset the pi0.5
checkpoint was fine-tuned on, already converted to LeRobot — so this loader reads
them through **openpi's own dataset code**, with the same TrainConfig the learner
runs. That is the point: a seeded transition is then bit-identical in shape,
key naming, chunk length and normalization to what the SFT saw, and the critic
starts on the same manifold the policy lives on.

Each demo episode is sampled at DECISION granularity (every `replan_steps`
frames), because that is what the online data looks like — mixing per-tick demo
transitions with per-decision online ones would put two different time scales in
one buffer and let the critic bootstrap across them.

Reward convention matches upstream's DROID loader and `droid_env.get_info_for_step`:
demos are successes, so the last decision carries the reward and terminates.
"""

import logging
from typing import Any

import numpy as np
from tqdm import tqdm

import openpi.training.data_loader as _data_loader


def _to_uint8_hwc(image: Any) -> np.ndarray:
    """LeRobot serves video frames as float CHW in [0, 1]; the transforms want uint8 HWC."""
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"expected a single HWC/CHW image, got shape {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def process_franka_lerobot_dataset(
    pi05_train_config,
    task_config,
    *,
    num_data: int = 0,
    repo_id: str | None = None,
    prompt: str | None = None,
    stride: int | None = None,
):
    """Return a list of decision-level transitions ready for `FrankaChunkReplayBuffer.insert`.

    Args:
        pi05_train_config: the openpi TrainConfig the learner uses (defines the
            dataset keys, the action horizon and the repack mapping).
        task_config: the task config (replan_steps, success_reward, prompt).
        num_data: max number of demo EPISODES to load (0 = all).
        repo_id: override the config's LeRobot repo id (the dataset may be
            re-converted under a new name on a different box).
        prompt: language instruction stored with each transition.
        stride: frames between decisions; defaults to task_config.replan_steps.
    """
    import dataclasses  # noqa: PLC0415

    stride = int(stride or task_config.replan_steps)
    prompt = prompt or task_config.language_instruction
    action_horizon = pi05_train_config.model.action_horizon

    if repo_id:
        pi05_train_config = dataclasses.replace(
            pi05_train_config,
            data=dataclasses.replace(pi05_train_config.data, repo_id=repo_id),
        )
    data_config = pi05_train_config.data.create(pi05_train_config.assets_dirs, pi05_train_config.model)

    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, pi05_train_config.model)
    # Only the key REPACK runs here: data transforms, normalization and tokenization
    # happen inside the buffer, once, against each transition's own state.
    repack = data_config.repack_transforms.inputs

    lerobot_ds = getattr(dataset, "_dataset", dataset)  # unwrap TransformedDataset if present
    ep_from = np.asarray(lerobot_ds.episode_data_index["from"])
    ep_to = np.asarray(lerobot_ds.episode_data_index["to"])
    n_eps = len(ep_from)
    if num_data and num_data > 0:
        n_eps = min(n_eps, int(num_data))
    logging.info(
        "Seeding from %s: %d/%d episodes, stride %d frames (%d Hz -> %.2f s per decision)",
        data_config.repo_id, n_eps, len(ep_from), stride, task_config.control_hz,
        stride / float(task_config.control_hz),
    )

    transitions = []
    for ep in tqdm(range(n_eps), desc="seeding demos"):
        start, end = int(ep_from[ep]), int(ep_to[ep])
        indices = list(range(start, end, stride))
        if len(indices) < 2:
            logging.warning("episode %d has %d decisions at stride %d; skipping", ep, len(indices), stride)
            continue
        for j, idx in enumerate(indices):
            item = dataset[idx]
            for tf in repack:
                item = tf(item)
            is_last = j == len(indices) - 1
            observations = {
                "observation/image": _to_uint8_hwc(item["observation/image"]),
                # RAW wrist frame: the wrist crop belongs to the input transform, so
                # the buffer must store what the serve would have been sent.
                "observation/wrist_image": _to_uint8_hwc(item["observation/wrist_image"]),
                "observation/state": np.asarray(item["observation/state"], dtype=np.float32),
                "prompt": prompt,
            }
            actions = np.asarray(item["actions"], dtype=np.float32)
            if actions.shape[0] != action_horizon:
                raise ValueError(f"expected {action_horizon} action rows, got {actions.shape}")
            transitions.append(
                {
                    "observations": observations,
                    "actions": actions,
                    "rewards": np.float32(task_config.success_reward if is_last else 0.0),
                    "masks": np.float32(0.0 if is_last else 1.0),
                    "dones": bool(is_last),
                    # Demos are expert successes: they belong in the actor's
                    # success-only BC pool and in the HIL pool the BC baseline uses.
                    "is_hil": True,
                    "is_success": True,
                }
            )
    logging.info("Seeded %d decision transitions from %d episodes", len(transitions), n_eps)
    return transitions
