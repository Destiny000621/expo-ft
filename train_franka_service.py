#!/usr/bin/env python3
"""EXPO-FT learner service for the Franka FR3 (avantbot station).

Upstream's entry point (`train_pi_robo.py`) owns BOTH the learner and the robot:
its loop calls `env.get_observation()` / `env.step(action)` over a websocket to
`client/run_client.py`, one 10 Hz DROID step at a time. On this station the robot
loop already exists and is not negotiable — avantbot's runner owns the 30 Hz
Cartesian tick, the ZED cameras, the recorder, the auto-home and the operator
keys — so this process keeps the learner half only, exactly as the DSRL baseline's
service does. The robot asks for a chunk; this answers and remembers.

Mapping onto `train_pi_robo.py`:

    upstream                                    here
    ------------------------------------------- ------------------------------
    `agent.sample_actions(observation)`          POST /infer  (one per replan)
    `env.step(...)` + `get_info_for_step()`      the avantbot runner + operator
    `batch_processor.insert_transition(...)`     staged at /infer, inserted at
                                                 /episode with the labels
    `if done:` update block                      POST /episode (same call)
    `save_checkpoint(...)`                       every N episodes + on SIGINT

Every EXPO-FT decision (N base samples -> N edited samples -> argmax_a Q) happens
here, inside /infer, so the robot never needs the model, the critic, or a GPU.

Wire format: msgpack_numpy over HTTP — the convention DSRL and SubRL already use
on this station, so the robot side shares one serialization helper.

    POST /infer    {episode_id, step_id, image, wrist_image, state (10,) f32, prompt}
                -> {actions (replan_steps, 10) f32 ABSOLUTE, base_policy, selected,
                    param_version, timing}
    POST /episode  {episode_id, is_success, env_steps}
                -> {decisions, updates, buffer_size, total_traj, success_rate}
    POST /abort    {episode_id}   -> drop the staged episode, learn nothing from it
    GET  /healthz                 -> counters

Images may be sent raw (uint8 HWC) or JPEG-encoded bytes; the learner decodes
either. JPEG exists for the remote-learner deployment, where a raw 720p wrist
frame per decision is 2.7 MB on a tunnel.
"""

from __future__ import annotations

import argparse
import http.server
import io
import json
import logging
import os
import pickle
import signal
import socketserver
import threading
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("expo")


def load_config_module(path: str):
    """Load `get_config()` from a configs/... module path or file path."""
    module_path = path.replace(".py", "").replace("/", ".")
    module = __import__(module_path, fromlist=["get_config"])
    return module.get_config()


def decode_image(value) -> np.ndarray:
    """Accept a raw uint8 HWC array or JPEG/PNG bytes; return uint8 HWC."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        from PIL import Image  # noqa: PLC0415

        return np.asarray(Image.open(io.BytesIO(bytes(value))).convert("RGB"), dtype=np.uint8)
    arr = np.asarray(value)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


class Learner:
    """EXPO-FT agent + replay buffer + upstream's episodic update schedule."""

    def __init__(self, v):
        import etils.epath as epath  # noqa: PLC0415
        import jax  # noqa: PLC0415

        import openpi.training.sharding as openpi_sharding  # noqa: PLC0415
        from expo_ft.agents import initialize_checkpoint_dir  # noqa: PLC0415
        from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint  # noqa: PLC0415
        from expo_ft.agents.vla.pi05 import build_pi05  # noqa: PLC0415
        from expo_ft.data.batch_processor import BatchProcessor  # noqa: PLC0415
        from expo_ft.data.franka_replay_buffer import FrankaChunkReplayBuffer  # noqa: PLC0415
        from expo_ft.utils.train_utils import build_pi05_config  # noqa: PLC0415

        self.v = v
        self.jax = jax
        self.lock = threading.Lock()

        self.model_config = load_config_module(v.config)
        self.task = load_config_module(v.config_task)
        if v.replan_steps:
            self.task.replan_steps = int(v.replan_steps)
        self.replan_steps = int(self.task.replan_steps)
        self.prompt = self.task.language_instruction

        jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
        self.mesh = openpi_sharding.make_mesh(v.fsdp_devices)
        self.data_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
        )
        self.replicated_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())

        # --- checkpoint dir -------------------------------------------------
        # ABSOLUTE, always: orbax/flax refuse relative checkpoint paths, and the
        # DSRL run lost 6,680 gradient steps of weights to exactly that (the save
        # silently wrote nothing while the buffer saved fine).
        self.run_dir = os.path.abspath(os.path.join(v.output_dir, v.run_name))
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.rows_dir = os.path.join(self.run_dir, "rows")
        os.makedirs(self.rows_dir, exist_ok=True)
        overwrite = bool(v.overwrite)
        if not overwrite and not v.resume and self._run_dir_is_empty():
            # Upstream refuses ANY existing checkpoint dir. A run that died before
            # its first save (a missing seed cache, a wandb prompt, a typo) leaves an
            # empty one behind, and then the corrected relaunch fails on the
            # leftover. An empty run dir has nothing to protect.
            logger.info("run dir %s exists but holds no checkpoint, rows or counters — "
                        "treating it as fresh", self.run_dir)
            overwrite = True
        self.checkpoint_manager, resuming = initialize_checkpoint_dir(
            epath.Path(self.ckpt_dir), keep_period=v.keep_period, overwrite=overwrite, resume=v.resume
        )
        self.resuming = bool(resuming)

        # --- the VLA + the EXPO-FT agent ------------------------------------
        _, pi05_train_config, _, _ = build_pi05_config(dict(self.model_config))
        self.pi05_train_config = pi05_train_config
        logger.info("pi0.5 config %s (horizon %d, action_dim %d)", pi05_train_config.name,
                    pi05_train_config.model.action_horizon, pi05_train_config.model.action_dim)

        actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
            self.model_config, v.seed, self.mesh, self.data_sharding, self.replicated_sharding,
            self.resuming, self.prompt,
        )

        # --- buffers --------------------------------------------------------
        rb_args = dict(
            example_action=np.zeros((self.task.action_dim,), dtype=np.float32),
            capacity=int(v.buffer_capacity),
            pi_train_config=pi05_train_config,
            resize_size=int(self.model_config.pi05_resize_size),
            task_description=self.prompt,
            discount=float(self.model_config.discount),
        )
        self.replay = FrankaChunkReplayBuffer(**rb_args)
        self.replay.seed(v.seed)
        self.offline_replay = FrankaChunkReplayBuffer(**rb_args)
        self.offline_replay.seed(v.seed)

        dataset = None
        seed_rows = None
        if not v.eval:
            seed_rows, dataset = self._seed_source()

        self.batch_processor = BatchProcessor(
            replay_buffer=self.replay,
            offline_replay_buffer=self.offline_replay,
            data_sharding=self.data_sharding,
            batch_size=int(v.batch_size),
            utd_ratio=int(v.utd_ratio),
            offline_ratio=float(v.offline_ratio),
            actor_success_only=bool(self.model_config.actor_success_only),
            use_dagger_hil_sampling=False,
            dataset=dataset,
        )
        if seed_rows is not None:
            self._insert_seed_rows(seed_rows)
        elif dataset is not None:
            self._write_seed_cache()
        if v.actor_batch_size:
            # The actor batch is a pi0.5 backward pass; the critic batch is a small
            # ResNet. They do not belong at the same size on one GPU.
            self.batch_processor.batch_size = int(v.actor_batch_size)

        example_obs, example_state, example_action = self.replay.convert_to_critic_format(
            {
                "base_image": self.replay.dataset_dict["base_image"][0],
                "left_wrist_image": self.replay.dataset_dict["left_wrist_image"][0],
                "state": self.replay.dataset_dict["state"][0],
                "actions": self.replay.dataset_dict["actions"][0],
            }
        )
        # Take these from the EXAMPLE arrays, not from the task config: the state
        # the critic and the output transform see is the model-PADDED one (32), while
        # the action dim is the environment's (10). Upstream sets them the same way;
        # hard-coding the task's 10 for the state would silently disagree with the
        # batches, which are built from the same buffer arrays.
        actor.action_dim = int(np.asarray(example_action).shape[-1])
        actor.state_dim = int(np.asarray(example_state).shape[-1])
        if actor.action_dim != int(self.task.action_dim):
            raise ValueError(
                f"buffer action dim {actor.action_dim} != task action_dim {self.task.action_dim}"
            )
        self.agent = load_agent(
            seed=v.seed,
            example_observation=example_obs,
            example_action=example_action,
            example_state=example_state,
            actor=actor,
            actor_train_state=actor_train_state,
            target_actor_params=target_actor_params,
            agent_kwargs=agent_kwargs,
            metadata=vla_metadata,
            mesh=self.mesh,
            data_sharding=self.data_sharding,
            replicated_sharding=self.replicated_sharding,
            resume=self.resuming,
            replan_steps=self.replan_steps,
            default_prompt=self.prompt,
            residual_action_xyzg=bool(self.task.residual_action_xyzg),
        )

        # --- counters -------------------------------------------------------
        self.updates = 0          # agent.update() calls = param_version
        self.total_traj = 0
        self.total_env_steps = 0
        self.successes: list[int] = []
        self.combine_rng = jax.random.PRNGKey(v.seed + 100)
        self._staged: dict[int, list] = {}
        self._episode_ids: list[int] = []

        if v.resume:
            # Rows and counters restore whenever --resume is asked for, whether or
            # not a checkpoint exists. A run that died before its first interval
            # save (episodes 1-4) has real robot episodes in rows/ and nothing in
            # checkpoints/; resuming it must keep the data, and say plainly that
            # the weights are fresh (the DSRL run lost episodes to this exact gap).
            self._restore()

        self.wandb = self._make_wandb()
        budget = int(getattr(v, "num_episodes", 0) or 0)
        if budget and not v.eval:
            # ~108 decisions per 90 s episode at replan 25; the buffer preallocates,
            # so an undersized capacity silently starts overwriting the run's own
            # early episodes instead of failing.
            expected = len(self.replay) + budget * (self.task.max_episode_steps // self.replan_steps)
            if expected > v.buffer_capacity:
                logger.warning(
                    "buffer capacity %d may be short: %d seeded + %d episodes x ~%d decisions "
                    "= ~%d transitions. The buffer is a ring — it would drop the oldest "
                    "(seeded demo) data first.",
                    v.buffer_capacity, len(self.replay), budget,
                    self.task.max_episode_steps // self.replan_steps, expected,
                )
        mode = "EVAL" if v.eval else "TRAIN"
        logger.info(
            "%s ready: replan %d rows (%.2f s @ %d Hz), N=%d + %d edits @ scale %.2f, "
            "discount %.4f^%s, buffer %d/%d, updates %d",
            mode, self.replan_steps, self.replan_steps / float(self.task.control_hz),
            self.task.control_hz, self.model_config.N, self.model_config.n_edit_samples,
            self.model_config.edit_scale, self.model_config.discount,
            self.model_config.discount_power, len(self.replay), v.buffer_capacity, self.updates,
        )

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------
    def _run_dir_is_empty(self) -> bool:
        if not os.path.isdir(self.ckpt_dir):
            return True
        has_ckpt = any(name.isdigit() for name in os.listdir(self.ckpt_dir))
        has_rows = any(name.endswith(".pkl") for name in os.listdir(self.rows_dir))
        has_counters = os.path.exists(os.path.join(self.run_dir, "counters.json"))
        return not (has_ckpt or has_rows or has_counters)

    def _seed_source(self):
        """Return (preprocessed_rows | None, transition_iterable | None).

        The buffer can be warm-started from two places, and only one runs:

        * ``rollouts`` (default) — this station's own recorded pi0.5 rollouts
          (``data_log_eval_wcrop``). On-policy for the checkpoint RL starts from,
          and it brings recorded FAILURES, which is the negative signal a
          success-only seed cannot give the critic.
        * ``lerobot`` — the SFT demonstrations, upstream's kind of seed. Requires
          an explicit ``--dataset_repo_id``: the demo set that matches this
          checkpoint is not the one sitting on every box, and seeding from the
          wrong conversion is invisible until the results are wrong.

        The cache holds PREPROCESSED rows (224 px, ~450 KB each), not raw
        transitions: a raw decision carries a 720p wrist frame, so a cached seed
        set would otherwise be tens of gigabytes.
        """
        cache = self.v.seed_cache
        if cache and os.path.exists(cache):
            logger.info("loading seeded rows from cache %s", cache)
            with open(cache, "rb") as f:
                return pickle.load(f), None

        source = self.v.seed_source
        if source == "none":
            logger.warning("no buffer seeding (--seed_source none): the critic starts blank "
                           "and the first robot episodes pay for that")
            return None, None
        if source == "rollouts":
            from expo_ft.env.franka_rollout_seed import process_franka_rollouts  # noqa: PLC0415

            if not self.v.rollout_seed_dir:
                raise SystemExit("--seed_source rollouts needs --rollout_seed_dir")
            if not self.v.allow_inline_seeding:
                # Decoding video inside THIS process is the one configuration that
                # bit us: by now pi0.5 is on the GPU, so the process carries a CUDA
                # context and JAX's thread pools, and ffmpeg's decode threads on a
                # 224-core box turned that into 1000+ threads making no progress.
                # Build the cache in a separate, GPU-free process instead — it is
                # also ~10x faster because it parallelises across episodes.
                raise SystemExit(
                    "no seed cache at "
                    f"{cache or '(--seed_cache not set)'}. Build it first (minutes, no GPU):\n"
                    f"  python scripts/franka/build_seed_cache.py \\\n"
                    f"      --rollout_dir {self.v.rollout_seed_dir} \\\n"
                    f"      --out {cache or '<path>.pkl'}\n"
                    "then start the learner with the same --seed_cache. Pass "
                    "--allow_inline_seeding 1 to decode here anyway."
                )
            return None, process_franka_rollouts(
                self.v.rollout_seed_dir,
                self.task,
                action_horizon=self.pi05_train_config.model.action_horizon,
                stride=self.v.seed_stride or self.replan_steps,
                num_episodes=int(self.v.num_data),
                include_failures=bool(self.v.rollout_include_failures),
                prompt=self.prompt,
            )
        if source == "lerobot":
            from expo_ft.env.franka_lerobot import process_franka_lerobot_dataset  # noqa: PLC0415

            if not self.v.dataset_repo_id:
                raise SystemExit(
                    "--seed_source lerobot needs an explicit --dataset_repo_id (the demo set "
                    "that matches this checkpoint is not necessarily the one on this box)"
                )
            return None, process_franka_lerobot_dataset(
                self.pi05_train_config,
                self.task,
                num_data=int(self.v.num_data),
                repo_id=self.v.dataset_repo_id,
                prompt=self.prompt,
                stride=self.v.seed_stride or self.replan_steps,
            )
        raise SystemExit(f"unknown --seed_source {source!r}")

    def _insert_seed_rows(self, rows) -> None:
        """Re-insert cached preprocessed seed rows (no transforms, no video decode)."""
        for row in rows:
            self.replay.insert_row(row)
        n_succ = int(np.sum(self.replay.dataset_dict["is_success"][: len(self.replay)]))
        logger.info("seeded %d transitions from cache (%d in the success/BC pool)",
                    len(rows), n_succ)

    def _write_seed_cache(self) -> None:
        cache = self.v.seed_cache
        size = len(self.replay)
        n_succ = int(np.sum(self.replay.dataset_dict["is_success"][:size]))
        logger.info("seeded %d transitions (%d in the success/BC pool)", size, n_succ)
        if not cache or size == 0:
            return
        rows = [
            {k: np.asarray(v[i]) if not isinstance(v, dict) else v for k, v in self.replay.dataset_dict.items()}
            for i in range(size)
        ]
        tmp = cache + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache)
        logger.info("cached %d preprocessed seed rows to %s", size, cache)

    def _make_wandb(self):
        if not self.v.wandb_project:
            logger.warning("wandb disabled (--wandb_project '') — the run will have no curve")
            return None
        import wandb  # noqa: PLC0415

        wandb.init(
            project=self.v.wandb_project,
            name=self.v.run_name,
            group=self.v.wandb_group or None,
            config={**dict(self.v.__dict__), "replan_steps": self.replan_steps},
            resume="allow",
            id=self.v.run_name,
        )
        return wandb

    # ------------------------------------------------------------------
    # decisions
    # ------------------------------------------------------------------
    def infer(self, episode_id: int, step_id: int, obs: dict) -> dict:
        """One EXPO-FT decision: N base chunks -> N edits -> argmax_a Q -> execute."""
        with self.lock:
            t0 = time.perf_counter()
            observations = {
                "observation/image": decode_image(obs["image"]),
                "observation/wrist_image": decode_image(obs["wrist_image"]),
                "observation/state": np.asarray(obs["state"], dtype=np.float32),
                "prompt": obs.get("prompt") or self.prompt,
            }
            # A new episode id flushes anything stale: sessions restart their
            # numbering, and a mid-episode crash must not splice two episodes
            # together under a reused id (the DSRL learner learned this live).
            if episode_id not in self._staged:
                self._staged = {episode_id: []}
                self._episode_ids = [episode_id]

            only_base = bool(self.v.eval_base_only)
            chunk, self.agent, info = self.agent.sample_actions(observations, only_base_actions=only_base)
            chunk = np.asarray(chunk, dtype=np.float32)
            executed = chunk[: self.replan_steps]

            if not self.v.eval:
                self._staged[episode_id].append((observations, executed.copy()))

            dt_ms = (time.perf_counter() - t0) * 1000.0
            if step_id % 10 == 0:
                # Not logging info["sample_time"]: it is a time.monotonic() delta taken
                # INSIDE a jitted function, i.e. a constant frozen at trace time
                # (~1.1 s), which misreads as a slow policy. dt_ms is the real cost.
                logger.info(
                    "ep %d decision %d: %.0f ms, |a| %.3f, buffer %d",
                    episode_id, step_id, dt_ms,
                    float(np.linalg.norm(executed[0, :3])), len(self.replay),
                )
            return {
                "actions": executed,
                # base_policy: nothing has been learned yet, so the decision is the
                # SFT policy plus an untrained edit — this is the frozen baseline.
                "base_policy": bool(self.updates == 0),
                "param_version": self.updates,
                "timing": {"decision_ms": dt_ms},
            }

    # ------------------------------------------------------------------
    # episodes
    # ------------------------------------------------------------------
    def close_episode(self, episode_id: int, is_success: bool, env_steps: int = 0) -> dict:
        with self.lock:
            staged = self._staged.pop(episode_id, [])
            n = len(staged)
            if self.v.eval:
                self.total_traj += 1
                self.successes.append(int(bool(is_success)))
                rate = float(np.mean(self.successes))
                logger.info(
                    "EVAL episode %d: %s — running %d/%d (%.0f%%)", episode_id,
                    "SUCCESS" if is_success else "failure", sum(self.successes),
                    len(self.successes), 100 * rate,
                )
                return {"decisions": n, "updates": 0, "buffer_size": 0,
                        "total_traj": self.total_traj, "success_rate": rate}

            if n == 0:
                logger.warning("episode %d closed with no decisions — nothing to learn from", episode_id)
                return {"decisions": 0, "updates": 0, "buffer_size": len(self.replay),
                        "total_traj": self.total_traj, "success_rate": self._rate()}

            inserted = self._insert_episode(staged, is_success)
            self.total_traj += 1
            self.total_env_steps += int(env_steps or n * self.replan_steps)
            self.successes.append(int(bool(is_success)))

            n_updates = 0
            if self._can_update():
                n_updates = int(self.v.num_updates)
                t0 = time.perf_counter()
                self._run_updates(n_updates)
                logger.info("episode %d: %d update(s) x utd %d in %.1f s",
                            episode_id, n_updates, self.v.utd_ratio, time.perf_counter() - t0)
            else:
                logger.info(
                    "episode %d: no updates yet (%d/%d episodes before training starts)",
                    episode_id, self.total_traj, self.v.min_episodes_before_update,
                )

            if self.v.checkpoint_interval_episodes and (
                self.total_traj % self.v.checkpoint_interval_episodes == 0
            ):
                self.save("interval")

            self._log_episode(is_success)
            budget = int(getattr(self.v, "num_episodes", 0) or 0)
            if budget:
                logger.info(
                    "PROGRESS: episode %d/%d of the robot budget — %d successes (%.0f%%), "
                    "last 10: %.0f%%, buffer %d/%d transitions",
                    self.total_traj, budget, sum(self.successes), 100 * self._rate(),
                    100 * float(np.mean(self.successes[-10:])), len(self.replay),
                    self.v.buffer_capacity,
                )
                if self.total_traj == budget:
                    logger.warning(
                        "the %d-episode budget is spent. The learner keeps serving; stop it "
                        "with Ctrl+C (which saves) when you are done.", budget
                    )
            return {
                "decisions": inserted,
                "updates": n_updates,
                "buffer_size": len(self.replay),
                "total_traj": self.total_traj,
                "success_rate": self._rate(),
            }

    def abort_episode(self, episode_id: int) -> dict:
        with self.lock:
            n = len(self._staged.pop(episode_id, []))
            logger.warning("episode %d ABORTED — %d staged decisions dropped", episode_id, n)
            return {"dropped": n, "buffer_size": len(self.replay)}

    def _insert_episode(self, staged, is_success: bool) -> int:
        from expo_ft.data.franka_replay_buffer import build_action_chunks  # noqa: PLC0415

        executed = np.stack([e for _, e in staged], axis=0)  # (n, replan_steps, 10)
        chunks = build_action_chunks(
            executed, self.pi05_train_config.model.action_horizon, self.replan_steps
        )
        n = len(staged)
        rows = []
        self.batch_processor.on_episode_start()
        for i, (obs, _) in enumerate(staged):
            last = i == n - 1
            row = self.replay.insert(
                {
                    "observations": obs,
                    "actions": chunks[i],
                    # Sparse and terminal, upstream's convention exactly
                    # (`droid_env.get_info_for_step`): reward on success only, and
                    # mask 0 on ANY terminal decision — including a failure, which is
                    # therefore learned as "this state is worth 0", not bootstrapped.
                    "rewards": np.float32(self.task.success_reward if (last and is_success) else 0.0),
                    "masks": np.float32(0.0 if last else 1.0),
                    "dones": bool(last),
                    "is_hil": False,
                    "is_success": False,  # set for the whole episode below
                }
            )
            rows.append(row)
        self.batch_processor.on_episode_done(bool(is_success))
        if self.v.checkpoint_buffer:
            self._save_rows(rows, is_success)
        logger.info("episode inserted: %d decisions, success=%s, buffer %d",
                    n, is_success, len(self.replay))
        return n

    def _can_update(self) -> bool:
        if self.v.eval:
            return False
        if self.total_traj < int(self.v.min_episodes_before_update):
            return False
        return len(self.replay) >= int(self.v.batch_size)

    def _run_updates(self, num_updates: int) -> None:
        import jax  # noqa: PLC0415

        t_block = time.perf_counter()
        for k in range(num_updates):
            t_upd = time.perf_counter()
            batch, actor_batch, self.combine_rng = self.batch_processor.next_batch(self.combine_rng)
            if actor_batch is None and bool(self.model_config.actor_success_only):
                # No success episodes in the pool yet: the critic can still learn,
                # but the pi0.5 BC step has no target. Skip the whole update rather
                # than distil toward failures — that is what actor_success_only means.
                logger.warning("no success episodes yet — skipping this update (actor_success_only)")
                return
            self.agent = self.agent.replace(rng=jax.device_put(self.agent.rng, self.replicated_sharding))
            self.agent, info = self.agent.update(self.agent, batch, int(self.v.utd_ratio), actor_batch)
            self.updates += 1
            self._log_update(info)
            if num_updates > 1:
                # A 20-update warm start is ~15 min of silence otherwise (the first
                # update alone is minutes of JIT), which reads as a hang.
                critic = info.get("critic_loss"); sel = info.get("select_ratio_with_residual")
                logger.info(
                    "  update %d/%d: %.1f s (%.1f min elapsed) critic_loss %s edit-selected %s",
                    k + 1, num_updates, time.perf_counter() - t_upd,
                    (time.perf_counter() - t_block) / 60.0,
                    f"{float(critic):.4f}" if critic is not None else "n/a",
                    f"{float(sel):.2f}" if sel is not None else "n/a",
                )

    def compile_decision_path(self) -> None:
        """Run one throwaway decision so the robot never pays for JIT.

        The first /infer compiles pi0.5 sampling for N candidates, the critic
        ensemble and the residual actor — minutes on a cold cache. Paying that
        here, before the robot connects, is the difference between "the arm is
        stuck" and "the learner is starting".
        """
        wrist = np.full((720, 1280, 3), 128, dtype=np.uint8)
        side = np.full((224, 224, 3), 128, dtype=np.uint8)
        obs = {
            "observation/image": side,
            "observation/wrist_image": wrist,
            "observation/state": np.zeros((int(self.task.state_dim),), dtype=np.float32),
            "prompt": self.prompt,
        }
        t0 = time.perf_counter()
        with self.lock:
            chunk, self.agent, _ = self.agent.sample_actions(
                obs, only_base_actions=bool(self.v.eval_base_only)
            )
        chunk = np.asarray(chunk)
        logger.info("decision path compiled in %.1f s (chunk %s -> executing %d rows)",
                    time.perf_counter() - t0, chunk.shape, self.replan_steps)

    def run_initial_updates(self) -> None:
        """Optional pre-training on the seeded demos before the robot connects.

        NOT upstream behaviour (`train_pi_robo.py` gates every update on 10 collected
        episodes). It exists because real episodes are the scarce resource here: the
        seeded demos already contain everything a fresh critic can learn offline, and
        spending robot time to discover that is waste. Off by default.
        """
        n = int(self.v.initial_updates)
        if n <= 0 or self.v.eval:
            return
        logger.info("running %d offline update(s) on the seeded buffer (first one JIT-compiles)", n)
        t0 = time.perf_counter()
        with self.lock:
            self._run_updates(n)
        logger.info("offline warm start done in %.1f s (%d updates)", time.perf_counter() - t0, self.updates)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _save_rows(self, rows, is_success: bool) -> None:
        """Persist an episode's PREPROCESSED rows (224px, normalized).

        Upstream pickles one raw transition per file. A raw Franka transition is a
        720p wrist frame (the wrist crop happens server-side, so the raw frame is
        what must be stored) — 2.7 MB each, ~300 MB per episode. The preprocessed
        row is what the buffer actually holds (~450 KB) and restoring it re-inserts
        the identical array, with no second pass through the transforms.
        """
        path = os.path.join(self.rows_dir, f"ep_{self.total_traj:05d}_{int(is_success)}.pkl")
        tmp = path + ".tmp"
        try:
            with open(tmp, "wb") as f:
                pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except Exception:  # noqa: BLE001
            logger.exception("could not persist episode rows")

    def _restore(self) -> None:
        from expo_ft.agents.alg.expo_ft import restore_checkpoint  # noqa: PLC0415

        steps = tuple(self.checkpoint_manager.all_steps())
        if steps:
            self.agent = restore_checkpoint(self.checkpoint_manager, self.agent)
            self.agent = self.agent.cache_infer_params()
            self.updates = max(steps)
            logger.info("restored agent from step %d", self.updates)
        else:
            logger.warning(
                "--resume but NO checkpoint in %s: weights are the SFT init (updates=0). "
                "Persisted episodes and counters are restored below; the first update "
                "block retrains critic/edit policy from that buffer.", self.ckpt_dir,
            )
        counters = os.path.join(self.run_dir, "counters.json")
        if os.path.exists(counters):
            with open(counters) as f:
                c = json.load(f)
            self.total_traj = c.get("total_traj", 0)
            self.total_env_steps = c.get("total_env_steps", 0)
            self.successes = c.get("successes", [])
            if not steps:
                # The counter describes WEIGHTS that no longer exist.
                self.updates = 0
        if self.v.eval:
            # An eval restores WEIGHTS, not data: re-inserting a whole run's rows
            # costs minutes and the buffer is never read in eval mode.
            logger.info("eval mode: skipping the replay-buffer restore")
            return
        files = sorted(f for f in os.listdir(self.rows_dir) if f.endswith(".pkl"))
        for name in files:
            with open(os.path.join(self.rows_dir, name), "rb") as f:
                rows = pickle.load(f)
            success = name.endswith("_1.pkl")
            start = self.replay._insert_index  # noqa: SLF001
            for row in rows:
                self.replay.insert_row(row)
            if success:
                self.replay.mark_episode_success(start, self.replay._insert_index)  # noqa: SLF001
        if files:
            logger.info("restored %d episode(s) of rows: buffer %d transitions",
                        len(files), len(self.replay))

    def save(self, reason: str) -> dict:
        from expo_ft.agents.alg.expo_ft import save_checkpoint  # noqa: PLC0415

        out = {}
        if self.v.eval:
            return out
        try:
            save_checkpoint(self.checkpoint_manager, self.agent, self.updates)
            out["checkpoint"] = f"{self.ckpt_dir}@{self.updates}"
        except Exception as exc:  # noqa: BLE001
            logger.error("checkpoint NOT saved (%s)", exc)
        try:
            path = os.path.join(self.run_dir, "counters.json")
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(
                    {"total_traj": self.total_traj, "updates": self.updates,
                     "total_env_steps": self.total_env_steps, "successes": self.successes},
                    f,
                )
            os.replace(tmp, path)
            out["counters"] = path
        except Exception as exc:  # noqa: BLE001
            logger.error("counters NOT saved (%s)", exc)
        logger.info("saved [%s]: %s", reason, out or "nothing")
        return out

    # ------------------------------------------------------------------
    # logging / health
    # ------------------------------------------------------------------
    def _rate(self) -> float:
        return float(np.mean(self.successes)) if self.successes else 0.0

    def _log_update(self, info) -> None:
        if self.wandb is None:
            return
        flat = {}
        for k, val in info.items():
            val = self.jax.device_get(val)
            if getattr(val, "ndim", 0) == 0:
                flat[f"training/{k}"] = float(val)
        self.wandb.log(flat, step=self.updates)

    def _log_episode(self, is_success: bool) -> None:
        if self.wandb is None:
            return
        self.wandb.log(
            {
                "is_success": int(bool(is_success)),
                "total_num_traj": self.total_traj,
                "env_steps": self.total_env_steps,
                "replay_buffer_size": len(self.replay),
                "episode_reward": float(bool(is_success)),
                # Parity aliases with the DSRL and SubRL learners so all three
                # baselines can be read off one dashboard.
                "success_rate_10": float(np.mean(self.successes[-10:])),
                "success_rate_20": float(np.mean(self.successes[-20:])),
            },
            step=self.updates,
        )

    def health(self) -> dict:
        return {
            "mode": "eval" if self.v.eval else "train",
            "base_policy": bool(self.updates == 0),
            "updates": self.updates,
            "total_traj": self.total_traj,
            "buffer_size": len(self.replay),
            "success_rate": self._rate(),
            "replan_steps": self.replan_steps,
            "action_dim": int(self.task.action_dim),
            "prompt": self.prompt,
        }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class _Handler(http.server.BaseHTTPRequestHandler):
    learner: Learner = None  # set on the class before serve_forever
    # HTTP/1.1 so the robot's client can keep ONE connection for the whole run.
    # The default (HTTP/1.0) closes after every response, and a fresh connection
    # per decision through the SSH tunnel costs a channel open plus TCP slow-start
    # on the ~140 KB observation — measured at ~300 ms of a 485 ms decision against
    # an 80 ms RTT, with the learner itself at ~95 ms. Every response below sets
    # Content-Length, which HTTP/1.1 keep-alive requires.
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: D102 — silence per-request stderr spam
        pass

    def _send(self, obj, code=200):
        from openpi_client import msgpack_numpy  # noqa: PLC0415

        body = msgpack_numpy.Packer().pack(obj)
        self.send_response(code)
        self.send_header("Content-Type", "application/msgpack")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/healthz"):
            body = json.dumps(self.learner.health()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802
        from openpi_client import msgpack_numpy  # noqa: PLC0415

        try:
            payload = msgpack_numpy.unpackb(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.startswith("/infer"):
                self._send(self.learner.infer(
                    int(payload["episode_id"]), int(payload.get("step_id", 0)), payload
                ))
            elif self.path.startswith("/episode"):
                self._send(self.learner.close_episode(
                    int(payload["episode_id"]), bool(payload["is_success"]),
                    int(payload.get("env_steps", 0)),
                ))
            elif self.path.startswith("/abort"):
                self._send(self.learner.abort_episode(int(payload["episode_id"])))
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            # The client's timeout fired and it hung up before the reply. The WORK
            # still happened — for /episode the insert and the update block both
            # completed; only the response was lost.
            logger.warning("client hung up before the reply on %s — the request WAS processed",
                           self.path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("request failed")
            try:
                self._send({"error": repr(exc)}, code=500)
            except (BrokenPipeError, ConnectionResetError):
                pass


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main(v) -> None:
    _Handler.learner = Learner(v)
    learner = _Handler.learner
    learner.compile_decision_path()
    learner.run_initial_updates()

    def _bye(signum, _frame):
        logger.warning("signal %s — saving before exit", signum)
        learner.save(f"signal-{signum}")
        os._exit(0)

    signal.signal(signal.SIGINT, _bye)
    signal.signal(signal.SIGTERM, _bye)

    server = _Server((v.host, int(v.port)), _Handler)
    logger.info("EXPO-FT learner listening on %s:%d — waiting for the robot loop", v.host, v.port)
    if v.eval:
        logger.info("EVALUATION mode: no inserts, no updates, no saves. The success "
                    "tally printed per episode IS the eval.")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant_json", required=True, help="JSON dump of the launcher's variant")
    args = parser.parse_args()
    with open(args.variant_json) as f:
        main(argparse.Namespace(**json.load(f)))
