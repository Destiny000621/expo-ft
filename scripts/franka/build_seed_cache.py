#!/usr/bin/env python3
"""Build the EXPO-FT seed cache from recorded rollouts, in parallel, without a GPU.

Seeding decodes two 720p HEVC videos per rollout and pushes every sampled decision
through openpi's transforms. That is minutes of CPU work, it needs no GPU, and
doing it inside the learner process is actively harmful: ffmpeg's frame-threading
opens one thread per core (224 on this box) alongside JAX's own pools, which wedged
a seeding run at 1061 threads and no progress.

So: run this once, then start the learner with --seed_cache pointing at the output.
Every later start — including a resume after a crash — loads rows instead of video.

    python scripts/franka/build_seed_cache.py \\
        --rollout_dir ~/expo/seed/data_log_eval_wcrop \\
        --out ~/expo/logs/expo_franka/seed_rows.pkl --workers 12
"""

import argparse
import concurrent.futures as cf
import logging
import multiprocessing as mp
import os
import pathlib
import pickle
import time

# CPU only: this step needs no GPU, and keeping CUDA out of the process is half the
# point of doing it here rather than inside the learner.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# NOTE: do NOT pin OMP_NUM_THREADS / OPENBLAS_NUM_THREADS to 1 here. Doing that
# wedged this script on the learner box — the process sat in futex_wait with 20
# threads, no I/O, right after loading an episode's npz, both sequentially and in
# a pool. The same work in a process without those pins runs in ~8 s per episode.

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("SEED_DEBUG") else logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("seed")


def _worker(args):
    ep, success, config_path, task_path, resize, stride = args
    from train_franka_service import load_config_module  # noqa: PLC0415

    from expo_ft.env.franka_rollout_seed import rows_for_episode  # noqa: PLC0415
    from expo_ft.utils.train_utils import build_pi05_config  # noqa: PLC0415

    task = load_config_module(task_path)
    _, pi05_train_config, _, _ = build_pi05_config(dict(load_config_module(config_path)))
    rows = rows_for_episode(
        ep, success, task, pi05_train_config, resize_size=resize, stride=stride
    )
    return str(ep), bool(success), rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/model/expo_ft_franka_config.py")
    p.add_argument("--config_task", default="configs/task/franka_cable.py")
    p.add_argument("--workers", type=int, default=1,
                   help="1 = sequential (the reliable path, ~8 s per episode). >1 uses a spawn pool, which deadlocked on this box — see the code")
    p.add_argument("--stride", type=int, default=0, help="0 = the task's replan_steps")
    p.add_argument("--num_episodes", type=int, default=0, help="0 = all")
    p.add_argument("--include_failures", type=int, default=1)
    args = p.parse_args()

    from train_franka_service import load_config_module

    from expo_ft.env.franka_rollout_seed import find_rollout_episodes

    task = load_config_module(args.config_task)
    model_cfg = load_config_module(args.config)
    stride = args.stride or int(task.replan_steps)
    resize = int(model_cfg.pi05_resize_size)

    episodes = find_rollout_episodes(pathlib.Path(args.rollout_dir).expanduser(), bool(args.include_failures))
    if args.num_episodes:
        episodes = episodes[: args.num_episodes]
    n_succ = sum(1 for _, s in episodes if s)
    logger.info(
        "%d episodes (%d success / %d failure) from %s, stride %d, %d workers",
        len(episodes), n_succ, len(episodes) - n_succ, args.rollout_dir, stride, args.workers,
    )

    jobs = [(ep, s, args.config, args.config_task, resize, stride) for ep, s in episodes]
    results = {}
    t0 = time.time()

    def _record(k, ep, success, rows):
        results[ep] = rows
        logger.info(
            "[%d/%d] %s: %d decisions (%s) — %.1f min elapsed",
            k, len(jobs), pathlib.Path(ep).name, len(rows),
            "success" if success else "failure", (time.time() - t0) / 60.0,
        )

    if args.workers <= 1:
        # The default, and the one that is known to work end to end: one episode
        # takes ~8 s (decode is ~2300 fps, the transforms are numpy), so the whole
        # 38-episode set is ~5 minutes — and this is a once-per-dataset cost,
        # because the learner then loads the cache.
        for k, (ep, success, *_rest) in enumerate(
            ((j[0], j[1]) + tuple(j[2:]) for j in jobs), 1
        ):
            try:
                _record(k, *_worker((ep, success, args.config, args.config_task, resize, stride)))
            except Exception:  # noqa: BLE001
                logger.exception("episode %s failed — skipping", ep)
    else:
        # Opt-in parallel path. NOTE: on this box it deadlocked — 12 spawn workers
        # sat in futex_wait with 20 threads each and made no progress past the first
        # episode. Sequential seeding is 5 minutes, so this is not worth debugging
        # unless a much larger rollout set ever needs seeding.
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futures = {ex.submit(_worker, j): j[0] for j in jobs}
            for k, fut in enumerate(cf.as_completed(futures), 1):
                try:
                    _record(k, *fut.result())
                except Exception:  # noqa: BLE001
                    logger.exception("episode %s failed — skipping", futures[fut])

    # Episode order is the recording order, not completion order: the buffer's own
    # episode bookkeeping (and any later inspection) reads chronologically.
    rows = [r for ep, _ in episodes if (r_list := results.get(str(ep))) for r in r_list]
    n_succ_rows = sum(1 for r in rows if bool(r["is_success"]))
    out = pathlib.Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out)
    logger.info(
        "wrote %d rows (%d in the success/BC pool) to %s in %.1f min — %.1f GB",
        len(rows), n_succ_rows, out, (time.time() - t0) / 60.0, out.stat().st_size / 1e9,
    )


if __name__ == "__main__":
    main()
