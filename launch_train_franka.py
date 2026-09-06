#!/usr/bin/env python3
"""Launcher for the Franka EXPO-FT learner service.

Upstream's flags live in `train_pi_robo.py` (absl). This port keeps the same knob
names where they mean the same thing, and states the reason for every default that
differs — the difference is almost always "~100 real episodes" instead of a long
DROID run.

    python -m launch_train_franka --port 9112
    python -m launch_train_franka --eval 1 --resume 1 --run_name <trained run>
"""

import argparse
import logging

import train_franka_service


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EXPO-FT learner for the Franka double-cable task")

    # --- what to run --------------------------------------------------------
    p.add_argument("--config", default="configs/model/expo_ft_franka_config.py")
    p.add_argument("--config_task", default="configs/task/franka_cable.py")
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 accepts the robot from another box")
    p.add_argument("--port", type=int, default=9112)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fsdp_devices", type=int, default=1)
    p.add_argument("--replan_steps", type=int, default=0,
                   help="override the task config's replan_steps (0 = use the task config). "
                        "MUST match the robot session's open_loop_horizon.")

    # --- run identity / persistence ----------------------------------------
    p.add_argument("--output_dir", default="./logs/expo_franka")
    p.add_argument("--run_name", default="expo_franka_cable_seed0")
    p.add_argument("--resume", type=int, default=0)
    p.add_argument("--overwrite", type=int, default=0)
    p.add_argument("--keep_period", type=int, default=None)
    # Every 5 episodes rather than upstream's every-N-gradient-steps: an episode is
    # the expensive unit here, and a crash must never cost more than a few of them.
    p.add_argument("--checkpoint_interval_episodes", type=int, default=5)
    p.add_argument("--checkpoint_buffer", type=int, default=1,
                   help="persist each episode's preprocessed rows so a run can be resumed")

    # --- buffer seeding -----------------------------------------------------
    # Default: this station's own recorded pi0.5 rollouts. They are on-policy for
    # the checkpoint RL starts from, and they carry recorded FAILURES — the
    # negative signal a success-only seed cannot give the critic, and the signal
    # the online run would otherwise have to buy with robot episodes.
    p.add_argument("--seed_source", choices=["rollouts", "lerobot", "none"], default="rollouts")
    p.add_argument("--rollout_seed_dir", default="",
                   help="recorder directory of pi0.5 rollouts (e.g. data_log_eval_wcrop)")
    p.add_argument("--rollout_include_failures", type=int, default=1,
                   help="also seed failed rollouts as critic data (rewards 0, terminal). "
                        "The actor's BC pool stays success-only regardless")
    p.add_argument("--num_data", type=int, default=0,
                   help="max EPISODES to seed from the chosen source (0 = all)")
    p.add_argument("--dataset_repo_id", default="",
                   help="--seed_source lerobot only, and REQUIRED there: the demo set that "
                        "matches this checkpoint is not necessarily the one on this box")
    p.add_argument("--seed_stride", type=int, default=0,
                   help="frames between seeded decisions (0 = replan_steps, which matches "
                        "the granularity of the online data)")
    p.add_argument("--allow_inline_seeding", type=int, default=0,
                   help="decode the rollout videos inside the learner process. Off by "
                        "default: this process holds a CUDA context and JAX's thread "
                        "pools, and ffmpeg's decode threads on a many-core box wedged it. "
                        "Use scripts/franka/build_seed_cache.py instead")
    p.add_argument("--seed_cache", default="",
                   help="pickle path to cache the PREPROCESSED seed rows (decoding the "
                        "rollout videos takes minutes; the cache makes a restart instant)")

    # --- optimisation -------------------------------------------------------
    p.add_argument("--batch_size", type=int, default=64, help="critic minibatch")
    # The actor step is a pi0.5 backward pass and the critic step is a small ResNet;
    # upstream runs both at `batch_size` on 4 GPUs. One learner GPU here.
    p.add_argument("--actor_batch_size", type=int, default=16)
    p.add_argument("--utd_ratio", type=int, default=20)
    # README guidance: env_steps / num_updates ~ 20-30. An episode is ~108 decisions
    # at replan 25, so 4 updates/episode lands in that band.
    p.add_argument("--num_updates", type=int, default=4)
    p.add_argument("--offline_ratio", type=float, default=0.0,
                   help="0 = seed the demos into the ONLINE buffer (upstream's default path)")
    # Upstream waits for 10 collected episodes before the first update. With the demos
    # already seeded there is something to learn from immediately, and real episodes
    # are the scarce resource, so this starts after 1.
    p.add_argument("--min_episodes_before_update", type=int, default=1)
    p.add_argument("--initial_updates", type=int, default=0,
                   help="offline updates on the seeded buffer BEFORE the robot connects. "
                        "Not upstream behaviour; it buys critic quality with GPU time "
                        "instead of robot time")
    # THE BUDGET: 100 real robot episodes, same as the DSRL baseline, so the two
    # runs are comparable episode-for-episode. Nothing is capped by it — the
    # learner keeps serving past 100 — but it sizes the buffer and drives the
    # progress line, so an operator can see where the run stands.
    p.add_argument("--num_episodes", type=int, default=100)
    # ~15k decisions expected (100 episodes x ~108 + ~4.6k seeded); the buffer
    # preallocates, so this is also ~9 GB of host RAM at 450 KB/transition.
    p.add_argument("--buffer_capacity", type=int, default=20000)

    # --- evaluation ---------------------------------------------------------
    p.add_argument("--eval", type=int, default=0,
                   help="answer decisions with the current policy, learn nothing, save nothing")
    p.add_argument("--eval_base_only", type=int, default=0,
                   help="eval the BASE pi0.5 chunk with no Q-selection and no edit — the "
                        "frozen-policy baseline row")

    # --- logging ------------------------------------------------------------
    p.add_argument("--wandb_project", default="expo_ft_franka")
    p.add_argument("--wandb_group", default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    for flag in ("resume", "overwrite", "eval", "eval_base_only", "checkpoint_buffer"):
        setattr(args, flag, bool(getattr(args, flag)))
    if args.eval:
        # An eval that quietly trains, or that starts from random weights because
        # --resume was forgotten, is worse than no eval at all.
        if not args.resume:
            raise SystemExit("--eval needs --resume 1 (and the run_name of the trained run)")
        args.checkpoint_buffer = False
        args.num_data = 0
        args.seed_source = "none"
    if args.offline_ratio != 0 and args.actor_batch_size:
        raise SystemExit(
            "--actor_batch_size only applies with --offline_ratio 0 (the mixed-batch path "
            "sizes its actor pools from --batch_size at construction time)"
        )
    logging.info("EXPO-FT launcher: %s", vars(args))
    train_franka_service.main(args)


if __name__ == "__main__":
    main()
