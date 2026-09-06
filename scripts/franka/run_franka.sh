#!/usr/bin/env bash
# EXPO-FT learner for the Franka double-cable task (training).
#
# Run this on the learner box. The robot side is a separate machine (or the same
# one) running:
#   cd ~/Desktop/Haply_Franka/vendor/avantbot && pixi shell -e droid-openpi
#   python -m avantbot.collect --config policy/franka_pi05_ee_fr3_expo
#
# There is NO openpi serve in this baseline: EXPO-FT fine-tunes the policy weights
# online, so the learner owns pi0.5 and answers every decision itself.
#
# Remote learner: forward the port from the ROBOT machine, then leave the session
# YAML's learner_url at 127.0.0.1:
#   ssh -N -L 9112:localhost:9112 <learner-host>
set -euo pipefail
cd "$(dirname "$0")/../.."

source .venv/bin/activate

EXP=${EXP:-$(pwd)/logs/expo_franka}
RUN=${RUN:-expo_franka_cable_seed0}

python -m launch_train_franka \
    --config configs/model/expo_ft_franka_config.py \
    --config_task configs/task/franka_cable.py \
    --port "${PORT:-9112}" \
    --output_dir "$EXP" \
    --run_name "$RUN" \
    --seed_source "${SEED_SOURCE:-rollouts}" \
    --rollout_seed_dir "${ROLLOUT_DIR:-$HOME/expo_seed/data_log_eval_wcrop}" \
    --rollout_include_failures "${INCLUDE_FAILURES:-1}" \
    --num_data "${NUM_DATA:-0}" \
    --seed_cache "$EXP/seed_cache.pkl" \
    --batch_size 64 \
    --actor_batch_size 16 \
    --utd_ratio 20 \
    --num_updates 4 \
    --checkpoint_interval_episodes 5 \
    --wandb_project expo_ft_franka \
    "$@"
