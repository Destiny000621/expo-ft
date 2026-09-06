#!/usr/bin/env bash
# EXPO-FT evaluation: answer decisions with a trained run's policy, learn nothing.
#
# The learner is what makes a session an eval — the robot side only points its
# recordings somewhere else:
#   python -m avantbot.collect --config policy/franka_pi05_ee_fr3_expo_eval
#
# The running success tally the learner prints per episode IS the eval.
# EVAL_BASE_ONLY=1 evaluates the BASE pi0.5 chunk with no Q-selection and no edit
# — the frozen-policy baseline row to compare the trained policy against.
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
    --eval 1 \
    --resume 1 \
    --eval_base_only "${EVAL_BASE_ONLY:-0}" \
    --wandb_project "" \
    "$@"
