# Environment for the EXPO-FT learner box (source, don't execute).
#
#   source scripts/franka/learner_env.sh
#   bash scripts/franka/run_franka.sh
#
# Everything here is about WHERE things live on a machine that is not the robot
# station. Nothing about the algorithm changes.

# openpi's download cache. On H200-5 the default (~/.cache/openpi) is a symlink
# onto the ephemeral local SSD, which is wiped on every restart — if that target
# is missing, norm-stat loading dies with a FileNotFoundError that looks like a
# config bug and is not one. Point it somewhere that exists.
# One root for everything the learner box holds. Relocating the whole run is
# then one variable: EXPO_ROOT=/mnt/localssd/Sichang source scripts/franka/learner_env.sh
export EXPO_ROOT="${EXPO_ROOT:-$HOME/expo}"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$EXPO_ROOT/openpi_cache}"
export HF_HOME="${HF_HOME:-$EXPO_ROOT/hf}"

# The pi0.5 SFT checkpoint RL starts from and normalizes with. The TrainConfig's
# default is HOME-relative (~/.cache/openpi/hf/...), which resolves on the station;
# set this when the checkpoint sits elsewhere on the learner box.
export EXPO_PI05_SFT_CKPT="${EXPO_PI05_SFT_CKPT:-$EXPO_ROOT/openpi_cache_hf/pi05_franka_double_cable_100_wcrop_10k}"

# Recorded pi0.5 rollouts used to warm-start the buffer (rsync'd from the station's
# Haply_Franka/data_log_eval_wcrop).
export ROLLOUT_DIR="${ROLLOUT_DIR:-$EXPO_ROOT/seed/data_log_eval_wcrop}"

export EXP="${EXP:-$EXPO_ROOT/logs/expo_franka}"

# One GPU is plenty (an H200 is 141 GB and the learner's own nets are 38 M params
# on top of pi0.5); pin it so a second experiment can share the box.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Report honest VRAM in nvidia-smi instead of JAX's 75% preallocation.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
