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
# Default: the local-SSD root on H200-5 when it exists, else $HOME/expo.
if [ -z "${EXPO_ROOT:-}" ] && [ -d /mnt/localssd/Sichang ]; then
    EXPO_ROOT=/mnt/localssd/Sichang
fi
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

# FOUR GPUs, data-parallel (openpi's make_mesh(fsdp_devices=1) shards the batch
# across every visible device). Not for memory — one H200 holds the whole learner
# at 33 GB — but for the update block: EXPO-FT's critic target draws 8 pi0.5 chunks
# for every state in every minibatch, and that block measured 132 s on one GPU vs
# 22 s on eight. Four is the budget on this SHARED box (user rule, 2026-09-06);
# the other four stay free for other people's jobs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Report honest VRAM in nvidia-smi instead of JAX's 75% preallocation.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
