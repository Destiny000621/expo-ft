"""EXPO-FT algorithm config for the Franka double-cable task.

Derived from `expo_ft_pi_config` (the paper's DROID/pi0.5 recipe). Everything that
differs from it is either a SHAPE forced by our checkpoint, or a knob resized for a
~100-real-episode budget instead of a long DROID run. Each such change is
justified inline, so a reader can tell "port" from "tuning".
"""

from configs.model import expo_ft_pi_config


def get_config():
    config = expo_ft_pi_config.get_config()

    # --- the VLA ------------------------------------------------------------
    # LoRA adapters on top of the wcrop SFT checkpoint; the config also carries the
    # SFT's own baked norm stats, so RL normalizes exactly as the frozen baseline
    # did. Paths are overridable from the launcher (--config.pi05_* ) because the
    # checkpoint sits at a different path on the learner box than on the station.
    config.pi05_config_name = "expo_pi05_franka_double_cable_r6_wcrop_lora"
    config.pi05_weight_loader_path = ""  # "" -> use the TrainConfig's own loader
    config.pi05_assets_dir = ""
    config.pi05_asset_id = ""
    config.pi05_resize_size = 224
    # The robot speaks openpi's SERVE wire ("observation/image", ...), not LeRobot
    # dataset keys, so the dataset repack must not run on live observations.
    config.pi05_use_repack = False
    # Frozen encoder for sampling: embed the prefix ONCE and draw N chunks off the
    # shared KV cache (openpi Franka_EXPO branch). Without it, N=8 candidates cost
    # 8 vision forwards per replan and the robot waits ~8x longer at every decision.
    config.freeze_pi05_encoder = True
    config.freeze_critic_encoder = False

    # --- EXPO-FT core -------------------------------------------------------
    # Upstream's values, kept: 8 base samples + 8 edited samples per decision, edit
    # scale 0.2 in NORMALIZED action space (the README's recommended starting point).
    config.N = 8
    config.n_edit_samples = 8
    config.edit_scale = 0.2
    config.actor_success_only = True

    # --- discounting --------------------------------------------------------
    # One transition in this port's buffer is one DECISION (25 control ticks),
    # not one control tick, so the critic's bootstrap must be raised to the power
    # 1, not to replan_steps. `discount` below is therefore per DECISION.
    # 0.99 per decision -> horizon ~100 decisions ~= one 90 s episode, which is the
    # right scale for a reward that only arrives at the end. (Upstream's 0.99 per
    # 10 Hz env step would be 0.78 per decision here: horizon ~4 decisions, i.e.
    # blind to the only reward the task has.)
    config.discount = 0.99
    config.discount_power = 1

    # --- critic -------------------------------------------------------------
    # Upstream's ensemble and encoder, unchanged (REDQ-10 with min-of-2 subsample).
    config.num_qs = 10
    config.num_min_qs = 2
    config.critic_layer_norm = True
    config.latent_dim_image = 512
    config.latent_dim_state = 64
    config.encoder_stage_sizes = (3, 4, 6, 3)
    config.encoder_num_filters = 64
    config.hidden_dims = (256, 256, 256)
    # The critic sees the two REAL cameras concatenated on the channel axis at the
    # model's own 224 resolution (6 channels: side + wrist). The zero right-wrist
    # slot pi0.5 pads with is deliberately NOT fed to the critic.
    config.include_state = True

    # Crop-only augmentation. Full augmentation (rotate + colour jitter) fights the
    # wrist crop this checkpoint depends on: the policy was trained on a fixed
    # fractional box, and rotating it moves the port out of the frame the critic is
    # supposed to be judging.
    config.use_full_augmentation = False

    # Batch splitting is a multi-GPU memory knob; one learner GPU here.
    config.encode_batch_split = 1
    config.batch_split = 1

    return config
