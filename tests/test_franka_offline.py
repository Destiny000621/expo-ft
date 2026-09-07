"""Offline gates for the Franka EXPO-FT port — no robot, no GPU, no model weights.

These are the checks that catch the failure modes that are invisible on hardware:
an action chunk credited to the wrong decision, a bootstrap discounted as if a
decision were 25 of them, a buffer whose "next observation" is 25 decisions away,
or a config pair whose replan_steps disagree. Everything here runs on CPU in
seconds against the real openpi transforms and the real norm stats.

    JAX_PLATFORMS=cpu pytest tests/test_franka_offline.py -q
"""

import pathlib

import numpy as np
import pytest

from configs.model import expo_ft_franka_config
from configs.task import franka_cable
from expo_ft.data.franka_replay_buffer import FrankaChunkReplayBuffer, build_action_chunks


@pytest.fixture(scope="module")
def task():
    return franka_cable.get_config()


@pytest.fixture(scope="module")
def model_cfg():
    return expo_ft_franka_config.get_config()


@pytest.fixture(scope="module")
def pi05_config(model_cfg):
    from expo_ft.utils.train_utils import build_pi05_config

    _, train_config, _, _ = build_pi05_config(dict(model_cfg))
    return train_config


# ---------------------------------------------------------------- configs


def test_task_and_model_configs_agree(task, model_cfg):
    assert task.action_dim == 10 and task.state_dim == 10, "rot6d10 contract"
    assert task.action_horizon == 50
    assert 0 < task.replan_steps <= task.action_horizon
    # A decision-level buffer bootstraps ONE decision ahead; discounting it as
    # replan_steps env steps would make the critic myopic by a factor of 25.
    assert model_cfg.discount_power == 1
    assert not model_cfg.pi05_use_repack, "live observations arrive in serve keys, not dataset keys"
    assert model_cfg.freeze_pi05_encoder, "N candidate chunks must share one prefix forward"


def test_pi05_train_config_is_the_sft_lineage(pi05_config):
    """The RL config must differ from the SFT config only where it is meant to."""
    import openpi.training.config as _config

    sft = _config.get_config("pi05_franka_double_cable_100_r6_rawrot_wcrop")
    rl = pi05_config
    assert rl.model.action_horizon == sft.model.action_horizon == 50
    assert rl.model.action_dim == sft.model.action_dim == 32
    assert rl.data.state_dim == sft.data.state_dim == 10
    assert rl.data.action_representation == sft.data.action_representation == "rot6d10"
    assert rl.data.normalize_rot6d == sft.data.normalize_rot6d
    assert rl.data.wrist_crop == sft.data.wrist_crop, "the crop box IS the checkpoint's input contract"
    assert "lora" in rl.model.paligemma_variant and "lora" in rl.model.action_expert_variant
    assert rl.ema_decay is None


def test_norm_stats_come_from_the_sft_checkpoint(pi05_config):
    dc = pi05_config.data.create(pi05_config.assets_dirs, pi05_config.model)
    assert dc.norm_stats is not None, "RL must normalize exactly as the SFT did"
    assert len(dc.norm_stats["state"].mean) == 10
    # rot6d dims bypass normalization by identity stats; if that ever silently
    # changed, every rollout would be off-manifold in rotation.
    np.testing.assert_allclose(dc.norm_stats["state"].std[3:9], 1.0)
    np.testing.assert_allclose(dc.norm_stats["state"].mean[3:9], 0.0)


def test_only_lora_params_train(pi05_config):
    import flax.nnx as nnx
    import jax

    abstract = nnx.eval_shape(pi05_config.model.create, jax.random.key(0))
    trainable = nnx.state(abstract, nnx.All(nnx.Param, nnx.Not(pi05_config.freeze_filter))).flat_state()
    llm_trainable = [p for p, _ in trainable.items() if any("llm" in str(x) for x in p)]
    assert llm_trainable, "nothing trainable in the backbone — RL could not move the policy"
    assert all(any("lora" in str(x) for x in p) for p in llm_trainable)


# ------------------------------------------------------------ action chunks


def test_build_action_chunks_stitches_consecutive_decisions():
    """Decision k's chunk must be the rows actually executed from k onward."""
    replan, horizon, dim = 5, 12, 3
    executed = np.arange(4 * replan * dim, dtype=np.float32).reshape(4, replan, dim)
    chunks = build_action_chunks(executed, horizon, replan)
    assert chunks.shape == (4, horizon, dim)
    flat = executed.reshape(-1, dim)
    # decision 0's chunk = executed rows 0..11 (spanning decisions 0, 1 and part of 2)
    np.testing.assert_array_equal(chunks[0], flat[0:horizon])
    np.testing.assert_array_equal(chunks[1], flat[replan : replan + horizon])
    # the last decision runs off the end of the episode -> tile the final row,
    # which is what upstream's retrospective fill does when it hits a `done`.
    tail = chunks[3]
    np.testing.assert_array_equal(tail[:replan], flat[15:20])
    np.testing.assert_array_equal(tail[replan:], np.repeat(flat[-1:], horizon - replan, axis=0))


def test_build_action_chunks_rejects_wrong_stride():
    with pytest.raises(ValueError, match="replan_steps"):
        build_action_chunks(np.zeros((3, 7, 10), np.float32), 50, 25)


# ----------------------------------------------------------------- buffer


def _buffer(pi05_config, task, capacity=64):
    return FrankaChunkReplayBuffer(
        example_action=np.zeros((task.action_dim,), np.float32),
        capacity=capacity,
        pi_train_config=pi05_config,
        resize_size=224,
        task_description=task.language_instruction,
        discount=0.99,
    )


def _decision(task, rng, state_bias=0.0):
    state = np.zeros((task.state_dim,), np.float32)
    state[:3] = [0.5 + state_bias, 0.0, 0.35]
    state[3:9] = [1, 0, 0, 0, 1, 0]  # identity rotation, rot6d
    state[9] = 0.4
    return {
        "observation/image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 255, (180, 320, 3), dtype=np.uint8),
        "observation/state": state,
        "prompt": task.language_instruction,
    }


def test_buffer_stride_is_one_decision(pi05_config, task):
    with pytest.raises(ValueError, match="stride must be 1"):
        FrankaChunkReplayBuffer(
            example_action=np.zeros((10,), np.float32),
            capacity=8,
            pi_train_config=pi05_config,
            resize_size=224,
            task_description="x",
            replan_steps=25,
        )


def test_buffer_rejects_a_single_action_row(pi05_config, task):
    buf = _buffer(pi05_config, task)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="action chunk of shape"):
        buf.insert(
            {
                "observations": _decision(task, rng),
                "actions": np.zeros((task.action_dim,), np.float32),
                "rewards": np.float32(0),
                "masks": np.float32(1),
                "dones": False,
            }
        )


def test_episode_roundtrip_through_the_buffer(pi05_config, task):
    """Insert a 4-decision success episode and check what the critic will see."""
    buf = _buffer(pi05_config, task)
    rng = np.random.default_rng(1)
    n, replan = 4, task.replan_steps
    executed = np.tile(
        np.array([0.5, 0.0, 0.35, 1, 0, 0, 0, 1, 0, 0.4], np.float32), (n, replan, 1)
    )
    executed[..., 0] += np.arange(n)[:, None] * 0.01  # a little motion in x
    chunks = build_action_chunks(executed, task.action_horizon, replan)

    rows = []
    for i in range(n):
        last = i == n - 1
        rows.append(
            buf.insert(
                {
                    "observations": _decision(task, rng, state_bias=0.01 * i),
                    "actions": chunks[i],
                    "rewards": np.float32(task.success_reward if last else 0.0),
                    "masks": np.float32(0.0 if last else 1.0),
                    "dones": bool(last),
                    "is_success": True,
                }
            )
        )
    assert len(buf) == n
    assert rows[0]["actions"].shape == (task.action_horizon, pi05_config.model.action_dim)
    assert rows[0]["base_image"].shape == (224, 224, 3)
    # Sparse terminal reward, and the terminal decision does not bootstrap.
    assert float(buf.dataset_dict["rewards"][n - 1]) == pytest.approx(task.success_reward)
    assert float(buf.dataset_dict["masks"][n - 1]) == 0.0
    assert bool(buf.dataset_dict["dones"][n - 1])

    batch = buf.sample_jax(8)
    assert batch["rewards"].shape == (8,)
    # valids must exist even though the n-step loop never runs at stride 1 —
    # the critic loss multiplies by it.
    np.testing.assert_allclose(np.asarray(batch["valids"]), 1.0)
    assert batch["next_state"].shape == batch["state"].shape


def test_insert_row_restores_identically(pi05_config, task):
    """The resume path re-inserts preprocessed rows; they must be the same rows."""
    buf = _buffer(pi05_config, task)
    rng = np.random.default_rng(2)
    chunk = np.tile(np.array([0.5, 0, 0.35, 1, 0, 0, 0, 1, 0, 0.4], np.float32),
                    (task.action_horizon, 1))
    row = buf.insert(
        {
            "observations": _decision(task, rng),
            "actions": chunk,
            "rewards": np.float32(1.0),
            "masks": np.float32(0.0),
            "dones": True,
            "is_success": True,
        }
    )
    restored = _buffer(pi05_config, task)
    restored.insert_row(row)
    for key in ("state", "actions", "base_image", "left_wrist_image", "tokenized_prompt"):
        np.testing.assert_array_equal(
            np.asarray(buf.dataset_dict[key][0]), np.asarray(restored.dataset_dict[key][0])
        )


def test_absolute_chunk_survives_the_normalize_unnormalize_round_trip(pi05_config, task):
    """The delta/absolute contract, end to end.

    The robot is sent ABSOLUTE targets. The buffer stores them as deltas against
    the transition's own state (`DeltaActions` on dims 0-8), and the learner's
    output transform must add that state back before anything is executed. This
    replays exactly that: insert -> read the stored normalized chunk -> run the
    OUTPUT pipeline the learner runs, anchored on the same state -> expect the
    original absolute chunk back.

    Anchoring on zeros instead (upstream's `process_transformed_outputs` default,
    which is harmless for DROID's cartesian VELOCITY actions) would return a chunk
    anchored at the origin — a plausible-looking chunk that drives the arm across
    the table. That is the failure this gate exists for.
    """
    import openpi.transforms as _transforms

    buf = _buffer(pi05_config, task)
    rng = np.random.default_rng(3)
    obs = _decision(task, rng)
    state = obs["observation/state"].copy()
    absolute = np.tile(state.astype(np.float32), (task.action_horizon, 1))
    absolute[:, 0] += np.linspace(0.0, 0.05, task.action_horizon)  # a 5 cm move in x
    absolute[:, 2] -= np.linspace(0.0, 0.02, task.action_horizon)

    row = buf.insert(
        {
            "observations": obs,
            "actions": absolute.copy(),
            "rewards": np.float32(0.0),
            "masks": np.float32(1.0),
            "dones": False,
        }
    )

    dc = buf._data_config  # noqa: SLF001
    output_pipeline = _transforms.compose(
        [
            *dc.model_transforms.outputs,
            _transforms.Unnormalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
            *dc.data_transforms.outputs,
        ]
    )
    recovered = output_pipeline(
        {
            "state": np.asarray(row["state"], np.float32),      # normalized, padded
            "actions": np.asarray(row["actions"], np.float32),  # normalized, padded
        }
    )["actions"]
    np.testing.assert_allclose(recovered, absolute, rtol=0, atol=2e-3)

    # And the zero-anchored version must NOT come back right, or this gate proves
    # nothing about the anchoring.
    wrong = output_pipeline(
        {
            "state": np.zeros_like(np.asarray(row["state"], np.float32)),
            "actions": np.asarray(row["actions"], np.float32),
        }
    )["actions"]
    assert np.abs(wrong[:, :3] - absolute[:, :3]).max() > 0.05, (
        "anchoring on zeros produced the same chunk — this dataset has no delta "
        "actions, so the round-trip gate is vacuous"
    )


# ------------------------------------------------------------- agent shapes


def test_learner_networks_build_with_the_right_shapes(pi05_config, task, model_cfg):
    """Build the critic / edit actor / encoder on CPU with a STUB VLA.

    This is the cheap version of "did the port wire up": it exercises the mesh,
    the example arrays taken off the buffer, and every dimension the algorithm
    derives from them — without loading 3B parameters. The numbers asserted here
    are the ones a shape bug would silently change, and which would then only
    show up as a critic that scores something other than what was executed.
    """
    import jax

    import openpi.training.sharding as sh
    from expo_ft.agents.alg.expo_ft import load_agent
    from expo_ft.utils.train_utils import build_pi05_config

    agent_kwargs, _, _, _ = build_pi05_config(dict(model_cfg))
    freeze = agent_kwargs.pop("freeze_pi05_encoder")
    agent_kwargs.pop("pi05_use_repack", None)

    _mesh = sh.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(_mesh, jax.sharding.PartitionSpec(sh.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(_mesh, jax.sharding.PartitionSpec())

    buf = _buffer(pi05_config, task, capacity=4)
    obs, state, act = buf.convert_to_critic_format(
        {
            "base_image": buf.dataset_dict["base_image"][0],
            "left_wrist_image": buf.dataset_dict["left_wrist_image"][0],
            "state": buf.dataset_dict["state"][0],
            "actions": buf.dataset_dict["actions"][0],
        }
    )
    # Two REAL cameras channel-concatenated at the model's own resolution; the
    # zero right-wrist slot pi0.5 pads with is deliberately not fed to the critic.
    assert obs.shape == (224, 224, 6)
    assert state.shape == (pi05_config.model.action_dim,)   # padded state
    assert act.shape == (task.action_horizon, task.action_dim)

    class _StubActor:
        infer_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        model_config = pi05_config.model
        mesh = _mesh
        action_dim = task.action_dim
        state_dim = pi05_config.model.action_dim

    agent = load_agent(
        seed=0, example_observation=obs, example_action=act, example_state=state,
        actor=_StubActor(), actor_train_state=None, target_actor_params=None,
        agent_kwargs=agent_kwargs,
        metadata=dict(action_horizon=task.action_horizon, resize_size=224, freeze_encoder=freeze),
        mesh=_mesh, data_sharding=data_sharding, replicated_sharding=replicated,
        resume=True,  # skip cache_infer_params: it needs a real pi0.5 train state
        replan_steps=task.replan_steps, default_prompt=task.language_instruction,
        residual_action_xyzg=task.residual_action_xyzg,
    )
    # The critic scores exactly the rows the arm executes: replan_steps x action_dim.
    assert agent.full_action_dim == task.replan_steps * task.action_dim == 250
    assert agent.action_dim == task.action_dim
    assert agent.replan_steps == task.replan_steps
    assert agent.discount_power == 1 and agent.discount == pytest.approx(0.99)
    assert agent.N == 8 and agent.n_edit_samples == 8
    # The edit policy emits ONE offset per chunk (chunk_offset), so its entropy
    # target is sized to action_dim, not to the 250-wide chunk the critic scores.
    assert agent.target_entropy == pytest.approx(-task.action_dim / 2)


# ---------------------------------------------------------- rollout seeding

ROLLOUT_DIRS = [
    pathlib.Path("~/Desktop/Haply_Franka/data_log_eval_wcrop").expanduser(),
    pathlib.Path("/mnt/localssd/Sichang/seed/data_log_eval_wcrop"),
    pathlib.Path("~/expo_seed/data_log_eval_wcrop").expanduser(),
    pathlib.Path("~/stage_expo/seed/data_log_eval_wcrop").expanduser(),
]


def _rollout_root():
    for p in ROLLOUT_DIRS:
        if p.exists():
            return p
    pytest.skip("no recorded rollouts on this box")


def test_rollout_seed_matches_the_live_wire(task):
    """One decision out of a recorded rollout must look like a live decision.

    The buffer cannot tell seeded data from online data, so anything that differs
    here is a silent distribution shift the critic will learn as signal: a gripper
    column in the wrong units, a side frame resampled differently than the client
    resamples it, a pre-cropped wrist, or a rotation that is not a rotation.
    """
    import itertools

    from expo_ft.env.franka_rollout_seed import find_rollout_episodes, process_franka_rollouts

    root = _rollout_root()
    episodes = find_rollout_episodes(root, include_failures=True)
    assert episodes, "no episodes found under the rollout root"
    assert any(s for _, s in episodes), "no SUCCESS episodes — nothing for the actor's BC pool"

    transitions = list(
        itertools.islice(
            process_franka_rollouts(
                root, task, action_horizon=task.action_horizon, num_episodes=1, include_failures=True
            ),
            4,
        )
    )
    assert transitions, "the first rollout yielded no decisions"
    tr = transitions[0]
    obs = tr["observations"]

    # The side view is pad-resized to 224 by the CLIENT; the wrist goes raw,
    # because the wrist crop is a learner-side transform.
    assert obs["observation/image"].shape == (224, 224, 3)
    assert obs["observation/image"].dtype == np.uint8
    assert obs["observation/wrist_image"].shape[0] > 224, "wrist frame must stay uncropped/unresized"

    state = obs["observation/state"]
    assert state.shape == (task.state_dim,)
    # Gripper in KNUCKLE RADIANS, not the recorder's 0-1 position: the checkpoint's
    # own norm stats have q99 = 0.7263 on this column.
    assert 0.0 <= float(state[9]) <= 0.7929 + 1e-3
    # rot6d must be an orthonormal frame, or Gram-Schmidt on the robot side
    # silently invents a different rotation.
    cols = np.stack([state[3:6], state[6:9]], axis=1)
    np.testing.assert_allclose(np.linalg.norm(cols, axis=0), 1.0, atol=1e-3)
    assert abs(float(cols[:, 0] @ cols[:, 1])) < 1e-3

    chunk = tr["actions"]
    assert chunk.shape == (task.action_horizon, task.action_dim)
    assert 0.0 <= float(chunk[0, 9]) <= 0.7929 + 1e-3
    # The recorded command LEADS the measured pose (that lead times the impedance
    # spring is what produced insertion force in the demos). A few mm to a few cm
    # is right; metres would mean state and action came from different frames.
    lead_mm = float(np.linalg.norm(chunk[0, :3] - state[:3])) * 1000.0
    assert 0.1 < lead_mm < 150.0, f"command-vs-measured lead {lead_mm:.1f} mm is not plausible"

    # Non-terminal decisions bootstrap; only the last one of an episode does not.
    assert float(tr["masks"]) == 1.0 and not tr["dones"]
    assert float(tr["rewards"]) == 0.0


def test_seeded_rollout_transition_enters_the_buffer(pi05_config, task):
    """A seeded decision must survive the same transforms an online one does."""
    import itertools

    from expo_ft.env.franka_rollout_seed import process_franka_rollouts

    root = _rollout_root()
    buf = _buffer(pi05_config, task, capacity=8)
    for tr in itertools.islice(
        process_franka_rollouts(root, task, action_horizon=task.action_horizon, num_episodes=1), 3
    ):
        row = buf.insert(tr)
    assert len(buf) == 3
    assert row["base_image"].shape == (224, 224, 3)
    assert row["actions"].shape == (task.action_horizon, pi05_config.model.action_dim)
    batch = buf.sample_jax(4)
    assert batch["state"].shape == (4, pi05_config.model.action_dim)


def test_uncommanded_ticks_are_hold_filled(task):
    """Rollouts contain ticks the agent did not command; they must not cost an episode.

    The recorder writes NaN for `active=False` ticks (pi0.5 waiting for its next
    chunk, or a stale pose stream) — 15-150 per episode in the 2026-09-05 set. scipy
    rejects a NaN quaternion outright, so without hold-filling, six of thirty-eight
    rollouts silently dropped out of the seed.
    """
    import itertools

    from expo_ft.env.franka_rollout_seed import _hold_fill, find_rollout_episodes, process_franka_rollouts

    rows = np.array([[np.nan, np.nan], [1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]], np.float32)
    filled = _hold_fill(rows, "test", "ep")
    np.testing.assert_array_equal(filled[0], [1.0, 2.0])   # leading run: back-filled
    np.testing.assert_array_equal(filled[2], [1.0, 2.0])   # interior: held
    np.testing.assert_array_equal(filled[3], [3.0, 4.0])
    assert np.isfinite(filled).all()

    root = _rollout_root()
    episodes = find_rollout_episodes(root, include_failures=True)
    dirty = [
        (ep, s) for ep, s in episodes
        if not np.isfinite(np.load(ep / "arm0_actions.npz")["target_pose"]).all()
    ]
    if not dirty:
        pytest.skip("no rollout with uncommanded ticks on this box")
    ep, success = dirty[0]
    transitions = list(
        itertools.islice(
            process_franka_rollouts(
                ep.parent, task, action_horizon=task.action_horizon, num_episodes=1,
                include_failures=True,
            ),
            3,
        )
    )
    assert transitions, f"{ep.name} yielded nothing despite hold-filling"
    for tr in transitions:
        assert np.isfinite(tr["actions"]).all()
        assert np.isfinite(tr["observations"]["observation/state"]).all()


def test_live_observation_survives_the_input_pipeline(pi05_config, task, model_cfg):
    """The exact dict the robot sends must pass the transforms the learner applies.

    This is the decision path's front half, and it is where a shape mismatch shows up
    as a crash at the FIRST live decision rather than in any offline check: the wire
    carries no actions, the learner injects a dummy one, and a delta-action config
    subtracts the state from it. A 1-D dummy (upstream's) cannot broadcast against a
    chunk.
    """
    import jax

    import openpi.training.sharding as sh
    from expo_ft.agents.vla.pi05 import Pi05Agent

    mesh = sh.make_mesh(1)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    # Constructed directly: this exercises the transform pipeline without loading
    # 3B parameters (initialize() would).
    actor = Pi05Agent(
        train_config=pi05_config,
        mesh=mesh,
        train_state_sharding=replicated,
        data_sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sh.DATA_AXIS)),
        replicated_sharding=replicated,
        default_prompt=task.language_instruction,
        use_repack=bool(model_cfg.pi05_use_repack),
    )
    # The service sets these from the buffer's example arrays after construction.
    actor.action_dim = task.action_dim
    actor.state_dim = pi05_config.model.action_dim
    rng = np.random.default_rng(0)
    state = np.zeros((task.state_dim,), np.float32)
    state[:3] = [0.5, 0.0, 0.35]
    state[3:9] = [1, 0, 0, 0, 1, 0]
    observation = {
        "observation/image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        # RAW wrist frame, as the client sends it — the learner applies the crop.
        "observation/wrist_image": rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8),
        "observation/state": state,
        "prompt": task.language_instruction,
    }
    processed = actor.process_raw_inputs(observation, task.action_dim, 224)
    assert processed["state"].shape == (1, pi05_config.model.action_dim)
    assert processed["image"]["base_0_rgb"].shape == (1, 224, 224, 3)
    assert processed["image"]["left_wrist_0_rgb"].shape == (1, 224, 224, 3)
    assert processed["tokenized_prompt"].shape[0] == 1

    # And the back half: model-space actions -> absolute chunk, anchored on that state.
    fake = np.zeros((2, pi05_config.model.action_horizon, task.action_dim), np.float32)
    out = actor.process_transformed_outputs(fake, state=processed["state"])
    assert out.shape == (2, pi05_config.model.action_horizon, task.action_dim)
    # Zero normalized actions do NOT decode to zero: they unnormalize and then get
    # the current pose added back, so the chunk must land near the robot.
    assert np.linalg.norm(out[0, 0, :3] - state[:3]) < 0.5


def test_chunk_offset_residual_keeps_the_chunk_smooth(pi05_config, task, model_cfg):
    """The edit must shift a chunk, not shred it.

    Upstream's per-row residual, applied to absolute 30 Hz position chunks, produced
    targets that stepped 30-40 mm per row (live, 2026-09-06). With "chunk_offset"
    + xyz-only, an edited chunk is the base chunk plus ONE constant xyz offset:
    identical row-to-row motion, untouched rotation and gripper.
    """
    import jax
    import jax.numpy as jnp

    import openpi.training.sharding as sh
    from expo_ft.agents.alg.expo_ft import load_agent
    from expo_ft.utils.train_utils import build_pi05_config

    agent_kwargs, _, _, _ = build_pi05_config(dict(model_cfg))
    freeze = agent_kwargs.pop("freeze_pi05_encoder")
    agent_kwargs.pop("pi05_use_repack", None)
    _mesh = sh.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(_mesh, jax.sharding.PartitionSpec(sh.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(_mesh, jax.sharding.PartitionSpec())
    buf = _buffer(pi05_config, task, capacity=4)
    obs, state, act = buf.convert_to_critic_format(
        {k: buf.dataset_dict[k][0] for k in ("base_image", "left_wrist_image", "state", "actions")}
    )

    class _StubActor:
        infer_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        model_config = pi05_config.model
        mesh = _mesh
        action_dim = task.action_dim
        state_dim = pi05_config.model.action_dim

    agent = load_agent(
        seed=0, example_observation=obs, example_action=act, example_state=state,
        actor=_StubActor(), actor_train_state=None, target_actor_params=None,
        agent_kwargs=agent_kwargs,
        metadata=dict(action_horizon=task.action_horizon, resize_size=224, freeze_encoder=freeze),
        mesh=_mesh, data_sharding=data_sharding, replicated_sharding=replicated, resume=True,
        replan_steps=task.replan_steps, default_prompt=task.language_instruction,
        residual_action_xyzg=task.residual_action_xyzg,
    )
    assert agent.residual_mode == "chunk_offset"
    assert agent.residual_dims == (0, 1, 2)
    # The edit policy outputs ONE action_dim vector, and the entropy target is sized to it.
    assert agent.target_entropy == pytest.approx(-task.action_dim / 2)

    r = jnp.asarray(np.array([[1.0, -0.5, 0.25, 9, 9, 9, 9, 9, 9, 9]], np.float32))  # rot/grip = poison
    expanded = np.asarray(agent._expand_residual(r)).reshape(task.replan_steps, task.action_dim)
    np.testing.assert_allclose(expanded[:, :3], np.tile([[1.0, -0.5, 0.25]], (task.replan_steps, 1)))
    assert np.all(expanded[:, 3:] == 0.0), "rot6d / gripper must be untouched by the edit"
    # constant across rows -> the residual adds no row-to-row motion at all
    assert np.abs(np.diff(expanded, axis=0)).max() == 0.0

    # And a sampled edit through the real distribution has the same shape.
    key = jax.random.PRNGKey(0)
    dist = agent.residual_actor.apply_fn(
        {"params": agent.residual_actor.params},
        jnp.ones((2, agent_kwargs["latent_dim_image"])),
        actions=jnp.zeros((2, agent.full_action_dim)),
        p=jnp.zeros((2, pi05_config.model.action_dim)),
    )
    sample = dist.sample(seed=key)
    assert sample.shape == (2, task.action_dim)
