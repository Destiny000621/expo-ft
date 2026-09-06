"""Offline gates for the Franka EXPO-FT port — no robot, no GPU, no model weights.

These are the checks that catch the failure modes that are invisible on hardware:
an action chunk credited to the wrong decision, a bootstrap discounted as if a
decision were 25 of them, a buffer whose "next observation" is 25 decisions away,
or a config pair whose replan_steps disagree. Everything here runs on CPU in
seconds against the real openpi transforms and the real norm stats.

    JAX_PLATFORMS=cpu pytest tests/test_franka_offline.py -q
"""

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
    assert agent.target_entropy == pytest.approx(-agent.full_action_dim / 2)
