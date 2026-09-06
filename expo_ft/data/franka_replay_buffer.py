"""Decision-level replay buffer for the Franka port.

Upstream stores one transition per ENV STEP and rebuilds each transition's action
chunk retrospectively: when step t+k is inserted, its executed action is written
into row k of the chunk belonging to step t (`PiReplayBuffer.insert`). That works
for DROID because its actions are cartesian *velocities* — the same number means
the same thing no matter which transition's chunk it lands in.

Two things break that here:

1. **Granularity.** avantbot runs the arm at 30 Hz; a decision covers 25 ticks.
   Storing every tick would mean shipping two camera frames to the learner 30x a
   second and holding ~2700 transitions per episode. One transition per DECISION
   is what the agent actually chose, and it is what the critic is asked about.
2. **Delta actions.** Our data config re-anchors a chunk on the state of the
   transition it belongs to (`DeltaActions` on dims 0-8). A normalized row written
   into a NEIGHBOUR's chunk would be a delta against the wrong pose — a silent,
   plausible-looking corruption of exactly the geometry the task is about.

So this buffer takes the whole action chunk up front, in RAW (absolute) space, and
transforms it once against its own observation. The caller assembles that chunk
from consecutive decisions, which is the same quantity upstream builds
retrospectively — just computed before the transform instead of after it.

`stride=1` (see __init__) tells `sample_jax` that the NEXT transition is the next
decision. The agent's own `replan_steps` (the chunk width the critic scores) stays
25 and is unaffected; the two numbers coincide upstream and must not here.
"""

from typing import Any, Dict

import numpy as np

from expo_ft.data.replay_buffer import PiReplayBuffer, _insert_recursively


class FrankaChunkReplayBuffer(PiReplayBuffer):
    """PiReplayBuffer whose transitions are decisions carrying a full action chunk."""

    def __init__(self, *args, **kwargs):
        # PiReplayBuffer uses `replan_steps` for BOTH the n-step reward accumulation
        # and the next-observation offset in sample_jax. At decision granularity both
        # are 1: the reward of a decision is already the reward of its 25 ticks, and
        # the next observation is the next decision's.
        kwargs.setdefault("replan_steps", 1)
        if kwargs["replan_steps"] != 1:
            raise ValueError(
                "FrankaChunkReplayBuffer stores one transition per decision, so its "
                f"sampling stride must be 1 (got {kwargs['replan_steps']}). The chunk "
                "width the critic scores is the AGENT's replan_steps, not this."
            )
        super().__init__(*args, **kwargs)

    def _build_transform_pipeline(self):
        """Same pipeline as upstream, minus the dataset REPACK.

        Repack renames LeRobot dataset columns ("observation.images.camera1", ...)
        into the model's input keys. Everything that reaches this buffer — live
        decisions from the robot and seeded demos alike — is already in the model's
        keys, because that is the wire openpi's own serving path speaks. Running
        repack here would look for dataset columns that do not exist.
        """
        import openpi.transforms as _transforms  # noqa: PLC0415

        if self._skip_norm_stats or self._data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats are required: RL must normalize exactly as the SFT "
                "did, or the policy is being steered in a different space than it was "
                "trained in. Point the TrainConfig's assets at the SFT checkpoint."
            )
        return _transforms.compose(
            [
                *self._data_config.data_transforms.inputs,
                _transforms.Normalize(
                    self._data_config.norm_stats, use_quantiles=self._data_config.use_quantile_norm
                ),
                *self._data_config.model_transforms.inputs,
            ]
        )

    def insert(self, data_dict: Dict[str, Any]):
        """Insert one decision.

        Expects ``data_dict["actions"]`` to be the full ``(action_horizon, raw_dim)``
        chunk of ABSOLUTE actions this decision is credited with — the rows the arm
        actually executed, continued by the rows the following decisions executed,
        tiled with the last row at the end of an episode.
        """
        actions = np.asarray(data_dict["actions"], dtype=np.float32)
        if actions.shape != (self._action_horizon, self._raw_action_dim):
            raise ValueError(
                f"expected an action chunk of shape {(self._action_horizon, self._raw_action_dim)}, "
                f"got {actions.shape}. Decision-level transitions carry the whole chunk; "
                "there is no retrospective fill in this buffer."
            )

        obs_data_dict = dict(data_dict["observations"])
        obs_data_dict["actions"] = actions
        obs_data_dict["rewards"] = np.asarray(data_dict["rewards"], dtype=np.float32)
        obs_data_dict["masks"] = np.asarray(data_dict["masks"], dtype=np.float32)
        obs_data_dict["dones"] = np.asarray(data_dict["dones"], dtype=bool)
        is_hil = bool(data_dict.get("is_hil", False))
        obs_data_dict["is_hil"] = np.asarray(is_hil, dtype=bool)
        # hil_chunk marks a chunk that CONTAINS human actions; at decision
        # granularity a decision is human or it is not, so the two coincide.
        obs_data_dict["hil_chunk"] = np.asarray(is_hil, dtype=bool)
        obs_data_dict["is_success"] = np.asarray(data_dict.get("is_success", False), dtype=bool)

        preprocessed = self._preprocess_single_transition(obs_data_dict)
        row = {k: preprocessed[k] for k in self.dataset_dict.keys()}
        self.insert_row(row)
        # Returned so the caller can persist the PREPROCESSED row (224 px, already
        # normalized and tokenized). Persisting the raw transition instead would
        # mean storing a 720p wrist frame per decision — the wrist crop is applied
        # learner-side, so the raw frame is what a re-insert would need.
        return row

    def insert_row(self, row: Dict[str, Any]) -> None:
        """Insert an already-preprocessed row (restore path; no transforms run)."""
        _insert_recursively(self.dataset_dict, row, self._insert_index)
        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)


def build_action_chunks(executed: np.ndarray, action_horizon: int, replan_steps: int) -> np.ndarray:
    """Turn an episode's executed rows into one action chunk per decision.

    ``executed`` is ``(n_decisions, replan_steps, raw_dim)`` — what each decision
    actually put on the wire. Decision k's chunk is the next ``action_horizon``
    executed rows starting at k's first row; past the end of the episode the last
    row is tiled, which is upstream's convention for a chunk that runs off the end
    (`PiReplayBuffer.insert` stops filling at a `done`).

    Returns ``(n_decisions, action_horizon, raw_dim)``.
    """
    executed = np.asarray(executed, dtype=np.float32)
    n_dec, k, dim = executed.shape
    if k != replan_steps:
        raise ValueError(f"executed rows per decision is {k}, expected replan_steps={replan_steps}")
    flat = executed.reshape(n_dec * k, dim)
    tail = np.repeat(flat[-1:], action_horizon, axis=0)
    padded = np.concatenate([flat, tail], axis=0)
    starts = np.arange(n_dec) * k
    return np.stack([padded[s : s + action_horizon] for s in starts], axis=0)
